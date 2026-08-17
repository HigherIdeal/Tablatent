#!/usr/bin/env python3
"""Lightweight temporal-safe tabular RAG for Qwen3-1.7B.

No embedding model is used. Retrieval is deterministic Python logic over train.csv:
1) only rows from seasons strictly earlier than the query season are eligible;
2) candidates are narrowed by pitcher/count/hand context with progressive fallback;
3) a small set of representative historical examples and empirical rates is returned.

The LLM is the only learned model in this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TARGET = "control_success"

# Stable context columns known to exist in the competition table. Missing columns
# are simply ignored so the retriever remains robust to processed variants.
CONTEXT_COLUMNS = [
    "season",
    "game_month",
    "game_dayofweek",
    "game_type",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state_before",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "score_diff_before",
    "run_top_before",
    "run_bot_before",
    "win_expectancy_before",
    "leverage_index_before",
]

# History/as-of columns are particularly useful to an LLM because their meaning
# is explicit. We include any columns with these prefixes if present.
HISTORY_PREFIXES = ("asof_", "prev1_", "prev3_", "prev5_")


@dataclass
class RetrievalResult:
    name: str
    payload: dict[str, Any]


class TemporalTabularRAG:
    def __init__(self, train: pd.DataFrame, seed: int = 42) -> None:
        if TARGET not in train.columns:
            raise ValueError(f"train frame must contain {TARGET}")
        if "season" not in train.columns:
            raise ValueError("train frame must contain season")
        self.train = train.reset_index(drop=True)
        self.seed = int(seed)
        self._season = pd.to_numeric(self.train["season"], errors="coerce").to_numpy()
        self._target = pd.to_numeric(self.train[TARGET], errors="coerce").to_numpy(dtype=float)
        self._pitcher_groups = self._group_indices("pitcher_id")
        self._batter_groups = self._group_indices("batter_id")

    def _group_indices(self, column: str) -> dict[str, np.ndarray]:
        if column not in self.train.columns:
            return {}
        groups: dict[str, np.ndarray] = {}
        values = self.train[column].astype("string")
        for key, idx in values.groupby(values, dropna=True).groups.items():
            groups[str(key)] = np.asarray(idx, dtype=np.int64)
        return groups

    @staticmethod
    def _scalar(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    def _before(self, indices: np.ndarray, season: int) -> np.ndarray:
        if indices.size == 0:
            return indices
        return indices[self._season[indices] < season]

    def _match(self, indices: np.ndarray, query: pd.Series, columns: list[str]) -> np.ndarray:
        out = indices
        for column in columns:
            if column not in self.train.columns or column not in query.index:
                continue
            value = query[column]
            if pd.isna(value):
                continue
            series = self.train.loc[out, column]
            mask = series.astype("string").to_numpy() == str(value)
            out = out[mask]
            if out.size == 0:
                break
        return out

    def _rate_payload(self, indices: np.ndarray) -> dict[str, Any]:
        y = self._target[indices]
        y = y[np.isfinite(y)]
        if y.size == 0:
            return {"n": 0, "success_rate": None}
        return {
            "n": int(y.size),
            "success_rate": float(y.mean()),
            "successes": int(np.round(y.sum())),
        }

    def pitcher_history(self, query: pd.Series) -> RetrievalResult:
        season = int(query["season"])
        key = str(query.get("pitcher_id"))
        idx = self._before(self._pitcher_groups.get(key, np.empty(0, dtype=np.int64)), season)
        payload = self._rate_payload(idx)
        if idx.size:
            seasons = self._season[idx]
            latest = int(np.nanmax(seasons))
            latest_idx = idx[seasons == latest]
            payload["latest_prior_season"] = latest
            payload["latest_prior_season_stats"] = self._rate_payload(latest_idx)
        return RetrievalResult("pitcher_history", payload)

    def batter_history(self, query: pd.Series) -> RetrievalResult:
        season = int(query["season"])
        key = str(query.get("batter_id"))
        idx = self._before(self._batter_groups.get(key, np.empty(0, dtype=np.int64)), season)
        return RetrievalResult("batter_history", self._rate_payload(idx))

    def context_history(self, query: pd.Series) -> RetrievalResult:
        season = int(query["season"])
        base = np.flatnonzero(self._season < season).astype(np.int64)
        levels = [
            ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand", "base_state_before"],
            ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"],
            ["balls_before", "strikes_before", "outs_before"],
            ["balls_before", "strikes_before"],
        ]
        chosen = base
        used: list[str] = []
        for columns in levels:
            candidate = self._match(base, query, columns)
            if candidate.size >= 100:
                chosen = candidate
                used = [c for c in columns if c in self.train.columns]
                break
        payload = self._rate_payload(chosen)
        payload["matched_on"] = used
        return RetrievalResult("context_history", payload)

    def matchup_history(self, query: pd.Series) -> RetrievalResult:
        season = int(query["season"])
        pitcher_key = str(query.get("pitcher_id"))
        idx = self._before(self._pitcher_groups.get(pitcher_key, np.empty(0, dtype=np.int64)), season)
        idx = self._match(idx, query, ["batter_id"])
        return RetrievalResult("matchup_history", self._rate_payload(idx))

    def similar_examples(self, query: pd.Series, k: int = 12) -> RetrievalResult:
        season = int(query["season"])
        pitcher_key = str(query.get("pitcher_id"))
        pitcher_idx = self._before(self._pitcher_groups.get(pitcher_key, np.empty(0, dtype=np.int64)), season)

        candidate_specs = [
            (pitcher_idx, ["balls_before", "strikes_before", "batter_hand"]),
            (pitcher_idx, ["balls_before", "strikes_before"]),
            (pitcher_idx, []),
        ]
        all_prior = np.flatnonzero(self._season < season).astype(np.int64)
        candidate_specs.extend([
            (all_prior, ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"]),
            (all_prior, ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]),
        ])

        selected = np.empty(0, dtype=np.int64)
        matched_on: list[str] = []
        for base, columns in candidate_specs:
            if base.size == 0:
                continue
            candidate = self._match(base, query, columns)
            if candidate.size >= k:
                selected = candidate
                matched_on = [c for c in columns if c in self.train.columns]
                break
            if candidate.size > selected.size:
                selected = candidate
                matched_on = [c for c in columns if c in self.train.columns]

        if selected.size == 0:
            return RetrievalResult("similar_examples", {"matched_on": [], "examples": []})

        # Prefer recent history, while keeping deterministic diversity.
        order = np.lexsort((selected, -self._season[selected]))
        selected = selected[order]
        if selected.size > max(k * 8, k):
            selected = selected[: k * 8]

        rng = np.random.default_rng(self.seed + int(query.name if isinstance(query.name, (int, np.integer)) else 0))
        if selected.size > k:
            pick = np.sort(rng.choice(selected.size, size=k, replace=False))
            selected = selected[pick]

        display_columns = [c for c in CONTEXT_COLUMNS if c in self.train.columns]
        history_columns = [
            c for c in self.train.columns
            if c.startswith(HISTORY_PREFIXES)
        ]
        # Keep prompts compact: only a bounded number of history fields.
        display_columns += history_columns[:12]
        examples = []
        for idx in selected:
            row = self.train.iloc[int(idx)]
            item = {c: self._scalar(row[c]) for c in display_columns}
            item[TARGET] = self._scalar(row[TARGET])
            examples.append(item)
        return RetrievalResult(
            "similar_examples",
            {
                "matched_on": matched_on,
                "candidate_count": int(selected.size),
                "examples": examples,
            },
        )

    def query_snapshot(self, query: pd.Series) -> dict[str, Any]:
        columns = [c for c in CONTEXT_COLUMNS if c in query.index]
        history_columns = [c for c in query.index if c.startswith(HISTORY_PREFIXES)]
        columns += history_columns[:20]
        return {c: self._scalar(query[c]) for c in columns}

    def call(self, name: str, query: pd.Series, **kwargs: Any) -> RetrievalResult:
        if name == "pitcher_history":
            return self.pitcher_history(query)
        if name == "batter_history":
            return self.batter_history(query)
        if name == "context_history":
            return self.context_history(query)
        if name == "matchup_history":
            return self.matchup_history(query)
        if name == "similar_examples":
            return self.similar_examples(query, k=int(kwargs.get("k", 12)))
        raise KeyError(f"unknown RAG tool: {name}")
