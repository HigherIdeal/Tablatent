#!/usr/bin/env python3
"""Temporal-safe tabular RAG for the offline Qwen3-1.7B experiment.

No embedding/learned retrieval model is used. Python performs deterministic
historical lookup only; Qwen3-1.7B remains the sole learned predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TARGET = "control_success"

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
    "base_state",
    "base_state_before",
    "runners",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "run_top",
    "run_bot",
    "run_total",
    "run_top_before",
    "run_bot_before",
    "score_diff",
    "score_diff_before",
    "win_expectancy",
    "win_expectancy_before",
    "li",
    "leverage_index_before",
]

FEATURE_PREFIXES = ("asof_", "prev1_", "prev3_", "prev5_", "tm_")
MATCH_COLUMNS = {
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "base_state",
    "base_state_before",
    "batter_id",
}


@dataclass
class RetrievalResult:
    name: str
    payload: dict[str, Any]


def canonical(value: Any) -> str:
    if value is None or pd.isna(value):
        return "<NA>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return format(number, ".12g")
    return str(value).strip()


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
        self._prior_index_cache: dict[int, np.ndarray] = {}
        self._context_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._string_arrays: dict[str, np.ndarray] = {}
        for column in MATCH_COLUMNS:
            if column in self.train.columns:
                self._string_arrays[column] = np.asarray(
                    [canonical(value) for value in self.train[column].to_numpy()],
                    dtype=object,
                )

    def _group_indices(self, column: str) -> dict[str, np.ndarray]:
        if column not in self.train.columns:
            return {}
        groups: dict[str, list[int]] = {}
        for idx, value in enumerate(self.train[column].to_numpy()):
            key = canonical(value)
            if key == "<NA>":
                continue
            groups.setdefault(key, []).append(idx)
        return {key: np.asarray(values, dtype=np.int64) for key, values in groups.items()}

    @staticmethod
    def _scalar(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            number = float(value)
            return number if np.isfinite(number) else None
        return value

    def prior_indices(self, season: int) -> np.ndarray:
        season = int(season)
        cached = self._prior_index_cache.get(season)
        if cached is None:
            cached = np.flatnonzero(self._season < season).astype(np.int64)
            self._prior_index_cache[season] = cached
        return cached

    def prior_rate(self, season: int) -> float:
        payload = self._rate_payload(self.prior_indices(season))
        value = payload.get("success_rate")
        return 0.5 if value is None else float(value)

    def _before(self, indices: np.ndarray, season: int) -> np.ndarray:
        if indices.size == 0:
            return indices
        return indices[self._season[indices] < int(season)]

    def _match(self, indices: np.ndarray, query: pd.Series, columns: list[str]) -> np.ndarray:
        out = indices
        for column in columns:
            if column not in self.train.columns or column not in query.index:
                continue
            value = canonical(query[column])
            if value == "<NA>":
                continue
            values = self._string_arrays.get(column)
            if values is None:
                values = np.asarray(
                    [canonical(item) for item in self.train[column].to_numpy()],
                    dtype=object,
                )
                self._string_arrays[column] = values
            out = out[values[out] == value]
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
        idx = self._before(
            self._pitcher_groups.get(canonical(query.get("pitcher_id")), np.empty(0, dtype=np.int64)),
            season,
        )
        payload = self._rate_payload(idx)
        if idx.size:
            seasons = self._season[idx]
            latest = int(np.nanmax(seasons))
            payload["latest_prior_season"] = latest
            payload["latest_prior_season_stats"] = self._rate_payload(idx[seasons == latest])
        return RetrievalResult("pitcher_history", payload)

    def batter_history(self, query: pd.Series) -> RetrievalResult:
        season = int(query["season"])
        idx = self._before(
            self._batter_groups.get(canonical(query.get("batter_id")), np.empty(0, dtype=np.int64)),
            season,
        )
        return RetrievalResult("batter_history", self._rate_payload(idx))

    def context_history(self, query: pd.Series) -> RetrievalResult:
        season = int(query["season"])
        base_column = "base_state" if "base_state" in self.train.columns else "base_state_before"
        levels = [
            ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand", base_column],
            ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"],
            ["balls_before", "strikes_before", "outs_before"],
            ["balls_before", "strikes_before"],
        ]
        base = self.prior_indices(season)
        chosen = base
        used: list[str] = []
        for columns in levels:
            present = [column for column in columns if column in self.train.columns and column in query.index]
            cache_key = (season, *[(column, canonical(query[column])) for column in present])
            cached = self._context_cache.get(cache_key)
            if cached is not None:
                if cached["n"] >= 100:
                    return RetrievalResult("context_history", dict(cached))
                continue
            candidate = self._match(base, query, present)
            payload = self._rate_payload(candidate)
            payload["matched_on"] = present
            self._context_cache[cache_key] = payload
            if candidate.size >= 100:
                return RetrievalResult("context_history", dict(payload))
            if candidate.size > 0:
                chosen = candidate
                used = present
        payload = self._rate_payload(chosen)
        payload["matched_on"] = used
        return RetrievalResult("context_history", payload)

    def matchup_history(self, query: pd.Series) -> RetrievalResult:
        season = int(query["season"])
        idx = self._before(
            self._pitcher_groups.get(canonical(query.get("pitcher_id")), np.empty(0, dtype=np.int64)),
            season,
        )
        idx = self._match(idx, query, ["batter_id"])
        return RetrievalResult("matchup_history", self._rate_payload(idx))

    def similar_examples(self, query: pd.Series, k: int = 12) -> RetrievalResult:
        season = int(query["season"])
        k = int(np.clip(k, 4, 20))
        pitcher_idx = self._before(
            self._pitcher_groups.get(canonical(query.get("pitcher_id")), np.empty(0, dtype=np.int64)),
            season,
        )
        all_prior = self.prior_indices(season)
        candidate_specs = [
            (pitcher_idx, ["balls_before", "strikes_before", "batter_hand"]),
            (pitcher_idx, ["balls_before", "strikes_before"]),
            (pitcher_idx, []),
            (all_prior, ["balls_before", "strikes_before", "outs_before", "pitcher_hand", "batter_hand"]),
            (all_prior, ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]),
        ]

        selected = np.empty(0, dtype=np.int64)
        matched_on: list[str] = []
        total_candidates = 0
        for base, columns in candidate_specs:
            if base.size == 0:
                continue
            candidate = self._match(base, query, columns)
            if candidate.size > selected.size:
                selected = candidate
                matched_on = [column for column in columns if column in self.train.columns]
                total_candidates = int(candidate.size)
            if candidate.size >= k:
                selected = candidate
                matched_on = [column for column in columns if column in self.train.columns]
                total_candidates = int(candidate.size)
                break

        if selected.size == 0:
            return RetrievalResult("similar_examples", {"matched_on": [], "candidate_count": 0, "examples": []})

        # Recent seasons first; sample deterministically from a bounded recent pool.
        order = np.lexsort((selected, -self._season[selected]))
        selected = selected[order][: max(k * 8, k)]
        seed_offset = int(query.name) if isinstance(query.name, (int, np.integer)) else 0
        rng = np.random.default_rng(self.seed + seed_offset)
        if selected.size > k:
            selected = selected[np.sort(rng.choice(selected.size, size=k, replace=False))]

        display_columns = [column for column in CONTEXT_COLUMNS if column in self.train.columns]
        extra_columns = [column for column in self.train.columns if column.startswith(FEATURE_PREFIXES)]
        display_columns += extra_columns[:12]
        examples = []
        for idx in selected:
            row = self.train.iloc[int(idx)]
            item = {column: self._scalar(row[column]) for column in display_columns}
            item[TARGET] = self._scalar(row[TARGET])
            examples.append(item)
        return RetrievalResult(
            "similar_examples",
            {
                "matched_on": matched_on,
                "candidate_count": total_candidates,
                "examples": examples,
            },
        )

    def query_snapshot(self, query: pd.Series) -> dict[str, Any]:
        columns = [column for column in CONTEXT_COLUMNS if column in query.index]
        extra_columns = [column for column in query.index if column.startswith(FEATURE_PREFIXES)]
        columns += extra_columns[:20]
        return {column: self._scalar(query[column]) for column in columns}

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
