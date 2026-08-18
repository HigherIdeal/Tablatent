from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


FOLD_WEIGHTS = {2022: 0.20, 2023: 0.30, 2024: 0.50}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else repo_root() / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {p}")
    value["_config_path"] = str(p)
    value["_repo_root"] = str(repo_root())
    return value


def ensure_worker_gpu(expected_physical: int) -> None:
    """Respect an externally isolated GPU and expose it as logical device 0.

    The stable project runtime intentionally hard-codes physical GPU 2.  The night
    campaign has two workers, so it does not call configure_cuda().  Instead each
    launcher sets CUDA_VISIBLE_DEVICES to physical GPU 2 or 3 and CatBoost always
    uses logical device 0 inside that isolated process.
    """
    expected = str(int(expected_physical))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = expected
        visible = expected
    first = visible.split(",")[0].strip()
    if first != expected:
        raise RuntimeError(
            f"worker expects physical GPU {expected}, but CUDA_VISIBLE_DEVICES={visible!r}"
        )


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy(np.float64)
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray | float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(q / (1.0 - q))


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    return {
        "rows": int(len(y)),
        "target_rate": float(np.mean(y)),
        "brier": brier(y, p),
        "logloss": logloss(y, p),
        "auc": auc(y, p),
        "prob_mean": float(np.mean(p)),
        "prob_std": float(np.std(p)),
    }


def objective_from_folds(
    fold_metrics: dict[int, dict[str, Any]],
    *,
    stability_penalty: float = 0.25,
) -> dict[str, float]:
    seasons = sorted(int(s) for s in fold_metrics)
    if not seasons:
        return {"objective": float("inf"), "weighted_brier": float("inf"), "std_brier": float("inf")}
    raw_weights = np.array([FOLD_WEIGHTS.get(s, 1.0) for s in seasons], dtype=np.float64)
    raw_weights /= raw_weights.sum()
    values = np.array([float(fold_metrics[s]["brier"]) for s in seasons], dtype=np.float64)
    weighted = float(np.sum(raw_weights * values))
    spread = float(np.std(values))
    return {
        "objective": weighted + float(stability_penalty) * spread,
        "weighted_brier": weighted,
        "std_brier": spread,
        "worst_brier": float(np.max(values)),
        "best_brier": float(np.min(values)),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
    return out


def utc_timestamp() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class CampaignTimer:
    hours: float
    reserve_minutes: float = 20.0

    def __post_init__(self) -> None:
        self.started = time.time()
        self.deadline = self.started + max(float(self.hours), 0.01) * 3600.0
        self.search_deadline = self.deadline - max(float(self.reserve_minutes), 0.0) * 60.0
        if self.search_deadline <= self.started:
            self.search_deadline = self.deadline

    def seconds_left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def search_seconds_left(self) -> float:
        return max(0.0, self.search_deadline - time.time())

    def searching(self) -> bool:
        return time.time() < self.search_deadline

    def total_elapsed(self) -> float:
        return max(0.0, time.time() - self.started)


class TrialRecorder:
    def __init__(
        self,
        output_dir: Path,
        *,
        worker: str,
        timer: CampaignTimer,
        stability_penalty: float,
    ) -> None:
        self.output_dir = output_dir
        self.worker = worker
        self.timer = timer
        self.stability_penalty = float(stability_penalty)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.output_dir / "trials.jsonl"
        self.best_path = self.output_dir / "best.json"
        self.best_md_path = self.output_dir / "best.md"
        self.heartbeat_path = self.output_dir / "heartbeat.json"
        self.best: dict[str, Any] | None = None
        previous = read_jsonl(self.trials_path)
        for trial in previous:
            if trial.get("status") == "ok":
                if self.best is None or float(trial.get("objective", float("inf"))) < float(
                    self.best.get("objective", float("inf"))
                ):
                    self.best = trial

    def heartbeat(self, *, phase: str, trial_id: str | None = None, extra: dict[str, Any] | None = None) -> None:
        value: dict[str, Any] = {
            "worker": self.worker,
            "phase": phase,
            "trial_id": trial_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "timestamp_utc": utc_timestamp(),
            "elapsed_seconds": self.timer.total_elapsed(),
            "seconds_left": self.timer.seconds_left(),
        }
        if extra:
            value.update(extra)
        atomic_write_json(self.heartbeat_path, value)

    def record(self, trial: dict[str, Any]) -> None:
        trial = dict(trial)
        trial.setdefault("worker", self.worker)
        trial.setdefault("timestamp_utc", utc_timestamp())
        trial.setdefault("status", "ok")
        append_jsonl(self.trials_path, trial)
        if trial.get("status") == "ok" and np.isfinite(float(trial.get("objective", float("inf")))):
            if self.best is None or float(trial["objective"]) < float(self.best.get("objective", float("inf"))):
                self.best = trial
                atomic_write_json(self.best_path, self.best)
                atomic_write_text(self.best_md_path, format_best_markdown(self.best))
        self.heartbeat(phase="search", trial_id=str(trial.get("trial_id")))


def format_best_markdown(trial: dict[str, Any]) -> str:
    lines = [
        "# Current best trial",
        "",
        f"- worker: `{trial.get('worker', '')}`",
        f"- trial: `{trial.get('trial_id', '')}`",
        f"- objective: `{float(trial.get('objective', float('nan'))):.9f}`",
        f"- weighted Brier: `{float(trial.get('weighted_brier', float('nan'))):.9f}`",
        f"- std Brier: `{float(trial.get('std_brier', float('nan'))):.9f}`",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(trial.get("config", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Folds",
        "",
        "| season | Brier | AUC | rows |",
        "|---:|---:|---:|---:|",
    ]
    for season, metrics in sorted((trial.get("folds") or {}).items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| {season} | {float(metrics.get('brier', float('nan'))):.9f} | "
            f"{float(metrics.get('auc', float('nan'))):.5f} | {int(metrics.get('rows', 0)):,} |"
        )
    return "\n".join(lines) + "\n"


def experience_groups(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    return np.where(
        n <= 0,
        "no_prior",
        np.where(n < 50, "lt50", np.where(n < 200, "50_199", np.where(n < 500, "200_499", "ge500"))),
    )


def grouped_metrics(y: np.ndarray, pred: np.ndarray, n: np.ndarray) -> dict[str, Any]:
    labels = experience_groups(n)
    out: dict[str, Any] = {}
    for label in ("no_prior", "lt50", "50_199", "200_499", "ge500"):
        mask = labels == label
        if not mask.any():
            continue
        metrics = classification_metrics(y[mask], pred[mask])
        out[label] = metrics
    return out


def weighted_profile_lookup(
    profiles: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    traits: Iterable[str],
    mode: str,
    current_seasons: Iterable[int],
) -> pd.DataFrame:
    """Build a frozen pitcher profile lookup for each current season.

    mode=prev uses only s-1; recent2 uses s-1/s-2; career uses every season < s.
    The returned fallback values are league-level weighted means from the same source
    seasons and therefore remain causal for the current season.
    """
    traits = list(dict.fromkeys(str(x) for x in traits))
    rows: list[pd.DataFrame] = []
    numeric_season = pd.to_numeric(profiles[season_col], errors="raise").astype(int)
    for current in sorted(int(x) for x in current_seasons):
        if mode == "prev":
            source_seasons = {current - 1}
        elif mode == "recent2":
            source_seasons = {current - 1, current - 2}
        elif mode == "career":
            source_seasons = set(int(x) for x in numeric_season.unique() if int(x) < current)
        else:
            raise ValueError(f"unknown history mode: {mode}")
        source = profiles[numeric_season.isin(source_seasons)].copy()
        if len(source) == 0:
            continue
        weights = pd.to_numeric(source["pitch_count"], errors="coerce").fillna(0).to_numpy(np.float64)
        source["_w"] = np.maximum(weights, 0.0)
        for trait in traits:
            rate = pd.to_numeric(source[trait], errors="coerce").to_numpy(np.float64)
            source[f"_ws_{trait}"] = np.where(np.isfinite(rate), rate, 0.0) * source["_w"].to_numpy(np.float64)
        agg_spec: dict[str, tuple[str, str]] = {"history_n": ("_w", "sum")}
        for trait in traits:
            agg_spec[f"_sum_{trait}"] = (f"_ws_{trait}", "sum")
        grouped = source.groupby(pitcher_col, dropna=False, sort=False).agg(**agg_spec).reset_index()
        denom = np.maximum(grouped["history_n"].to_numpy(np.float64), 1e-12)
        grouped["current_season"] = current
        total_w = float(np.sum(source["_w"].to_numpy(np.float64)))
        for trait in traits:
            grouped[f"history_raw_{trait}"] = grouped[f"_sum_{trait}"].to_numpy(np.float64) / denom
            fallback = (
                float(np.sum(source[f"_ws_{trait}"].to_numpy(np.float64)) / total_w)
                if total_w > 0
                else 0.5
            )
            grouped[f"history_fallback_{trait}"] = fallback
        keep = ["current_season", pitcher_col, "history_n"]
        keep += [f"history_raw_{x}" for x in traits]
        keep += [f"history_fallback_{x}" for x in traits]
        rows.append(grouped[keep])
    if not rows:
        return pd.DataFrame(columns=["current_season", pitcher_col, "history_n"])
    return pd.concat(rows, axis=0, ignore_index=True)


def attach_history_mode(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    traits: Iterable[str],
    mode: str,
) -> pd.DataFrame:
    traits = list(dict.fromkeys(str(x) for x in traits))
    keys = frame[[season_col, pitcher_col]].copy()
    keys["current_season"] = pd.to_numeric(keys[season_col], errors="raise").astype(int)
    keys["_row"] = np.arange(len(keys), dtype=np.int64)
    attached = keys[["_row", "current_season", pitcher_col]].merge(
        lookup,
        on=["current_season", pitcher_col],
        how="left",
        sort=False,
    )
    attached = attached.sort_values("_row", kind="stable").reset_index(drop=True)
    out = frame.copy()
    n = pd.to_numeric(attached.get("history_n"), errors="coerce").fillna(0).to_numpy(np.float64)
    out[f"hist_{mode}_n"] = n.astype(np.float32)
    out[f"hist_{mode}_log_n"] = np.log1p(n).astype(np.float32)
    out[f"hist_{mode}_has"] = (n > 0).astype(np.float32)
    current = pd.to_numeric(out[season_col], errors="raise").astype(int).to_numpy()
    for trait in traits:
        raw_series = attached.get(f"history_raw_{trait}")
        fallback_series = attached.get(f"history_fallback_{trait}")
        raw = pd.to_numeric(raw_series, errors="coerce").to_numpy(np.float64)
        fallback = pd.to_numeric(fallback_series, errors="coerce").to_numpy(np.float64)
        # A missing pitcher still gets the source-season league fallback.  Recover it
        # from any lookup row for the same current season.
        fallback_by_season = (
            lookup.groupby("current_season", sort=False)[f"history_fallback_{trait}"].first().to_dict()
            if len(lookup)
            else {}
        )
        fallback2 = np.array([float(fallback_by_season.get(int(s), 0.5)) for s in current], dtype=np.float64)
        fallback = np.where(np.isfinite(fallback), fallback, fallback2)
        raw = np.where(np.isfinite(raw), raw, fallback)
        out[f"hist_{mode}_raw_{trait}"] = np.clip(raw, 0.0, 1.0).astype(np.float32)
        out[f"hist_{mode}_fallback_{trait}"] = np.clip(fallback, 0.0, 1.0).astype(np.float32)
    return out


def shrunk_trait(frame: pd.DataFrame, *, mode: str, trait: str, k: float) -> np.ndarray:
    n = pd.to_numeric(frame[f"hist_{mode}_n"], errors="coerce").fillna(0).to_numpy(np.float64)
    raw = pd.to_numeric(frame[f"hist_{mode}_raw_{trait}"], errors="coerce").to_numpy(np.float64)
    fallback = pd.to_numeric(frame[f"hist_{mode}_fallback_{trait}"], errors="coerce").to_numpy(np.float64)
    reliability = n / (n + float(k))
    return np.clip(reliability * raw + (1.0 - reliability) * fallback, 0.0, 1.0)
