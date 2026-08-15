from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_trackman_mapping_feasibility as core


def _null_consistency_fixed(
    matches: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Permutation null for cross-season ID persistence.

    Only the Trackman ID assignment is permuted within each season. The null
    statistic needs only season/main-id/best-Trackman-id; it must not call the
    richer diagnostic helper that expects score/margin/mutual columns.
    """
    rng = np.random.default_rng(seed)
    required = {"season", "main_pitcher_id", "best_trackman_id"}
    missing = required - set(matches.columns)
    if missing:
        raise KeyError(f"Missing columns for null consistency: {sorted(missing)}")

    base = matches[["season", "main_pitcher_id", "best_trackman_id"]].copy()
    scores: list[float] = []

    for _ in range(repetitions):
        shuffled_parts = []
        for _, group in base.groupby("season", sort=False):
            temp = group.copy()
            temp["best_trackman_id"] = rng.permutation(
                temp["best_trackman_id"].to_numpy()
            )
            shuffled_parts.append(temp)
        shuffled = pd.concat(shuffled_parts, ignore_index=True)

        per_pitcher: list[float] = []
        for _, group in shuffled.groupby("main_pitcher_id", sort=False):
            if len(group) < 2:
                continue
            counts = group["best_trackman_id"].value_counts()
            per_pitcher.append(float(counts.iloc[0] / len(group)))

        scores.append(float(np.mean(per_pitcher)) if per_pitcher else np.nan)

    return np.asarray(scores, dtype=np.float64)


# Compatibility patch for the original side-experiment script.
core._null_consistency = _null_consistency_fixed


if __name__ == "__main__":
    core.main()
