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

import run_regime_candidate_robustness as robust


def test_numeric_codes_cover_all_rows() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0, np.nan, 5.0], dtype=float)
    codes, edges, n_groups = robust._numeric_codes(values, bins=4, xp=np)
    assert len(codes) == len(values)
    assert len(edges) >= 3
    assert int(codes[-2]) == n_groups - 1  # missing bin
    assert np.all(codes >= 0)


def test_profile_centering_zeroes_season_weighted_mean() -> None:
    year_idx = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
    groups = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int32)
    response = np.asarray([0.0, 0.0, 1.0, 1.0, 0.2, 0.2, 0.8, 0.8], dtype=float)
    counts, effects = robust._profile_from_codes(
        response=response,
        year_idx=year_idx,
        group_codes=groups,
        n_groups=2,
        mask=np.ones(len(response), dtype=bool),
        center_by_season=True,
        xp=np,
    )
    for season in [0, 1]:
        valid = counts[season] > 0
        assert np.isclose(np.average(effects[season, valid], weights=counts[season, valid]), 0.0)


def test_era_shift_detects_reversal() -> None:
    counts = np.full((6, 2), 1000.0)
    effects = np.asarray(
        [
            [0.10, -0.10],
            [0.09, -0.09],
            [0.11, -0.11],
            [0.10, -0.10],
            [-0.08, 0.08],
            [-0.09, 0.09],
        ],
        dtype=float,
    )
    metrics = robust._era_shift(
        counts,
        effects,
        robust.OLD_IDX,
        robust.RECENT_IDX,
        min_era_count=100,
    )
    assert metrics["shift_rmse"] > 0.15
    assert metrics["sign_flip_rate"] > 0.99
    assert metrics["changepoint_ratio"] > 1.5


def test_best_changepoint_prefers_2023_for_clean_shift() -> None:
    counts = np.full((6, 2), 1000.0)
    effects = np.asarray(
        [
            [0.05, -0.05],
            [0.05, -0.05],
            [0.05, -0.05],
            [0.05, -0.05],
            [-0.05, 0.05],
            [-0.05, 0.05],
        ],
        dtype=float,
    )
    year, ratio = robust._best_changepoint(counts, effects, min_era_count=100)
    assert year == 2023
    assert ratio > 1.5


def test_candidate_verdict_requires_bin_robustness_in_both_types() -> None:
    rows = []
    for gt in ["R", "F"]:
        for bins in robust.BIN_SETTINGS:
            rows.append(
                {
                    "candidate": "synthetic",
                    "bins": bins,
                    "game_type": gt,
                    "shift_2023_rmse": 0.010,
                    "changepoint_ratio_2023": 1.8,
                    "best_change_year": 2023,
                    "recent_direction_consistency": 0.8,
                    "same_player_delta_correlation": 0.8,
                }
            )
    verdict = robust._candidate_verdict(pd.DataFrame(rows))
    assert verdict["verdict"] == "ROBUST_IN_BOTH_R_F"
