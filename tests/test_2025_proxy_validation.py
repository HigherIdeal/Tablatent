from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_2025_proxy_validation import FoldSpec, _weighted_summary, fold_masks


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2023, 2023, 2024, 2024, 2024, 2024, 2024, 2024],
            "game_month": [8, 4, 9, 4, 5, 6, 7, 8, 10],
        }
    )


def test_season_forward_fold_is_strictly_temporal() -> None:
    recent, stable, valid = fold_masks(
        _frame(),
        FoldSpec("season_forward_2024", 0.5, "season_forward"),
        "season",
        "game_month",
    )
    assert recent.tolist() == [False, True, True, False, False, False, False, False, False]
    assert stable.tolist() == [True, True, True, False, False, False, False, False, False]
    assert valid.tolist() == [False, False, False, True, True, True, True, True, True]
    assert not bool((recent & valid).any())
    assert not bool((stable & valid).any())


def test_mid_2024_fold_uses_only_observed_2024_prefix() -> None:
    recent, stable, valid = fold_masks(
        _frame(),
        FoldSpec("mid_2024", 0.2, "within_2024", 5, 6, 7),
        "season",
        "game_month",
    )
    assert recent.tolist() == [False, True, True, True, True, False, False, False, False]
    assert stable.tolist() == [True, True, True, True, True, False, False, False, False]
    assert valid.tolist() == [False, False, False, False, False, True, True, False, False]
    assert not bool((recent & valid).any())
    assert not bool((stable & valid).any())


def test_late_2024_fold_expands_training_through_july() -> None:
    recent, stable, valid = fold_masks(
        _frame(),
        FoldSpec("late_2024", 0.3, "within_2024", 7, 8, None),
        "season",
        "game_month",
    )
    assert recent.tolist() == [False, True, True, True, True, True, True, False, False]
    assert stable.tolist() == [True, True, True, True, True, True, True, False, False]
    assert valid.tolist() == [False, False, False, False, False, False, False, True, True]
    assert not bool((recent & valid).any())
    assert not bool((stable & valid).any())


def test_weighted_summary_prefers_higher_weighted_score() -> None:
    rows = []
    weights = {"season_forward_2024": 0.5, "mid_2024": 0.2, "late_2024": 0.3}
    for fold, weight in weights.items():
        rows.append(
            {
                "fold": fold,
                "recent_iterations": 250,
                "stable_iterations": 250,
                "alpha_recent": 0.30,
                "raw_score": 800.0,
                "brier": 0.248,
                "delta_raw_vs_best_recent": 20.0,
            }
        )
        rows.append(
            {
                "fold": fold,
                "recent_iterations": 400,
                "stable_iterations": 250,
                "alpha_recent": 0.30,
                "raw_score": 790.0,
                "brier": 0.249,
                "delta_raw_vs_best_recent": 10.0,
            }
        )
    summary = _weighted_summary(pd.DataFrame(rows), weights)
    best = summary.iloc[0]
    assert int(best["recent_iterations"]) == 250
    assert np.isclose(float(best["weighted_raw_score"]), 800.0)
    assert int(best["improved_folds_vs_best_recent"]) == 3
