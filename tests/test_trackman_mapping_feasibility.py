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

import run_trackman_mapping_feasibility as probe


def test_exact_overlap_identical_and_partial_profiles() -> None:
    identical = probe._exact_overlap({1: 5, 2: 3}, {1: 5, 2: 3})
    assert np.allclose(identical, [1.0, 1.0, 1.0])

    overlap, jaccard, count_ratio = probe._exact_overlap({1: 5, 2: 5}, {1: 3})
    assert np.isclose(overlap, 1.0)
    assert np.isclose(jaccard, 0.3)
    assert np.isclose(count_ratio, 0.3)


def test_cross_season_consistency_finds_persistent_pair() -> None:
    matches = pd.DataFrame(
        {
            "season": [2021, 2022, 2023, 2021, 2022, 2023],
            "main_pitcher_id": ["A", "A", "A", "B", "B", "B"],
            "best_trackman_id": ["X", "X", "X", "Y", "Z", "Y"],
            "best_score": [0.9, 0.91, 0.92, 0.6, 0.61, 0.62],
            "margin": [0.3, 0.3, 0.3, 0.05, 0.05, 0.05],
            "coarse_mutual_nearest": [True, True, True, False, False, False],
        }
    )
    result = probe._cross_season_consistency(matches).set_index("main_pitcher_id")
    assert result.loc["A", "all_same"]
    assert np.isclose(result.loc["A", "consistency"], 1.0)
    assert np.isclose(result.loc["B", "consistency"], 2 / 3)


def test_score_one_season_recovers_permuted_identity_from_context_histograms() -> None:
    main_counts = pd.DataFrame(
        {
            "season": [2024] * 6,
            "pitcher_id": ["A", "A", "A", "B", "B", "B"],
            "context_token": [1, 2, 3, 4, 5, 6],
            "count": [20, 10, 5, 18, 9, 4],
        }
    )
    tm_counts = pd.DataFrame(
        {
            "season": [2024] * 6,
            "pitcher_trackman_id": ["Y", "Y", "Y", "X", "X", "X"],
            "context_token": [4, 5, 6, 1, 2, 3],
            "count": [18, 9, 4, 20, 10, 5],
        }
    )
    main_totals = main_counts.groupby(["season", "pitcher_id"], as_index=False)["count"].sum().rename(columns={"count": "rows"})
    tm_totals = tm_counts.groupby(["season", "pitcher_trackman_id"], as_index=False)["count"].sum().rename(columns={"count": "rows"})

    result = probe._score_one_season(
        main_counts,
        main_totals,
        tm_counts,
        tm_totals,
        main_pitcher_col="pitcher_id",
        tm_pitcher_col="pitcher_trackman_id",
        hash_dim=256,
        top_k=2,
    ).set_index("main_pitcher_id")

    assert result.loc["A", "best_trackman_id"] == "X"
    assert result.loc["B", "best_trackman_id"] == "Y"
    assert result.loc["A", "best_score"] > 0.99
    assert result.loc["B", "best_score"] > 0.99
    assert bool(result.loc["A", "coarse_mutual_nearest"])
    assert bool(result.loc["B", "coarse_mutual_nearest"])


def test_verdict_requires_multiple_identity_like_signals() -> None:
    strong = {
        "cross_season_mean_consistency": 0.90,
        "null_consistency_p95": 0.60,
        "mutual_fraction": 0.60,
        "median_margin": 0.10,
    }
    weak = {
        "cross_season_mean_consistency": 0.65,
        "null_consistency_p95": 0.62,
        "mutual_fraction": 0.20,
        "median_margin": 0.01,
    }
    assert probe._verdict(strong) == "STRONG_IDENTITY_LIKE_SIGNAL"
    assert probe._verdict(weak) == "WEAK_OR_AMBIGUOUS_SIGNAL"
