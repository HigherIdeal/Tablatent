from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_legacy_mapping_validation as v


def test_split_counts_preserves_every_profile_count() -> None:
    counts = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "pitcher_id": ["A", "A", "B"],
            "context_token": [1, 2, 3],
            "count": [11, 20, 7],
        }
    )
    a, b = v._split_counts(counts, seed=42)
    aa = a.set_index(["season", "pitcher_id", "context_token"])["count"]
    bb = b.set_index(["season", "pitcher_id", "context_token"])["count"]
    keys = counts.set_index(["season", "pitcher_id", "context_token"])["count"]
    for key, total in keys.items():
        assert int(aa.get(key, 0) + bb.get(key, 0)) == int(total)


def test_hungarian_map_recovers_known_global_permutation() -> None:
    score = np.array(
        [
            [0.1, 0.95, 0.2],
            [0.9, 0.2, 0.1],
            [0.2, 0.1, 0.92],
        ]
    )
    result = v._hungarian_map(["A", "B", "C"], ["X", "Y", "Z"], score, season=2024)
    mapping = dict(zip(result["main_pitcher_id"], result["best_trackman_id"]))
    assert mapping == {"A": "Y", "B": "X", "C": "Z"}
    assert result["is_local_best"].all()


def test_mapping_agreement_measures_same_pair_rate() -> None:
    a = pd.DataFrame(
        {
            "main_pitcher_id": ["A", "B", "C"],
            "best_trackman_id": ["X", "Y", "Z"],
        }
    )
    b = pd.DataFrame(
        {
            "main_pitcher_id": ["A", "B", "C"],
            "best_trackman_id": ["X", "Z", "Z"],
        }
    )
    n, rate = v._mapping_agreement(a, b)
    assert n == 3
    assert np.isclose(rate, 2 / 3)


def test_cross_season_consistency_and_null_do_not_need_score_columns() -> None:
    maps = pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2022, 2023, 2024],
            "main_pitcher_id": ["A", "A", "A", "B", "B", "B"],
            "best_trackman_id": ["X", "X", "X", "Y", "Z", "Y"],
        }
    )
    c = v._cross_season_consistency(maps).set_index("main_pitcher_id")
    assert np.isclose(c.loc["A", "consistency"], 1.0)
    assert np.isclose(c.loc["B", "consistency"], 2 / 3)
    null1 = v._permutation_cross_season_null(maps, repetitions=10, seed=7)
    null2 = v._permutation_cross_season_null(maps, repetitions=10, seed=7)
    assert len(null1) == 10
    assert np.allclose(null1, null2, equal_nan=True)


def test_legacy_score_uses_context_when_pitchmix_missing() -> None:
    context = np.array([[0.2, 0.4], [0.6, 0.8]])
    mix = np.array([[1.0, np.nan], [0.0, 0.5]])
    score = v._legacy_score(context, mix)
    assert np.isclose(score[0, 0], 0.85 * 0.2 + 0.15)
    assert np.isclose(score[0, 1], 0.4)
    assert np.isclose(score[1, 0], 0.85 * 0.6)
    assert np.isclose(score[1, 1], 0.85 * 0.8 + 0.15 * 0.5)


def test_verdict_requires_two_independent_signals() -> None:
    strong = {
        "cross_season_excess_vs_null95": 0.12,
        "mean_split_same_pair_rate": 0.25,
        "mean_context_map_pitchmix_excess": 0.01,
    }
    weak = {
        "cross_season_excess_vs_null95": 0.02,
        "mean_split_same_pair_rate": 0.08,
        "mean_context_map_pitchmix_excess": 0.02,
    }
    assert v._summary_verdict(strong) == "CORRESPONDENCE_SIGNAL_WORTH_USING"
    assert v._summary_verdict(weak) == "NO_RELIABLE_CORRESPONDENCE_SIGNAL"
