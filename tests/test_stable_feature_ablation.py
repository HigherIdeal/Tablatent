from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_stable_feature_ablation as ablation


def test_stable_variants_are_valid_subsets() -> None:
    feature_sets = ablation.build_stable_feature_sets()
    baseline = feature_sets["baseline_drop_game_type"]
    baseline_set = set(baseline)
    assert "game_type" not in baseline_set
    for name, features in feature_sets.items():
        assert set(features).issubset(baseline_set), name
        assert len(features) == len(set(features)), name


def test_locked_config_is_inside_secondary_grid() -> None:
    assert ablation.RECENT_ITERATIONS == 400
    assert ablation.LOCKED_STABLE_ITERATIONS in ablation.STABLE_ITERATIONS_GRID
    assert ablation.LOCKED_ALPHA in ablation.ALPHAS


def test_recent_success_family_removes_derived_encodings_too() -> None:
    drops = set(ablation.STABLE_DROP_VARIANTS["drop_pitcher_recent_success_family"])
    assert set(ablation.SUCCESS_STATE_FEATURES).issubset(drops)
    assert {
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    }.issubset(drops)


def test_locked_aggregate_uses_same_fold_baseline() -> None:
    rows = pd.DataFrame(
        [
            {"fold": "f1", "variant": "baseline_drop_game_type", "raw_score": 100.0, "brier": 0.25},
            {"fold": "f2", "variant": "baseline_drop_game_type", "raw_score": 200.0, "brier": 0.24},
            {"fold": "f1", "variant": "drop_season", "raw_score": 110.0, "brier": 0.249},
            {"fold": "f2", "variant": "drop_season", "raw_score": 190.0, "brier": 0.241},
        ]
    )
    summary = ablation._aggregate_locked(rows, {"f1": 0.5, "f2": 0.5})
    row = summary.loc[summary["variant"].eq("drop_season")].iloc[0]
    assert float(row["weighted_delta_raw_vs_baseline"]) == 0.0
    assert float(row["worst_delta_raw_vs_baseline"]) == -10.0
    assert int(row["improved_folds_vs_baseline"]) == 1
