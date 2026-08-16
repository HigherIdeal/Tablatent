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

import run_regime_atlas as atlas


def _synthetic_flip_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pitchers = list(range(40))
    for year in atlas.YEARS:
        for pitcher_id in pitchers:
            for game_type in ["F", "R"]:
                n = 60
                if year <= 2022:
                    rate = 0.68 if game_type == "F" else 0.46
                else:
                    rate = 0.44 if game_type == "F" else 0.54
                y = rng.binomial(1, rate, size=n)
                rows.extend(
                    {
                        "season": year,
                        "pitcher_id": pitcher_id,
                        "game_type": game_type,
                        "control_success": int(target),
                    }
                    for target in y
                )
    return pd.DataFrame(rows)


def test_same_player_cohort_keeps_bridge_pitchers() -> None:
    frame = pd.DataFrame(
        {
            "pitcher_id": [1, 1, 1, 1, 2, 2, 2, 3, 3],
            "season": [2019, 2021, 2023, 2024, 2022, 2023, 2024, 2023, 2024],
        }
    )
    mask, stats = atlas._same_player_mask(
        frame,
        pitcher_col="pitcher_id",
        season_col="season",
        min_old_seasons=2,
    )
    assert set(frame.loc[mask, "pitcher_id"].unique()) == {1}
    assert stats["pitchers"] == 1


def test_distribution_jsd_zero_for_equal_composition() -> None:
    groups = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B"])
    season = pd.Series([2019, 2019, 2020, 2020, 2023, 2023, 2024, 2024])
    value = atlas._distribution_jsd(groups, season, [2019, 2020], [2023, 2024])
    assert np.isclose(value, 0.0)


def test_game_type_flip_detects_2023_and_survives_same_player_control() -> None:
    frame = _synthetic_flip_frame()
    same_mask, _ = atlas._same_player_mask(
        frame,
        pitcher_col="pitcher_id",
        season_col="season",
        min_old_seasons=2,
    )
    summary, profiles, changepoints = atlas._signal_summary(
        frame,
        signal="game_type",
        components=["game_type"],
        target_col="control_success",
        season_col="season",
        categorical_features={"game_type"},
        numeric_bins=6,
        max_auto_categories=20,
        min_era_count=100,
        min_effect_for_flip=0.002,
        same_player_mask=same_mask,
    )
    assert int(summary["best_change_year"]) == 2023
    assert summary["shift_2023_rmse"] > 0.10
    assert summary["sign_flip_rate_2023"] > 0.9
    assert summary["same_player_preservation"] > 0.8
    assert set(profiles["season"].unique()) == set(atlas.YEARS)
    assert set(changepoints["change_year"].unique()) == set(atlas.CHANGE_YEARS)


def test_classification_marks_persistent_2023_shift() -> None:
    row = pd.Series(
        {
            "shift_2023_rmse": 0.02,
            "changepoint_ratio_2023": 4.0,
            "recent_internal_rmse": 0.003,
            "sign_flip_rate_2023": 0.9,
            "same_player_preservation": 0.95,
            "best_change_year": 2023,
        }
    )
    assert atlas._classification(row) == "post_2023_regime"
