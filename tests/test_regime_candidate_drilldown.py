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

import run_regime_candidate_drilldown as drill


def test_global_quantile_groups_are_shared_and_target_independent() -> None:
    series = pd.Series(np.arange(100, dtype=float), name="x")
    groups, edges = drill._global_quantile_groups(series, 4)
    assert len(edges) == 5
    assert groups.nunique() == 4
    assert groups.iloc[0].startswith("Q1[")
    assert groups.iloc[-1].startswith("Q4[")


def test_game_type_controlled_residual_centers_each_season_type_cell() -> None:
    frame = pd.DataFrame(
        {
            "season": [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
            "game_type": ["F", "F", "R", "R", "F", "F", "R", "R"],
            "control_success": [1, 0, 0, 0, 1, 1, 0, 1],
        }
    )
    drill._add_game_type_controlled_residual(
        frame,
        target_col="control_success",
        season_col="season",
    )
    means = frame.groupby(["season", "game_type"])[drill.RESIDUAL_COL].mean()
    assert np.allclose(means.to_numpy(float), 0.0)


def test_group_shift_table_detects_old_to_recent_reversal() -> None:
    rows = []
    for year in drill.YEARS:
        effect = 0.03 if year <= 2022 else -0.02
        rows.append(
            {
                "candidate": "synthetic",
                "cohort": "all",
                "season": year,
                "group": "A",
                "count": 1000,
                "success_rate": 0.5 + effect,
                "season_centered_effect": effect,
                "game_type_controlled_effect": effect,
            }
        )
    profile = pd.DataFrame(rows)
    shifts = drill._group_shift_table(profile, candidate="synthetic", min_era_count=500)
    assert len(shifts) == 1
    row = shifts.iloc[0]
    assert np.isclose(row["old_game_type_controlled_effect"], 0.03)
    assert np.isclose(row["recent_game_type_controlled_effect"], -0.02)
    assert np.isclose(row["controlled_effect_delta"], -0.05)
    assert row["recent_same_direction"]
