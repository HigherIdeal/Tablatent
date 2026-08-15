from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

screen = importlib.import_module("run_dual_track_late2024_screen")


def test_make_masks_uses_2023_plus_early_2024_for_recent() -> None:
    frame = pd.DataFrame(
        {
            "season": [2022, 2023, 2023, 2024, 2024, 2024, 2024],
            "game_month": [9, 4, 9, 4, 7, 8, 10],
        }
    )
    recent, stable, valid = screen.make_masks(
        frame,
        season_col="season",
        month_col="game_month",
        valid_year=2024,
        cutoff_month=7,
    )

    assert recent.tolist() == [False, True, True, True, True, False, False]
    assert stable.tolist() == [True, True, True, True, True, False, False]
    assert valid.tolist() == [False, False, False, False, False, True, True]


def test_masks_are_disjoint_from_late_holdout() -> None:
    frame = pd.DataFrame(
        {
            "season": [2019, 2023, 2024, 2024],
            "game_month": [5, 10, 7, 8],
        }
    )
    recent, stable, valid = screen.make_masks(
        frame,
        season_col="season",
        month_col="game_month",
        valid_year=2024,
        cutoff_month=7,
    )

    assert not bool((recent & valid).any())
    assert not bool((stable & valid).any())
    assert int(valid.sum()) == 1
