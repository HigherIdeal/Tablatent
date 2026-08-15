from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_context_interaction_screen as ctx


def test_context_interactions_are_stable_categorical_crosses() -> None:
    frame = pd.DataFrame(
        {
            "balls_before": [0, 3, 1],
            "strikes_before": [0, 2, 1],
            "outs_before": [0, 2, 1],
            "pitcher_hand": ["R", "L", "R"],
            "batter_hand": ["L", "R", "R"],
            "base_state": ["___", "123", None],
        }
    )

    ctx.add_context_interactions(frame)

    assert frame.loc[0, "ctx_count_state"] == "0|0"
    assert frame.loc[1, "ctx_hand_matchup"] == "L|R"
    assert frame.loc[1, "ctx_count_hand"] == "3|2|L|R"
    assert frame.loc[1, "ctx_count_pressure"] == "3|2|123|2"
    assert frame.loc[2, "ctx_count_base"] == "1|1|<MISSING>"
    assert not frame[ctx.INTERACTION_COLUMNS].isna().any().any()


def test_prepare_x_keeps_engineered_crosses_categorical() -> None:
    frame = pd.DataFrame(
        {
            "balls_before": [0, 1],
            "strikes_before": [0, 2],
            "outs_before": [0, 1],
            "pitcher_hand": ["R", "L"],
            "batter_hand": ["L", "R"],
            "base_state": ["___", "1__"],
            "some_numeric": [1.25, 2.5],
        }
    )
    ctx.add_context_interactions(frame)

    x, categorical = ctx.prepare_x(
        frame,
        ["some_numeric", "ctx_count_hand", "ctx_count_pressure"],
    )

    assert categorical == ["ctx_count_hand", "ctx_count_pressure"]
    assert x["ctx_count_hand"].dtype == object
    assert x["ctx_count_pressure"].dtype == object
    assert str(x["some_numeric"].dtype) == "float32"


def test_interaction_spec_excludes_ids_target_and_game_type() -> None:
    forbidden = {"pitcher_id", "batter_id", "control_success", "game_type"}
    used_sources = {source for sources in ctx.INTERACTION_SPECS.values() for source in sources}
    assert used_sources.isdisjoint(forbidden)
