from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_recent_regime_submissions as core
import train_recent_regime_models as train_core


def test_recent_regime_training_seasons_are_exact() -> None:
    assert train_core.TRAIN_SEASONS == [2023, 2024]
    assert train_core.VARIANTS == ["recent_raw_game_type", "recent_drop_game_type"]


def test_two_variants_differ_only_by_game_type() -> None:
    raw = core.feature_set("recent_raw_game_type")
    drop = core.feature_set("recent_drop_game_type")

    assert "game_type" in raw
    assert "game_type" not in drop
    assert [feature for feature in raw if feature != "game_type"] == drop


def test_success_state_is_present_in_both_variants() -> None:
    raw = set(core.feature_set("recent_raw_game_type"))
    drop = set(core.feature_set("recent_drop_game_type"))

    assert set(core.asof_core.SUCCESS_STATE).issubset(raw)
    assert set(core.asof_core.SUCCESS_STATE).issubset(drop)
