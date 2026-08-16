from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_gated_r_specialist_submission.py"
spec = importlib.util.spec_from_file_location("build_gated_r_specialist_submission", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_gated_prediction_changes_only_r_rows() -> None:
    p_full = np.asarray([0.40, 0.50, 0.60])
    p_recent = np.asarray([0.50, 0.40, 0.70])
    p_r_fast = np.asarray([0.90, 0.10, 0.20])
    is_r = np.asarray([True, False, True])
    out = mod.gated_prediction(
        p_full,
        p_recent,
        p_r_fast,
        is_r,
        alpha_recent=0.2,
        beta_r=0.1,
    )
    base = 0.8 * p_full + 0.2 * p_recent
    expected = base.copy()
    expected[is_r] = 0.9 * base[is_r] + 0.1 * p_r_fast[is_r]
    assert np.allclose(out, expected)
    assert np.isclose(out[1], base[1])


def test_default_training_seasons_match_final_2025_policy() -> None:
    assert mod.FULL_TRAIN_SEASONS == [2019, 2020, 2021, 2022, 2023, 2024]
    assert mod.RECENT_TRAIN_SEASONS == [2023, 2024]


def test_r_fast_feature_set_is_small_and_has_no_game_type() -> None:
    base = mod.recent_core.feature_set("recent_raw_game_type")
    sets = mod.gated_core._feature_sets(base)
    assert "game_type" in sets["full_raw"]
    assert "game_type" in sets["recent_raw"]
    assert "game_type" not in sets["r_fast"]
    assert len(sets["r_fast"]) == 17
    assert "asof_pitcher_fastball_rate" in sets["r_fast"]
    assert "batter_hand" in sets["r_fast"]
