from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

screen = importlib.import_module("run_dual_track_blend_screen")
builder = importlib.import_module("build_dual_track_submission")
recent_core = importlib.import_module("build_recent_regime_submissions")


def test_dual_track_feature_sets_differ_only_by_game_type() -> None:
    recent = recent_core.feature_set(screen.RECENT_VARIANT)
    stable = recent_core.feature_set(screen.STABLE_VARIANT)
    assert "game_type" in recent
    assert "game_type" not in stable
    assert set(recent) - {"game_type"} == set(stable)


def test_both_experts_keep_success_state_features() -> None:
    import run_asof_state_engineering as asof_core

    recent = set(recent_core.feature_set(screen.RECENT_VARIANT))
    stable = set(recent_core.feature_set(screen.STABLE_VARIANT))
    assert set(asof_core.SUCCESS_STATE).issubset(recent)
    assert set(asof_core.SUCCESS_STATE).issubset(stable)


def test_analytic_alpha_recovers_interior_optimum() -> None:
    y = np.array([0.2, 0.8], dtype=float)
    p_recent = np.array([0.0, 1.0], dtype=float)
    p_stable = np.array([0.4, 0.6], dtype=float)
    alpha = screen.clipped_analytic_alpha(y, p_recent, p_stable)
    assert np.isclose(alpha, 0.5)


def test_alpha_grid_contains_endpoints() -> None:
    values = screen.alpha_grid(0.05)
    assert np.isclose(values[0], 0.0)
    assert np.isclose(values[-1], 1.0)


def test_final_training_season_contract() -> None:
    assert builder.RECENT_TRAIN_SEASONS == [2023, 2024]
    assert builder.STABLE_TRAIN_SEASONS == [2019, 2020, 2021, 2022, 2023, 2024]
