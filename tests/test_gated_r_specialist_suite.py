from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_gated_r_specialist_suite.py"
spec = importlib.util.spec_from_file_location("gated_r_specialist_suite", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_gated_prediction_leaves_non_r_on_base() -> None:
    old = np.asarray([0.2, 0.3, 0.4, 0.5])
    recent = np.asarray([0.4, 0.5, 0.6, 0.7])
    specialist = np.asarray([0.9, 0.1, 0.8, 0.2])
    is_r = np.asarray([True, False, True, False])
    out = mod.gated_prediction(old, recent, specialist, is_r, alpha_recent=0.25, beta_r=0.5)
    base = 0.75 * old + 0.25 * recent
    assert np.allclose(out[~is_r], base[~is_r])
    assert np.allclose(out[is_r], 0.5 * base[is_r] + 0.5 * specialist[is_r])


def test_quadratic_brier_matches_direct_brier() -> None:
    y = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0])
    p1 = np.asarray([0.1, 0.8, 0.6, 0.3, 0.7])
    p2 = np.asarray([0.2, 0.7, 0.5, 0.4, 0.9])
    p3 = np.asarray([0.3, 0.6, 0.9, 0.2, 0.8])
    mask = np.asarray([True, True, False, True, True])
    coeff = np.asarray([0.3, 0.4, 0.3])
    moment = mod._moments(y, [p1, p2, p3], mask)
    calculated = mod.quadratic_brier(moment, coeff)
    direct_p = coeff[0] * p1 + coeff[1] * p2 + coeff[2] * p3
    direct = float(np.mean((y[mask] - direct_p[mask]) ** 2))
    assert np.isclose(calculated, direct)


def test_grid_beta_zero_matches_base_everywhere() -> None:
    y = np.asarray([0.0, 1.0, 1.0, 0.0])
    old = np.asarray([0.2, 0.4, 0.6, 0.8])
    recent = np.asarray([0.3, 0.5, 0.7, 0.9])
    specialist = np.asarray([0.9, 0.1, 0.2, 0.3])
    is_r = np.asarray([True, False, True, False])
    is_f = ~is_r
    rows = mod.evaluate_gated_grid(
        y=y,
        is_r=is_r,
        is_f=is_f,
        p_old=old,
        p_recent=recent,
        p_specialist=specialist,
        alpha_values=np.asarray([0.4]),
        beta_values=np.asarray([0.0]),
    )
    assert len(rows) == 1
    base = 0.6 * old + 0.4 * recent
    expected = float(np.mean((y - base) ** 2))
    assert np.isclose(rows[0]["brier"], expected)


def test_specialist_feature_sets_exclude_game_type() -> None:
    base = [
        "game_type", "game_month", "inning", "top_bottom", "balls_before", "strikes_before",
        "outs_before", "base_state", "li", "pitcher_hand", "batter_hand", "asof_pitcher_n",
        "asof_pitcher_success_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "eng_ps_recent_mean_135",
        "eng_ps_recent_mean_minus_long", "eng_ps_recent_range_135", "eng_ps_prev1_minus_long",
        "eng_ps_prev3_minus_long", "eng_ps_prev5_minus_long", "eng_ps_prev1_minus_prev3",
        "eng_ps_prev3_minus_prev5", "eng_ps_prev1_minus_prev5",
    ]
    sets = mod._feature_sets(base)
    for name in ["r_full", "r_fast", "r_range", "r_both", "stable_drop_gt"]:
        assert "game_type" not in sets[name]
    assert "game_type" in sets["recent_raw"]
    assert "game_type" in sets["full_raw"]
