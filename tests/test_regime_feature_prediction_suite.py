from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_regime_feature_prediction_suite.py"
spec = importlib.util.spec_from_file_location("regime_feature_suite", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2022, 2022, 2023, 2023, 2024, 2024, 2024, 2024],
            "game_type": ["R", "F", "R", "R", "R", "R", "F", "R"],
            "batter_hand": [1, 2, 1, 2, 1, 2, 1, 2],
            "asof_pitcher_fastball_rate": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
            "eng_ps_recent_range_135": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
        }
    )


def test_quantile_codes_use_training_edges() -> None:
    train = pd.Series([0.0, 1.0, 2.0, 3.0], name="x")
    edges = mod.fit_quantile_edges(train, bins=2)
    codes = mod.apply_quantile_codes(pd.Series([-10.0, 0.25, 2.75, 99.0, np.nan]), edges)
    assert codes.tolist() == [0, 0, 1, 1, -1]


def test_regime_continuous_masks_separate_old_recent_r_and_hand() -> None:
    raw = _frame()
    train = raw.iloc[:4].copy()
    valid = raw.iloc[4:].copy()
    mod.add_regime_features(train, valid, season_col="season", bins=2)

    assert train[mod.RECENT_FLAG].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert valid[mod.RECENT_FLAG].tolist() == [1.0, 1.0, 1.0, 1.0]

    # old R, hand=1 appears only in the first training row
    assert np.isclose(train.loc[0, "ro_fastball_hand1"], 0.40)
    assert train["rr_fastball_hand1"].notna().sum() == 1
    assert train["rr_fastball_hand2"].notna().sum() == 1

    # F rows must not leak into R-specific masks
    f_row = valid.index[valid["game_type"].eq("F")][0]
    assert pd.isna(valid.loc[f_row, "rr_fastball_hand1"])
    assert pd.isna(valid.loc[f_row, "rr_fastball_hand2"])
    assert valid.loc[f_row, mod.FAST_CAT] == "NON_R"
    assert valid.loc[f_row, mod.RANGE_CAT] == "NON_R"


def test_categorical_cross_marks_regime_and_hand() -> None:
    raw = _frame()
    train = raw.iloc[:4].copy()
    valid = raw.iloc[4:].copy()
    mod.add_regime_features(train, valid, season_col="season", bins=2)
    r_rows = valid.loc[valid["game_type"].eq("R")]
    assert r_rows[mod.FAST_CAT].str.startswith("RECENT_R|H=").all()
    assert r_rows[mod.RANGE_CAT].str.startswith("RECENT_R|RANGE=").all()


def test_conditional_bias_rmse_detects_group_bias_not_global_bias() -> None:
    y = np.asarray([1.0, 1.0, 0.0, 0.0])
    groups = np.asarray([0, 0, 1, 1])
    perfect_group_cal = np.asarray([1.0, 1.0, 0.0, 0.0])
    biased = np.asarray([0.5, 0.5, 0.5, 0.5])
    assert np.isclose(mod.conditional_bias_rmse(y, perfect_group_cal, groups, min_count=1), 0.0)
    assert mod.conditional_bias_rmse(y, biased, groups, min_count=1) > 0.49


def test_summary_uses_fixed_fold_weights() -> None:
    rows = []
    for fold, weight, recent_brier, full_brier in [
        ("a", 0.5, 0.20, 0.22),
        ("b", 0.2, 0.30, 0.31),
        ("c", 0.3, 0.40, 0.41),
    ]:
        for variant, policy, brier in [
            ("recent_base", "recent", recent_brier),
            ("full_base", "full", full_brier),
            ("full_regime_flag", "full", full_brier - 0.01),
        ]:
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "train_policy": policy,
                    "iterations": 200,
                    "brier": brier,
                    "raw_score": 0.0,
                    "r_brier": brier,
                    "f_brier": brier,
                    "r_fastball_hand_bias_rmse": 0.01,
                    "r_recent_range_bias_rmse": 0.02,
                }
            )
    result = mod.build_summary(pd.DataFrame(rows), {"a": 0.5, "b": 0.2, "c": 0.3})
    regime = result.loc[result["variant"].eq("full_regime_flag")].iloc[0]
    expected = 0.5 * 0.21 + 0.2 * 0.30 + 0.3 * 0.40
    assert np.isclose(regime["weighted_brier"], expected)
    assert np.isclose(regime["weighted_delta_vs_full_base"], -0.01)
