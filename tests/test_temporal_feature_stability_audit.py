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

import run_temporal_feature_stability_audit as audit


def test_weighted_rmse() -> None:
    value = audit._weighted_rmse(np.array([1.0, 3.0]), np.array([1.0, 3.0]))
    assert np.isclose(value, np.sqrt(7.0))


def test_classify_clear_changepoint_as_regime_sensitive() -> None:
    label = audit._classify(
        shift_rmse=0.020,
        old_internal=0.003,
        recent_internal=0.002,
        sign_flip_rate=0.8,
    )
    assert label == "regime_sensitive"


def test_classify_small_shift_as_stable() -> None:
    label = audit._classify(
        shift_rmse=0.001,
        old_internal=0.002,
        recent_internal=0.0015,
        sign_flip_rate=0.0,
    )
    assert label == "stable"


def test_audit_feature_detects_synthetic_regime_flip() -> None:
    rows = []
    rng = np.random.default_rng(7)
    for year in audit.ALL_YEARS:
        for group in ["F", "R"]:
            n = 4000
            if year <= 2022:
                rate = 0.65 if group == "F" else 0.45
            else:
                rate = 0.45 if group == "F" else 0.55
            y = rng.binomial(1, rate, size=n)
            rows.extend(
                {
                    "season": year,
                    "game_type": group,
                    "control_success": int(target),
                }
                for target in y
            )
    frame = pd.DataFrame(rows)
    summary, long_df = audit.audit_feature(
        frame=frame,
        feature="game_type",
        target_col="control_success",
        season_col="season",
        categorical_features={"game_type"},
        numeric_bins=10,
        max_auto_categories=20,
        min_era_count=500,
        min_effect_for_flip=0.002,
    )
    assert summary["classification"] == "regime_sensitive"
    assert summary["sign_flip_rate"] > 0.9
    assert summary["old_recent_effect_rmse"] > 0.05
    assert set(long_df["season"].unique()) == set(audit.ALL_YEARS)


def test_numeric_grouping_is_target_independent_and_shared() -> None:
    series = pd.Series(np.arange(100, dtype=float))
    groups, grouping, count = audit._make_groups(
        series,
        categorical=False,
        numeric_bins=10,
        max_auto_categories=5,
    )
    assert grouping == "quantile_bins"
    assert 8 <= count <= 10
    assert len(groups) == len(series)
