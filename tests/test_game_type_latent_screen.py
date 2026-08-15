from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_game_type_latent_screen as latent


def _context_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_month": [3, 4, 4],
            "game_dayofweek": ["Tue", "Wed", "Thu"],
            "top_bottom": ["T", "B", "T"],
            "base_state": ["___", "1__", "_2_"],
            "inning": [1, 5, 9],
            "balls_before": [0, 3, 1],
            "strikes_before": [0, 2, 1],
            "outs_before": [0, 1, 2],
            "run_total_before": [0, 4, 8],
            "score_diff_home": [0, 2, -2],
            "pitcher_team_win_expectancy": [0.50, 0.70, 0.30],
            "li": [1.0, 1.8, 0.7],
        }
    )


def test_encoder_context_excludes_temporal_target_and_identity_leakage() -> None:
    used = set(latent.CONTEXT_CATEGORICAL) | set(latent.CONTEXT_NUMERIC)
    forbidden = {
        "season",
        "game_type",
        "control_success",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
        "pitcher_hand",
        "batter_hand",
        "asof_pitcher_success_rate",
        "asof_batter_success_rate",
    }
    assert used.isdisjoint(forbidden)


def test_context_preprocessor_learns_categories_and_numeric_stats_from_fit_only() -> None:
    fit = _context_frame().iloc[:2].copy()
    apply = _context_frame().iloc[2:].copy()
    preprocessor = latent.GameContextPreprocessor.fit(fit)

    fit_cat, fit_num = preprocessor.transform(fit)
    apply_cat, apply_num = preprocessor.transform(apply)

    assert fit_cat.shape == (2, len(latent.CONTEXT_CATEGORICAL))
    assert fit_num.shape == (2, len(latent.CONTEXT_NUMERIC))
    assert apply_cat.shape == (1, len(latent.CONTEXT_CATEGORICAL))
    assert apply_num.shape == (1, len(latent.CONTEXT_NUMERIC))

    # Thu and _2_ occur only in apply rows and must map to reserved unknown index 0.
    dow_index = latent.CONTEXT_CATEGORICAL.index("game_dayofweek")
    base_index = latent.CONTEXT_CATEGORICAL.index("base_state")
    assert apply_cat[0, dow_index] == 0
    assert apply_cat[0, base_index] == 0
    assert np.isfinite(apply_num).all()


def test_game_type_encoding_is_strict_f_vs_r() -> None:
    encoded = latent.encode_game_type(pd.Series(["R", "F", "R", "F"]))
    np.testing.assert_array_equal(encoded, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))

    with pytest.raises(ValueError, match="Unexpected game_type"):
        latent.encode_game_type(pd.Series(["R", "X"]))


def test_feature_variants_replace_raw_game_type_as_intended() -> None:
    raw = latent.feature_set("raw_game_type", latent_dim=4)
    dropped = latent.feature_set("drop_game_type", latent_dim=4)
    soft = latent.feature_set("soft_proxy", latent_dim=4)
    proxy = latent.feature_set("latent_proxy", latent_dim=4)
    raw_plus = latent.feature_set("raw_plus_latent", latent_dim=4)
    latent_columns = latent.latent_columns(4)

    assert "game_type" in raw
    assert "game_type" not in dropped
    assert "game_type" not in soft and latent.SOFT_PROXY_COLUMN in soft
    assert "game_type" not in proxy and all(column in proxy for column in latent_columns)
    assert "game_type" in raw_plus and all(column in raw_plus for column in latent_columns)
    assert len(proxy) == len(set(proxy))
