from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_model_family_screen as screen


def test_native_categorical_vocab_is_fit_only() -> None:
    fit = pd.DataFrame(
        {
            "pitcher_hand": ["L", "R", "L"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    apply = pd.DataFrame(
        {
            "pitcher_hand": ["R", "S"],
            "value": [4.0, 5.0],
        }
    )

    x_fit, x_apply, categorical = screen.prepare_native_categorical_pair(
        fit, apply, ["pitcher_hand", "value"]
    )

    assert categorical == ["pitcher_hand"]
    assert isinstance(x_fit["pitcher_hand"].dtype, pd.CategoricalDtype)
    assert isinstance(x_apply["pitcher_hand"].dtype, pd.CategoricalDtype)
    assert list(x_fit["pitcher_hand"].cat.categories) == ["L", "R"]
    assert list(x_apply["pitcher_hand"].cat.categories) == ["L", "R"]
    assert x_apply["pitcher_hand"].isna().tolist() == [False, True]
    assert x_fit["value"].dtype == np.float32
    assert x_apply["value"].dtype == np.float32


def test_brier_cost_matches_manual_value() -> None:
    y = np.array([0.0, 1.0, 1.0])
    p = np.array([0.2, 0.8, 0.4])
    expected = ((0.2 - 0.0) ** 2 + (0.8 - 1.0) ** 2 + (0.4 - 1.0) ** 2) / 3.0
    assert abs(screen.brier_cost(y, p) - expected) < 1e-12


def test_model_feature_variants_keep_only_expected_hand_cross() -> None:
    success = screen.feature_set("success_state")
    hand = screen.feature_set("success_plus_hand_matchup")
    assert "ctx_hand_matchup" not in success
    assert "ctx_hand_matchup" in hand
    assert len(hand) == len(success) + 1
