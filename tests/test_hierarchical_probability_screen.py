from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_hierarchical_probability_screen as hp


def test_smoothed_probability_handles_unseen_child_without_readonly_failure() -> None:
    source = pd.DataFrame(
        {
            "balls_before": [0, 0, 0, 1],
            "strikes_before": [0, 0, 0, 0],
            "game_type": ["R", "R", "F", "R"],
            "control_success": [1, 0, 1, 0],
        }
    )
    target_frame = pd.DataFrame(
        {
            "balls_before": [0, 0, 1],
            "strikes_before": [0, 0, 0],
            "game_type": ["R", "X", "R"],
        }
    )

    probability, diagnostics = hp.apply_smoothed_probability(
        source=source,
        target_frame=target_frame,
        keys=["balls_before", "strikes_before", "game_type"],
        parent_keys=["balls_before", "strikes_before"],
        target="control_success",
        alpha=2.0,
    )

    assert probability.shape == (3,)
    assert probability.dtype == np.float32
    assert np.isfinite(probability).all()
    assert 0.0 <= probability.min() <= probability.max() <= 1.0
    assert 0.0 < diagnostics["exact_match_fraction"] < 1.0


def test_smoothed_probability_empty_history_returns_neutral_half() -> None:
    source = pd.DataFrame(
        columns=["balls_before", "strikes_before", "control_success"]
    )
    target_frame = pd.DataFrame(
        {
            "balls_before": [0, 1],
            "strikes_before": [0, 0],
        }
    )

    probability, diagnostics = hp.apply_smoothed_probability(
        source=source,
        target_frame=target_frame,
        keys=["balls_before", "strikes_before"],
        parent_keys=[],
        target="control_success",
        alpha=200.0,
    )

    np.testing.assert_allclose(probability, np.array([0.5, 0.5], dtype=np.float32))
    assert diagnostics["exact_match_fraction"] == 0.0
