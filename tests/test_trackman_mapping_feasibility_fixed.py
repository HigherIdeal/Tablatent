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

import run_trackman_mapping_feasibility_fixed as fixed


def test_null_consistency_uses_only_assignment_columns() -> None:
    matches = pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2022, 2023, 2024],
            "main_pitcher_id": ["A", "A", "A", "B", "B", "B"],
            "best_trackman_id": ["X", "X", "X", "Y", "Z", "Y"],
        }
    )
    scores = fixed._null_consistency_fixed(matches, repetitions=8, seed=42)
    assert scores.shape == (8,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_null_consistency_is_deterministic_for_seed() -> None:
    matches = pd.DataFrame(
        {
            "season": [2022, 2023, 2024, 2022, 2023, 2024],
            "main_pitcher_id": ["A", "A", "A", "B", "B", "B"],
            "best_trackman_id": ["X", "X", "X", "Y", "Z", "Y"],
        }
    )
    a = fixed._null_consistency_fixed(matches, repetitions=10, seed=7)
    b = fixed._null_consistency_fixed(matches, repetitions=10, seed=7)
    assert np.allclose(a, b, equal_nan=True)
