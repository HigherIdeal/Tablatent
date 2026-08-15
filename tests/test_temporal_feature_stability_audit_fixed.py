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

import run_temporal_feature_stability_audit_fixed as fixed


def test_internal_rmse_accepts_old_effect_column() -> None:
    profiles = {
        2019: pd.DataFrame(
            {
                "group": ["A", "B"],
                "count": [10, 20],
                "effect": [0.10, -0.05],
            }
        )
    }
    era = pd.DataFrame(
        {
            "old_count": [10, 20],
            "old_effect": [0.08, -0.04],
        },
        index=["A", "B"],
    )
    value = fixed._internal_rmse_renamed_safe(profiles, [2019], era)
    expected = np.sqrt((10 * 0.02**2 + 20 * 0.01**2) / 30)
    assert np.isclose(value, expected)


def test_internal_rmse_accepts_recent_effect_column() -> None:
    profiles = {
        2024: pd.DataFrame(
            {
                "group": ["F", "R"],
                "count": [100, 100],
                "effect": [-0.02, 0.02],
            }
        )
    }
    era = pd.DataFrame(
        {
            "recent_count": [100, 100],
            "recent_effect": [-0.03, 0.03],
        },
        index=["F", "R"],
    )
    value = fixed._internal_rmse_renamed_safe(profiles, [2024], era)
    assert np.isclose(value, 0.01)
