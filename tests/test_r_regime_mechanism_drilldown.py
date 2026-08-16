from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_r_regime_mechanism_drilldown as drill


def test_mechanism_table_detects_2023_break_and_persistence() -> None:
    counts = np.full((6, 2), 1000.0)
    effects = np.asarray(
        [
            [-0.01, +0.01],
            [-0.01, +0.01],
            [-0.01, +0.01],
            [-0.01, +0.01],
            [+0.02, -0.02],
            [+0.018, -0.018],
        ],
        dtype=float,
    )
    table = drill._mechanism_table(
        counts,
        effects,
        min_old_count=500,
        min_recent_count=300,
    ).set_index("group_code")

    assert np.isclose(table.loc[0, "break_22_23"], 0.03)
    assert np.isclose(table.loc[1, "break_22_23"], -0.03)
    assert table.loc[0, "persistent_side_2024"] == 1.0
    assert table.loc[1, "persistent_side_2024"] == 1.0


def test_summary_weights_supported_groups() -> None:
    counts = np.full((6, 1), 1000.0)
    effects = np.asarray([[-0.01], [-0.01], [-0.01], [-0.01], [0.02], [0.018]], dtype=float)
    table = drill._mechanism_table(
        counts,
        effects,
        min_old_count=500,
        min_recent_count=300,
    )
    summary = drill._summary(table)
    assert summary["supported_groups"] == 1
    assert summary["weighted_abs_break_22_23"] > summary["weighted_abs_change_23_24"]
    assert summary["persistent_side_share_2024"] == 1.0


def test_same_player_delta_correlation() -> None:
    import pandas as pd

    full = pd.DataFrame({"group_code": [0, 1, 2], "regime_delta": [-0.02, 0.0, 0.03]})
    same = pd.DataFrame({"group_code": [0, 1, 2], "regime_delta": [-0.01, 0.0, 0.015]})
    corr = drill._same_player_delta_correlation(full, same)
    assert np.isclose(corr, 1.0)
