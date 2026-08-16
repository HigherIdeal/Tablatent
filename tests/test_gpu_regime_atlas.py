from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_gpu_regime_atlas as gpu


def test_residual_profile_centers_each_season() -> None:
    groups = pd.Series(["A", "A", "B", "B"])
    residual = np.array([0.2, 0.0, -0.1, -0.1])
    season = pd.Series([2024, 2024, 2024, 2024])
    prof = gpu._residual_profile(groups, residual, season, 2024)
    weighted = np.average(prof["effect"].to_numpy(float), weights=prof["count"].to_numpy(float))
    assert np.isclose(weighted, 0.0)
    assert np.isclose(float(prof["season_prior"].iloc[0]), 0.0)


def test_recent_residual_structure_detects_persistent_groups() -> None:
    p23 = pd.DataFrame(
        {
            "season": [2023, 2023],
            "group": ["A", "B"],
            "count": [1000, 1000],
            "mean": [0.02, -0.02],
            "effect": [0.02, -0.02],
            "season_prior": [0.0, 0.0],
        }
    )
    p24 = p23.copy()
    p24["season"] = 2024
    result = gpu._recent_residual_structure(p23, p24, min_year_count=100)
    assert result["recent_supported_groups"] == 2
    assert np.isclose(result["recent_residual_rms"], 0.02)
    assert np.isclose(result["recent_residual_corr"], 1.0)
    assert np.isclose(result["recent_same_sign_rate"], 1.0)


def test_residual_audit_flags_persistent_recent_signal() -> None:
    rows = []
    groups = []
    residual = []
    for year in [2022, 2023, 2024]:
        for group in ["A", "B"]:
            for _ in range(500):
                rows.append({"season": year})
                groups.append(group)
                if year == 2022:
                    residual.append(0.0)
                else:
                    residual.append(0.02 if group == "A" else -0.02)
    frame = pd.DataFrame(rows)
    summary, _ = gpu._audit_residual_signal(
        "synthetic",
        pd.Series(groups),
        frame,
        np.asarray(residual, dtype=float),
        min_year_count=100,
    )
    assert summary["recent_residual_rms"] > 0.015
    assert summary["recent_residual_corr"] > 0.99
    assert summary["recent_same_sign_rate"] > 0.99
    assert summary["residual_classification"] == "persistent_recent_residual"


def test_evidence_label_separates_new_expert_from_absorbed_shift() -> None:
    strong = pd.Series(
        {
            "classification": "regime_candidate",
            "raw_residual_classification": "persistent_recent_residual",
            "raw_recent_residual_rms": 0.005,
            "drop_recent_residual_rms": 0.006,
        }
    )
    absorbed = pd.Series(
        {
            "classification": "regime_candidate",
            "raw_residual_classification": "weak",
            "raw_recent_residual_rms": 0.001,
            "drop_recent_residual_rms": 0.004,
        }
    )
    assert gpu._evidence_label(strong) == "STRONG_NEW_EXPERT_CANDIDATE"
    assert gpu._evidence_label(absorbed) == "RAW_GAME_TYPE_ABSORBS_SHIFT"


def test_parse_years_rejects_unsorted_duplicates() -> None:
    assert gpu._parse_years("2022,2023,2024") == [2022, 2023, 2024]
    try:
        gpu._parse_years("2023,2022")
    except ValueError:
        pass
    else:
        raise AssertionError("unsorted fold years must be rejected")
