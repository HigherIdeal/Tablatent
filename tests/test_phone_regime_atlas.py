from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_phone_regime_atlas as atlas


def _synthetic_flip_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for year in atlas.ALL_YEARS:
        for pitcher in range(12):
            for game_type in ["F", "R"]:
                n = 200
                if year <= 2022:
                    rate = 0.68 if game_type == "F" else 0.48
                else:
                    rate = 0.46 if game_type == "F" else 0.52
                y = rng.binomial(1, rate, size=n)
                rows.extend(
                    {
                        "season": year,
                        "control_success": int(v),
                        "pitcher_id": f"P{pitcher}",
                        "game_type": game_type,
                        "balls_before": pitcher % 4,
                        "strikes_before": pitcher % 3,
                        "asof_pitcher_n": 1000 + 50 * pitcher,
                    }
                    for v in y
                )
    frame = pd.DataFrame(rows)
    frame["season"] = frame["season"].astype(np.int16)
    frame["control_success"] = frame["control_success"].astype(np.float32)
    return frame


def test_quantile_groups_are_target_independent_and_bounded() -> None:
    series = pd.Series(np.arange(1000, dtype=float))
    groups, grouping, count = atlas._quantile_groups(series, bins=6)
    assert grouping == "quantile_bins"
    assert 5 <= count <= 6
    assert len(groups) == len(series)


def test_game_type_flip_is_detected_as_regime_candidate() -> None:
    frame = _synthetic_flip_frame()
    groups, _, _ = atlas._make_groups(frame["game_type"], categorical=True, bins=6)
    cohort, _ = atlas._same_entity_mask(frame, "pitcher_id", min_old=100, min_recent=100)
    summary, _ = atlas._audit_one(
        "game_type",
        groups,
        frame,
        min_era_count=500,
        min_year_count=100,
        cohort_mask=cohort,
    )
    assert summary["classification"] == "regime_candidate"
    assert summary["old_recent_effect_rmse"] > 0.05
    assert summary["shock_2022_2023_rmse"] > summary["post_2023_2024_rmse"]
    assert summary["sign_flip_rate"] > 0.9
    assert summary["same_pitcher_vs_all_ratio"] > 0.7


def test_same_pitcher_mask_requires_presence_in_both_eras() -> None:
    frame = pd.DataFrame(
        {
            "season": [2019, 2020, 2023, 2024, 2019, 2020],
            "pitcher_id": ["A", "A", "A", "A", "B", "B"],
            "control_success": [1, 0, 1, 0, 1, 1],
        }
    )
    mask, summary = atlas._same_entity_mask(frame, "pitcher_id", min_old=2, min_recent=2)
    assert summary["eligible_entities"] == 1
    assert mask.tolist() == [True, True, True, True, False, False]


def test_interaction_builder_game_type_count() -> None:
    frame = pd.DataFrame(
        {
            "game_type": ["F", "R"],
            "balls_before": [3, 0],
            "strikes_before": [2, 1],
        }
    )
    groups = atlas._build_interaction(frame, "ix_game_type_x_count", bins=6)
    assert groups.tolist() == ["F|3|2", "R|0|1"]


def test_js_divergence_zero_for_same_composition() -> None:
    a = pd.DataFrame({"group": ["A", "B"], "count": [100, 50]})
    b = pd.DataFrame({"group": ["A", "B"], "count": [200, 100]})
    assert np.isclose(atlas._js_divergence(a, b), 0.0)
