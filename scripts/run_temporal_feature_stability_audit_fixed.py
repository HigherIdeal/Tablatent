from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_temporal_feature_stability_audit as core


def _internal_rmse_renamed_safe(profiles, years, era_profile):
    """Compatibility fix for era profiles renamed to old_effect/recent_effect.

    The original audit renames aggregate columns before calling _internal_rmse,
    while _internal_rmse expects a column literally named ``effect``.  Accept
    exactly one supported effect column and keep the original weighting logic.
    """
    effect_columns = [
        column
        for column in ("effect", "old_effect", "recent_effect")
        if column in era_profile.columns
    ]
    if len(effect_columns) != 1:
        raise ValueError(
            "era_profile must contain exactly one of effect/old_effect/recent_effect; "
            f"got columns={list(era_profile.columns)}"
        )

    total_sq = 0.0
    total_weight = 0.0
    era_effect = era_profile[effect_columns[0]].to_dict()
    for year in years:
        prof = profiles[year]
        for row in prof.itertuples(index=False):
            if row.group not in era_effect:
                continue
            diff = float(row.effect) - float(era_effect[row.group])
            weight = float(row.count)
            total_sq += weight * diff * diff
            total_weight += weight

    if total_weight <= 0:
        return float("nan")
    return float(math.sqrt(total_sq / total_weight))


def main() -> None:
    core._internal_rmse = _internal_rmse_renamed_safe
    core.main()


if __name__ == "__main__":
    main()
