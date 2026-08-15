from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_catboost_ablation.py"

spec = importlib.util.spec_from_file_location("run_catboost_ablation", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_raw_score_preserves_negative_skill() -> None:
    y = np.array([0.0, 1.0] * 50, dtype=np.float64)
    p = np.full_like(y, 0.9)
    result = module.metrics(y, p)

    assert result["raw_competition_score"] < 0.0
    assert result["competition_score"] == 0.0
    assert np.isclose(
        result["raw_competition_score"],
        100000.0 * result["brier_skill"],
    )
