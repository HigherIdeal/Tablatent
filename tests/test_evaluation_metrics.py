import numpy as np

from src.evaluation_metrics import probability_metrics


def test_raw_score_preserves_negative_skill_while_clipped_score_is_zero():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    metric = probability_metrics(y, p)

    assert metric["brier"] > metric["reference_brier"]
    assert metric["raw_score"] < 0.0
    assert metric["clipped_score"] == 0.0
    assert metric["competition_score"] == metric["clipped_score"]
