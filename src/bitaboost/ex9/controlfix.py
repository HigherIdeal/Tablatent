from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bitaboost.baseline import _params, _prepare_x
from bitaboost.features import AUX_NAMES


def fit_direct_weighted_exact_scale(
    cfg: dict[str, Any],
    tr: pd.DataFrame,
    va: pd.DataFrame,
    aux_train: pd.DataFrame,
    features: list[str],
    domain_weight: np.ndarray,
) -> np.ndarray:
    """EX9 direct fit with baseline-preserving weight scale.

    The original EX9 normalized the final CatBoost sample weights to mean 1.0.
    SAFE's direct head does not do that, so alpha=0 changed the effective
    regularization strength and failed to reproduce the exact retrain control.

    Here the density-weighted vector is normalized to the *same total weight as
    SAFE's original R/F weighting*. Therefore alpha=0 is bit-for-bit equivalent
    to the SAFE direct training weight vector (up to the existing float32 cast),
    while alpha>0 changes only relative sample importance.
    """
    from catboost import CatBoostRegressor, Pool

    x, cats = _prepare_x(tr, features)
    xv, _ = _prepare_x(va, features)
    keep = aux_train[list(AUX_NAMES)].notna().all(axis=1).to_numpy()

    success = tr.loc[keep, "control_success"].to_numpy(np.float32)
    av = aux_train.loc[keep, list(AUX_NAMES)].to_numpy(np.float32)
    repeats = int(cfg["recipe"]["direct"]["success_repeats"])
    labels = np.column_stack([*[success] * repeats, av])

    f_weight = float(cfg["recipe"]["direct"]["f_weight"])
    base_weight = np.where(
        tr.loc[keep, "game_type"].astype(str).to_numpy() == "F", f_weight, 1.0
    ).astype(np.float64)

    density = np.asarray(domain_weight, np.float64)[keep]
    weight = base_weight * density

    # Preserve the exact total scale of the SAFE direct-head weights.  This is a
    # controlled reweighting experiment, not a regularization-strength experiment.
    total = float(weight.sum())
    baseline_total = float(base_weight.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("EX9 invalid weighted sample total")
    weight *= baseline_total / total

    pool = Pool(
        x.loc[keep],
        labels,
        weight=weight.astype(np.float32),
        cat_features=cats,
        feature_names=features,
    )
    vp = Pool(xv, cat_features=cats, feature_names=features)
    model = CatBoostRegressor(**_params(cfg, "MultiRMSE")).fit(pool)
    pred = np.clip(
        model.predict(vp, ntree_end=int(cfg["recipe"]["direct"]["tree"])),
        0.0,
        1.0,
    )[:, 0].astype(np.float64)
    return pred
