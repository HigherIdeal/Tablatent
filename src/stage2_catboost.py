from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import load_frame, split_masks
from .knn_probability import _load_latents
from .utils import save_json, seed_everything


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    pred_class = (p >= threshold).astype(np.float32)
    bce = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    return {
        "bce": float(bce),
        "brier": float(np.mean(np.square(p - y))),
        "accuracy": float(np.mean(pred_class == y)),
        "auc": float(roc_auc_score(y, p)),
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
    }


def train_stage2(config: dict) -> dict:
    """Train CatBoost directly on the frozen 32-D Stage-1 posterior means."""
    try:
        import catboost
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError(
            "catboost가 없습니다. pip install -r configs/requirements.txt 를 실행하세요."
        ) from exc

    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    z = _load_latents(config)
    if len(z) != len(frame):
        raise ValueError(f"latent rows={len(z):,}, frame rows={len(frame):,} 불일치")

    context_dim = int(config["stage1"]["context"]["latent_dim"])
    history_dim = int(config["stage1"]["history"]["latent_dim"])
    if z.shape[1] != context_dim + history_dim:
        raise ValueError(
            f"latent_dim={z.shape[1]} but context+history={context_dim + history_dim}"
        )

    target_col = config["data"]["target_col"]
    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=np.float32)
    train_mask = split["train"]
    val_mask = split["val"]
    train_z = np.asarray(z[train_mask], dtype=np.float32)
    val_z = np.asarray(z[val_mask], dtype=np.float32)
    train_y = y[train_mask]
    val_y = y[val_mask]

    feature_names = [f"context_z{i:02d}" for i in range(context_dim)] + [
        f"history_z{i:02d}" for i in range(history_dim)
    ]
    train_pool = Pool(train_z, label=train_y, feature_names=feature_names)
    val_pool = Pool(val_z, label=val_y, feature_names=feature_names)

    cfg = config.get("stage2_catboost", {})
    threshold = float(cfg.get("threshold", 0.5))
    task_type = str(cfg.get("task_type", "GPU")).upper()
    devices = str(cfg.get("devices", "0"))

    # BrierScore is our primary reported probability metric, but CatBoost does not
    # provide GPU support for it as a training/eval metric. Keep the GPU probe fast
    # and reproducible by selecting trees with Logloss, then compute Brier exactly
    # from validation probabilities after training.
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": int(cfg.get("iterations", 2000)),
        "learning_rate": float(cfg.get("learning_rate", 0.03)),
        "depth": int(cfg.get("depth", 6)),
        "l2_leaf_reg": float(cfg.get("l2_leaf_reg", 10.0)),
        "random_strength": float(cfg.get("random_strength", 1.0)),
        "random_seed": int(config["seed"]),
        "task_type": task_type,
        "verbose": int(cfg.get("verbose", 50)),
        "allow_writing_files": False,
    }
    if task_type == "GPU":
        params["devices"] = devices
    thread_count = cfg.get("thread_count")
    if thread_count is not None:
        params["thread_count"] = int(thread_count)

    early_stopping_rounds = int(cfg.get("early_stopping_rounds", 100))
    model = CatBoostClassifier(**params)

    print(
        f"[CatBoost Stage2] train={len(train_z):,}, val={len(val_z):,}, "
        f"latent_dim={z.shape[1]}, task_type={task_type}, catboost={catboost.__version__}"
    )
    print(
        f"[CatBoost Stage2] frozen posterior means only: context({context_dim}D) + "
        f"history({history_dim}D) -> CatBoostClassifier; "
        "loss/selection=Logloss, reported primary metric=Brier"
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
    )

    val_pred = model.predict_proba(val_pool)[:, 1].astype(np.float32, copy=False)
    metrics = _binary_metrics(val_y, val_pred, threshold)

    train_mean = float(train_y.mean())
    val_rate = float(val_y.mean())
    train_mean_brier = float(
        np.mean(np.square(np.full_like(val_y, train_mean, dtype=np.float32) - val_y))
    )
    official_baseline_brier = float(val_rate * (1.0 - val_rate))
    official_style_score = max(
        0.0,
        100000.0 * (1.0 - metrics["brier"] / official_baseline_brier),
    )

    best_iteration_zero_based = int(model.get_best_iteration())
    best_iteration = best_iteration_zero_based + 1 if best_iteration_zero_based >= 0 else None
    output_dir = Path(config["paths"]["output_dir"]) / "stage2_catboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "stage2_catboost.cbm"))

    importance = model.get_feature_importance(type="FeatureImportance")
    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importance, dtype=float)}
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(output_dir / "feature_importance.csv", index=False)

    val_global = np.flatnonzero(val_mask)
    pred_frame = pd.DataFrame(
        {
            "global_index": val_global,
            "target": val_y,
            "probability": val_pred,
            "predicted_class": (val_pred >= threshold).astype(np.int8),
        }
    )
    row_id_col = config["data"].get("row_id_col")
    if row_id_col and row_id_col in frame.columns:
        pred_frame.insert(1, row_id_col, frame.loc[val_mask, row_id_col].to_numpy())
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    result = {
        "model": "CatBoostClassifier",
        "catboost_version": catboost.__version__,
        "input": "frozen Stage1 posterior means [mu_context; mu_history]",
        "latent_dim": int(z.shape[1]),
        "context_dim": context_dim,
        "history_dim": history_dim,
        "train_rows": int(len(train_z)),
        "val_rows": int(len(val_z)),
        "best_iteration": best_iteration,
        "tree_count": int(model.tree_count_),
        "selection_metric": "validation Logloss",
        "primary_report_metric": "validation Brier computed from predict_proba",
        "training_loss": "Logloss",
        "validation": metrics,
        "baseline": {
            "train_mean_probability": train_mean,
            "train_mean_brier_on_validation": train_mean_brier,
            "validation_rate": val_rate,
            "official_baseline_brier": official_baseline_brier,
            "official_style_score": official_style_score,
        },
        "params": params,
        "early_stopping_rounds": early_stopping_rounds,
        "standardized_latent": False,
    }
    save_json(result, output_dir / "metrics.json")

    print("\n[CatBoost Stage2 validation]")
    print(f"best iteration             : {best_iteration}")
    print(f"validation BCE             : {metrics['bce']:.8f}")
    print(f"validation Brier           : {metrics['brier']:.8f}")
    print(f"validation accuracy        : {metrics['accuracy']:.6f}")
    print(f"validation AUC             : {metrics['auc']:.6f}")
    print(
        f"prediction mean / std      : {metrics['prediction_mean']:.6f} / "
        f"{metrics['prediction_std']:.6f}"
    )
    print(f"official baseline Brier    : {official_baseline_brier:.8f}")
    print(f"official-style score       : {official_style_score:.2f}")
    return result
