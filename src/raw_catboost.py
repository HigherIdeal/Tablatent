from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import load_frame, split_masks
from .utils import save_json, seed_everything


DEFAULT_ID_COLUMNS = [
    "row_id",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
]

DEFAULT_CATEGORICAL_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "base_state",
    "pitcher_hand",
    "batter_hand",
]


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


def _prepare_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    x = frame.loc[:, feature_columns].copy()
    categorical_set = set(categorical_columns)
    for col in feature_columns:
        if col in categorical_set:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(np.float32)
            x[col] = x[col].replace([np.inf, -np.inf], np.nan)
    return x


def train_raw_catboost(config: dict) -> dict:
    """Raw train.csv CatBoost baseline using every non-target feature except IDs."""
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

    cfg = config.get("raw_catboost", {})
    target_col = config["data"]["target_col"]
    id_columns = list(cfg.get("exclude_id_columns", DEFAULT_ID_COLUMNS))
    excluded = set(id_columns + [target_col])
    missing_ids = [col for col in id_columns if col not in frame.columns]
    if missing_ids:
        raise ValueError(f"configured ID columns missing from data: {missing_ids}")

    feature_columns = [col for col in frame.columns if col not in excluded]
    if not feature_columns:
        raise ValueError("no raw CatBoost features remain after exclusions")

    requested_categorical = list(
        cfg.get("categorical_columns", DEFAULT_CATEGORICAL_COLUMNS)
    )
    categorical_columns = [
        col for col in requested_categorical if col in feature_columns
    ]

    unlisted_object = [
        col
        for col in feature_columns
        if col not in categorical_columns
        and (
            pd.api.types.is_object_dtype(frame[col].dtype)
            or isinstance(frame[col].dtype, pd.StringDtype)
            or isinstance(frame[col].dtype, pd.CategoricalDtype)
        )
    ]
    if unlisted_object:
        raise ValueError(
            "raw_catboost.categorical_columns에 추가해야 하는 문자열 컬럼: "
            + ", ".join(unlisted_object)
        )

    train_mask = split["train"]
    val_mask = split["val"]
    train_frame = frame.loc[train_mask]
    val_frame = frame.loc[val_mask]
    train_x = _prepare_features(train_frame, feature_columns, categorical_columns)
    val_x = _prepare_features(val_frame, feature_columns, categorical_columns)
    train_y = pd.to_numeric(train_frame[target_col], errors="raise").to_numpy(
        dtype=np.float32
    )
    val_y = pd.to_numeric(val_frame[target_col], errors="raise").to_numpy(
        dtype=np.float32
    )

    train_pool = Pool(
        train_x,
        label=train_y,
        cat_features=categorical_columns,
        feature_names=feature_columns,
    )
    val_pool = Pool(
        val_x,
        label=val_y,
        cat_features=categorical_columns,
        feature_names=feature_columns,
    )

    threshold = float(cfg.get("threshold", 0.5))
    task_type = str(cfg.get("task_type", "GPU")).upper()
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": int(cfg.get("iterations", 3000)),
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
        params["devices"] = str(cfg.get("devices", "0"))
    thread_count = cfg.get("thread_count")
    if thread_count is not None:
        params["thread_count"] = int(thread_count)

    early_stopping_rounds = int(cfg.get("early_stopping_rounds", 100))
    model = CatBoostClassifier(**params)

    print(
        f"[Raw CatBoost] train={len(train_x):,}, val={len(val_x):,}, "
        f"features={len(feature_columns)}, categorical={len(categorical_columns)}, "
        f"task_type={task_type}, catboost={catboost.__version__}"
    )
    print(f"[Raw CatBoost] excluded IDs: {', '.join(id_columns)}")
    print("[Raw CatBoost] Stage1 latent is NOT used; raw non-ID columns only")

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
    )

    val_pred = model.predict_proba(val_pool)[:, 1].astype(np.float32, copy=False)
    metrics = _binary_metrics(val_y, val_pred, threshold)

    val_rate = float(val_y.mean())
    official_baseline_brier = float(val_rate * (1.0 - val_rate))
    official_style_score = max(
        0.0,
        100000.0 * (1.0 - metrics["brier"] / official_baseline_brier),
    )
    best_zero = int(model.get_best_iteration())
    best_iteration = best_zero + 1 if best_zero >= 0 else None

    output_dir = Path(config["paths"]["output_dir"]) / "raw_catboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "raw_catboost.cbm"))

    importance = model.get_feature_importance(type="FeatureImportance")
    pd.DataFrame(
        {"feature": feature_columns, "importance": np.asarray(importance, dtype=float)}
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )

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
        pred_frame.insert(
            1,
            row_id_col,
            frame.loc[val_mask, row_id_col].to_numpy(),
        )
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    result = {
        "model": "CatBoostClassifier",
        "catboost_version": catboost.__version__,
        "input": "raw train.csv features excluding target and configured ID columns",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "excluded_id_columns": id_columns,
        "train_rows": int(len(train_x)),
        "val_rows": int(len(val_x)),
        "best_iteration": best_iteration,
        "tree_count": int(model.tree_count_),
        "validation": metrics,
        "official_baseline_brier": official_baseline_brier,
        "official_style_score": official_style_score,
        "params": params,
        "early_stopping_rounds": early_stopping_rounds,
    }
    save_json(result, output_dir / "metrics.json")

    print("\n[Raw CatBoost validation]")
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
