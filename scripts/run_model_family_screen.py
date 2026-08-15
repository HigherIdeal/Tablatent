from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_asof_state_engineering as asof_core
import run_context_interaction_screen as context_core
from src.canonical_features import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


FEATURE_VARIANTS: dict[str, list[str]] = {
    "success_state": list(asof_core.SUCCESS_STATE),
    "success_plus_hand_matchup": list(asof_core.SUCCESS_STATE) + ["ctx_hand_matchup"],
}
MODEL_FAMILIES = ["catboost", "lightgbm", "xgboost"]
ENGINEERED_CATEGORICAL = {"ctx_hand_matchup"}


def parse_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("at least one fold is required")
    return result


def parse_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def feature_set(variant: str) -> list[str]:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")
    result = list(CANONICAL_FEATURES) + list(FEATURE_VARIANTS[variant])
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate features in {variant}")
    return result


def categorical_columns(features: list[str]) -> list[str]:
    categorical = set(CANONICAL_CATEGORICAL) | ENGINEERED_CATEGORICAL
    return [feature for feature in features if feature in categorical]


def prepare_catboost(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = frame.loc[:, features].copy()
    categorical = categorical_columns(features)
    categorical_set = set(categorical)
    for column in features:
        if column in categorical_set:
            x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype(np.float32)
            x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x, categorical


def prepare_native_categorical_pair(
    fit_frame: pd.DataFrame,
    apply_frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare LightGBM/XGBoost frames using categories learned from fit_frame only.

    Validation/test-only category values become missing instead of expanding the
    training vocabulary. This keeps preprocessing deployment-safe.
    """
    x_fit = fit_frame.loc[:, features].copy()
    x_apply = apply_frame.loc[:, features].copy()
    categorical = categorical_columns(features)
    categorical_set = set(categorical)

    for column in features:
        if column in categorical_set:
            fit_tokens = x_fit[column].astype("string").fillna("<MISSING>").astype(str)
            apply_tokens = x_apply[column].astype("string").fillna("<MISSING>").astype(str)
            categories = pd.Index(pd.unique(fit_tokens))
            x_fit[column] = pd.Categorical(fit_tokens, categories=categories)
            x_apply[column] = pd.Categorical(apply_tokens, categories=categories)
        else:
            x_fit[column] = pd.to_numeric(x_fit[column], errors="coerce").astype(np.float32)
            x_apply[column] = pd.to_numeric(x_apply[column], errors="coerce").astype(np.float32)
            x_fit[column] = x_fit[column].replace([np.inf, -np.inf], np.nan)
            x_apply[column] = x_apply[column].replace([np.inf, -np.inf], np.nan)

    return x_fit, x_apply, categorical


def brier_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((y_pred - y_true) ** 2))


def lightgbm_brier(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    return "brier", brier_cost(y_true, y_pred), False


def _package_versions(models: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for model in models:
        try:
            if model == "catboost":
                import catboost

                versions[model] = catboost.__version__
            elif model == "lightgbm":
                import lightgbm

                versions[model] = lightgbm.__version__
            elif model == "xgboost":
                import xgboost

                versions[model] = xgboost.__version__
        except ImportError:
            versions[model] = "NOT_INSTALLED"
    return versions


def _require_packages(models: list[str]) -> None:
    missing: list[str] = []
    for model in models:
        try:
            if model == "catboost":
                import catboost  # noqa: F401
            elif model == "lightgbm":
                import lightgbm  # noqa: F401
            elif model == "xgboost":
                import xgboost  # noqa: F401
        except ImportError:
            missing.append(model)
    if missing:
        install_names = {"catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost"}
        packages = " ".join(install_names[x] for x in missing)
        raise RuntimeError(
            f"Missing packages: {missing}. Install in the active environment with: pip install {packages}"
        )


def select_catboost_iterations(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    max_estimators: int,
    early_stopping_rounds: int,
    seed: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> tuple[int, dict[str, float]]:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = prepare_catboost(train, features)
    x_valid, _ = prepare_catboost(valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
    y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)

    params = {
        "iterations": int(max_estimators),
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
        "random_seed": int(seed),
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "has_time": True,
        "one_hot_max_size": 10,
        "allow_writing_files": False,
        "task_type": task_type,
        "verbose": verbose,
    }
    if task_type == "GPU":
        params["devices"] = devices

    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, label=y_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose=verbose,
    )
    best_n = int(model.get_best_iteration()) + 1
    if best_n <= 0:
        best_n = int(model.tree_count_)
    prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
    metric = probability_metrics(y_valid, prediction)

    del model, train_pool, valid_pool, x_train, x_valid, y_train, y_valid, prediction
    gc.collect()
    return best_n, metric


def refit_predict_catboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    n_estimators: int,
    seed: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = prepare_catboost(train, features)
    x_valid, _ = prepare_catboost(valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
    params = {
        "iterations": int(n_estimators),
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
        "random_seed": int(seed),
        "loss_function": "Logloss",
        "has_time": True,
        "one_hot_max_size": 10,
        "allow_writing_files": False,
        "task_type": task_type,
        "verbose": verbose,
    }
    if task_type == "GPU":
        params["devices"] = devices

    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)
    prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)

    del model, train_pool, valid_pool, x_train, x_valid, y_train
    gc.collect()
    return prediction


def select_lightgbm_iterations(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    max_estimators: int,
    early_stopping_rounds: int,
    seed: int,
    device_type: str,
) -> tuple[int, dict[str, float]]:
    import lightgbm as lgb

    x_train, x_valid, categorical = prepare_native_categorical_pair(train, valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.int8)
    y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="None",
        n_estimators=int(max_estimators),
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        reg_alpha=0.0,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
        device_type=device_type,
    )
    callbacks = [
        lgb.early_stopping(int(early_stopping_rounds), first_metric_only=True, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric=lightgbm_brier,
        categorical_feature=categorical,
        callbacks=callbacks,
    )
    best_n = int(model.best_iteration_ or model.n_estimators_)
    prediction = model.predict_proba(x_valid, num_iteration=best_n)[:, 1].astype(np.float64)
    metric = probability_metrics(y_valid, prediction)

    del model, x_train, x_valid, y_train, y_valid, prediction
    gc.collect()
    return best_n, metric


def refit_predict_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    n_estimators: int,
    seed: int,
    device_type: str,
) -> np.ndarray:
    import lightgbm as lgb

    x_train, x_valid, categorical = prepare_native_categorical_pair(train, valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.int8)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        reg_alpha=0.0,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=-1,
        device_type=device_type,
    )
    model.fit(x_train, y_train, categorical_feature=categorical)
    prediction = model.predict_proba(x_valid)[:, 1].astype(np.float64)

    del model, x_train, x_valid, y_train
    gc.collect()
    return prediction


def select_xgboost_iterations(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    max_estimators: int,
    early_stopping_rounds: int,
    seed: int,
    device: str,
) -> tuple[int, dict[str, float]]:
    import xgboost as xgb

    x_train, x_valid, _ = prepare_native_categorical_pair(train, valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.int8)
    y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(max_estimators),
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        reg_alpha=0.0,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        eval_metric=brier_cost,
        early_stopping_rounds=int(early_stopping_rounds),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    best_iteration = getattr(model, "best_iteration", None)
    best_n = int(best_iteration) + 1 if best_iteration is not None else int(max_estimators)
    prediction = model.predict_proba(x_valid)[:, 1].astype(np.float64)
    metric = probability_metrics(y_valid, prediction)

    del model, x_train, x_valid, y_train, y_valid, prediction
    gc.collect()
    return best_n, metric


def refit_predict_xgboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    n_estimators: int,
    seed: int,
    device: str,
) -> np.ndarray:
    import xgboost as xgb

    x_train, x_valid, _ = prepare_native_categorical_pair(train, valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.int8)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(n_estimators),
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        reg_alpha=0.0,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        eval_metric=brier_cost,
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(x_train, y_train, verbose=False)
    prediction = model.predict_proba(x_valid)[:, 1].astype(np.float64)

    del model, x_train, x_valid, y_train
    gc.collect()
    return prediction


def select_iterations(
    family: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[int, dict[str, float]]:
    if family == "catboost":
        return select_catboost_iterations(
            train, valid, target, features, args.max_estimators,
            args.early_stopping_rounds, seed, args.catboost_task_type,
            args.catboost_devices, args.verbose,
        )
    if family == "lightgbm":
        return select_lightgbm_iterations(
            train, valid, target, features, args.max_estimators,
            args.early_stopping_rounds, seed, args.lightgbm_device,
        )
    if family == "xgboost":
        return select_xgboost_iterations(
            train, valid, target, features, args.max_estimators,
            args.early_stopping_rounds, seed, args.xgboost_device,
        )
    raise ValueError(f"Unknown family: {family}")


def refit_predict(
    family: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    n_estimators: int,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    if family == "catboost":
        return refit_predict_catboost(
            train, valid, target, features, n_estimators, seed,
            args.catboost_task_type, args.catboost_devices, args.verbose,
        )
    if family == "lightgbm":
        return refit_predict_lightgbm(
            train, valid, target, features, n_estimators, seed, args.lightgbm_device
        )
    if family == "xgboost":
        return refit_predict_xgboost(
            train, valid, target, features, n_estimators, seed, args.xgboost_device
        )
    raise ValueError(f"Unknown family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CatBoost, LightGBM, and XGBoost using deployment-safe temporal "
            "inner validation to select boosting iterations by Brier score."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--models", default="all")
    parser.add_argument("--feature-variants", default="success_state,success_plus_hand_matchup")
    parser.add_argument("--max-estimators", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--catboost-devices", default="0")
    parser.add_argument("--lightgbm-device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--xgboost-device", default="cuda")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if args.max_estimators <= 0 or args.early_stopping_rounds <= 0:
        raise ValueError("max-estimators and early-stopping-rounds must be positive")

    folds = parse_ints(args.folds)
    models = list(MODEL_FAMILIES) if args.models == "all" else parse_strings(args.models)
    variants = parse_strings(args.feature_variants)
    unknown_models = [x for x in models if x not in MODEL_FAMILIES]
    unknown_variants = [x for x in variants if x not in FEATURE_VARIANTS]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    if unknown_variants:
        raise ValueError(f"Unknown feature variants: {unknown_variants}")
    _require_packages(models)

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame = load_frame(config).copy()
    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(
        raw_canonical
        + CANONICAL_SOURCE_COLUMNS
        + [
            target, season, row_id,
            "asof_pitcher_success_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_success_rate", "asof_batter_middle_rate", "asof_pitcher_n",
            "pitcher_hand", "batter_hand",
        ]
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    asof_core.add_asof_state_features(frame)
    context_core.add_context_interactions(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    output_dir = Path(config["paths"]["output_dir"]) / "model_family_screen"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = {variant: feature_set(variant) for variant in variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    versions = _package_versions(models)
    print(
        f"[Model Family Screen] folds={folds}, models={models}, variants={variants}, "
        f"max_estimators={args.max_estimators}, early_stop={args.early_stopping_rounds}"
    )
    print(f"[Model Family Screen] versions={versions}")
    print("[Model Family Screen] NO row sampling.")
    print("[Model Family Screen] For eval Y: inner train <Y-1, inner valid=Y-1 selects Brier-best iteration;")
    print("                      then refit on all <Y using that iteration count and evaluate Y.")
    print("[Model Family Screen] categorical vocabularies for LightGBM/XGBoost are learned from fit rows only.")

    rows: list[dict] = []
    for eval_year in folds:
        inner_year = eval_year - 1
        inner_train = frame.loc[frame[season] < inner_year]
        inner_valid = frame.loc[frame[season] == inner_year]
        outer_train = frame.loc[frame[season] < eval_year]
        outer_valid = frame.loc[frame[season] == eval_year]
        if inner_train.empty or inner_valid.empty or outer_train.empty or outer_valid.empty:
            raise ValueError(f"Fold {eval_year}: missing temporal split")
        y_outer = pd.to_numeric(outer_valid[target], errors="raise").to_numpy(np.float64)

        print(
            f"\n[Fold {eval_year}] inner={inner_train[season].min()}..{inner_train[season].max()} "
            f"({len(inner_train):,}) -> {inner_year} ({len(inner_valid):,}); "
            f"refit={len(outer_train):,}; eval={len(outer_valid):,}"
        )

        for variant in variants:
            features = feature_sets[variant]
            for family in models:
                print(f"  [{family:8s}] {variant:<27s} selecting...", flush=True)
                best_n, inner_metric = select_iterations(
                    family, inner_train, inner_valid, target, features, args, seed
                )
                print(
                    f"             best_n={best_n:4d} inner_brier={inner_metric['brier']:.8f} "
                    f"inner_raw_score={inner_metric['raw_score']:+.2f}; refitting...",
                    flush=True,
                )
                prediction = refit_predict(
                    family, outer_train, outer_valid, target, features, best_n, args, seed
                )
                metric = probability_metrics(y_outer, prediction)
                rows.append(
                    {
                        "evaluation_year": int(eval_year),
                        "inner_validation_year": int(inner_year),
                        "model_family": family,
                        "feature_variant": variant,
                        "feature_count": int(len(features)),
                        "categorical_count": int(len(categorical_columns(features))),
                        "selected_estimators": int(best_n),
                        "inner_brier": inner_metric["brier"],
                        "inner_raw_score": inner_metric["raw_score"],
                        "train_rows": int(len(outer_train)),
                        "val_rows": int(len(outer_valid)),
                        **metric,
                    }
                )
                print(
                    f"             eval brier={metric['brier']:.8f} raw_score={metric['raw_score']:+.2f} "
                    f"auc={metric['auc']:.5f} p_std={metric['prediction_std']:.5f}"
                )
                del prediction
                gc.collect()

        del y_outer
        gc.collect()

    results = pd.DataFrame(rows)
    baseline = (
        results.loc[
            (results["model_family"] == "catboost")
            & (results["feature_variant"] == "success_state"),
            ["evaluation_year", "brier"],
        ]
        .rename(columns={"brier": "catboost_success_brier"})
    )
    results = results.merge(baseline, on="evaluation_year", how="left")
    results["delta_brier_vs_catboost_success"] = results["brier"] - results["catboost_success_brier"]
    results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        results.groupby(["model_family", "feature_variant"], as_index=False)
        .agg(
            folds=("evaluation_year", "count"),
            mean_selected_estimators=("selected_estimators", "mean"),
            min_selected_estimators=("selected_estimators", "min"),
            max_selected_estimators=("selected_estimators", "max"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_delta_vs_catboost_success=("delta_brier_vs_catboost_success", "mean"),
            worst_delta_vs_catboost_success=("delta_brier_vs_catboost_success", "max"),
            best_delta_vs_catboost_success=("delta_brier_vs_catboost_success", "min"),
            mean_raw_score=("raw_score", "mean"),
            worst_raw_score=("raw_score", "min"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["mean_delta_vs_catboost_success", "worst_delta_vs_catboost_success"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "folds": folds,
            "models": models,
            "feature_variants": variants,
            "feature_sets": feature_sets,
            "max_estimators": int(args.max_estimators),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "versions": versions,
            "devices": {
                "catboost_task_type": args.catboost_task_type,
                "catboost_devices": args.catboost_devices,
                "lightgbm_device": args.lightgbm_device,
                "xgboost_device": args.xgboost_device,
            },
            "canonical_invariants": invariant_check,
            "sampling": "none",
            "selection_protocol": "eval Y: train <Y-1, select on Y-1 by Brier, refit <Y at selected estimator count, evaluate Y",
        },
        output_dir / "run_config.json",
    )

    print("\n[Model Family Summary: lower delta is better]")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_selected_estimators": "{:.1f}".format,
                "mean_brier": "{:.8f}".format,
                "worst_brier": "{:.8f}".format,
                "mean_delta_vs_catboost_success": "{:+.8f}".format,
                "worst_delta_vs_catboost_success": "{:+.8f}".format,
                "best_delta_vs_catboost_success": "{:+.8f}".format,
                "mean_raw_score": "{:+.2f}".format,
                "worst_raw_score": "{:+.2f}".format,
                "mean_auc": "{:.5f}".format,
            },
        )
    )
    print("\n[Per-fold ranking]")
    display = results[
        [
            "evaluation_year", "model_family", "feature_variant", "selected_estimators",
            "brier", "raw_score", "auc", "delta_brier_vs_catboost_success",
        ]
    ].sort_values(["evaluation_year", "brier"])
    print(
        display.to_string(
            index=False,
            formatters={
                "brier": "{:.8f}".format,
                "raw_score": "{:+.2f}".format,
                "auc": "{:.5f}".format,
                "delta_brier_vs_catboost_success": "{:+.8f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
