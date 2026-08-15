from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_asof_state_engineering as asof_core
import run_catboost_ablation as core
from src.canonical_features import (
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


BASE_VARIANTS = {
    "reference_canonical": [],
    "add_success_state": list(asof_core.SUCCESS_STATE),
}

CALIBRATORS = [
    "raw",
    "mean_shift",
    "shrink_to_prior",
    "logit_intercept",
    "temperature",
    "affine_logit",
]

EPS = 1e-6


def parse_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("at least one fold is required")
    return result


def parse_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def feature_set(variant: str) -> list[str]:
    if variant not in BASE_VARIANTS:
        raise ValueError(f"Unknown base variant: {variant}")
    return list(CANONICAL_FEATURES) + list(BASE_VARIANTS[variant])


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=np.float64)

    if method == "raw":
        return {}

    if method == "mean_shift":
        return {"delta": float(y.mean() - p.mean())}

    if method == "shrink_to_prior":
        center = float(y.mean())
        x = p - center
        target = y - center
        denom = float(np.dot(x, x))
        scale = 0.0 if denom <= 0.0 else float(np.dot(x, target) / denom)
        return {"center": center, "scale": float(np.clip(scale, 0.0, 2.5))}

    z = logit(p)

    if method == "logit_intercept":
        result = minimize_scalar(
            lambda b: brier(y, sigmoid(z + float(b))),
            bounds=(-3.0, 3.0),
            method="bounded",
            options={"xatol": 1e-7},
        )
        if not result.success:
            raise RuntimeError(f"logit_intercept optimization failed: {result.message}")
        return {"intercept": float(result.x)}

    if method == "temperature":
        result = minimize_scalar(
            lambda log_t: brier(y, sigmoid(z / np.exp(float(log_t)))),
            bounds=(-2.0, 2.0),
            method="bounded",
            options={"xatol": 1e-7},
        )
        if not result.success:
            raise RuntimeError(f"temperature optimization failed: {result.message}")
        return {"temperature": float(np.exp(result.x))}

    if method == "affine_logit":
        def objective(theta: np.ndarray) -> float:
            log_scale, intercept = float(theta[0]), float(theta[1])
            scale = np.exp(log_scale)
            return brier(y, sigmoid(scale * z + intercept))

        result = minimize(
            objective,
            x0=np.array([0.0, 0.0], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(-2.0, 2.0), (-3.0, 3.0)],
            options={"ftol": 1e-12, "maxiter": 200},
        )
        if not result.success:
            raise RuntimeError(f"affine_logit optimization failed: {result.message}")
        return {
            "scale": float(np.exp(result.x[0])),
            "intercept": float(result.x[1]),
        }

    raise ValueError(f"Unknown calibrator: {method}")


def apply_calibrator(method: str, p: np.ndarray, params: dict[str, float]) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    if method == "raw":
        return p
    if method == "mean_shift":
        return np.clip(p + params["delta"], 0.0, 1.0)
    if method == "shrink_to_prior":
        center = params["center"]
        return np.clip(center + params["scale"] * (p - center), 0.0, 1.0)

    z = logit(p)
    if method == "logit_intercept":
        return sigmoid(z + params["intercept"])
    if method == "temperature":
        return sigmoid(z / params["temperature"])
    if method == "affine_logit":
        return sigmoid(params["scale"] * z + params["intercept"])
    raise ValueError(f"Unknown calibrator: {method}")


def catboost_params(config: dict, iterations: int, task_type: str, devices: str, verbose: int) -> dict:
    params = {
        "iterations": int(iterations),
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
        "random_seed": int(config["seed"]),
        "loss_function": "Logloss",
        "has_time": True,
        "one_hot_max_size": 10,
        "allow_writing_files": False,
        "task_type": task_type,
        "verbose": verbose,
    }
    if task_type == "GPU":
        params["devices"] = devices
    return params


def fit_predict(
    train: pd.DataFrame,
    predict_frame: pd.DataFrame,
    target: str,
    features: list[str],
    params: dict,
    verbose: int,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = core.prepare_x(train, features)
    x_pred, _ = core.prepare_x(predict_frame, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    pred_pool = Pool(x_pred, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)
    prediction = model.predict_proba(pred_pool)[:, 1].astype(np.float64)
    del model, train_pool, pred_pool, x_train, x_pred, y_train
    gc.collect()
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict temporal calibration screen. For evaluation year Y, fit a calibrator on "
            "Y-1 predictions from a model trained through Y-2, then apply it to Y predictions "
            "from a fresh model trained through Y-1."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--base-variants", default="reference_canonical,add_success_state")
    parser.add_argument("--calibrators", default="all")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    folds = parse_ints(args.folds)
    base_variants = parse_strings(args.base_variants)
    calibrators = list(CALIBRATORS) if args.calibrators == "all" else parse_strings(args.calibrators)

    unknown_bases = [v for v in base_variants if v not in BASE_VARIANTS]
    unknown_cal = [v for v in calibrators if v not in CALIBRATORS]
    if unknown_bases:
        raise ValueError(f"Unknown base variants: {unknown_bases}")
    if unknown_cal:
        raise ValueError(f"Unknown calibrators: {unknown_cal}")
    if "raw" not in calibrators:
        calibrators = ["raw", *calibrators]

    frame = load_frame(config).copy()
    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(
        raw_canonical
        + CANONICAL_SOURCE_COLUMNS
        + [
            target,
            season,
            row_id,
            "asof_pitcher_success_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
            "asof_pitcher_n",
        ]
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    asof_core.add_asof_state_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    output_dir = Path(config["paths"]["output_dir"]) / "temporal_calibration_screen"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = {v: feature_set(v) for v in base_variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)
    print(
        f"[Temporal Calibration Screen] folds={folds}, bases={base_variants}, "
        f"calibrators={calibrators}, iterations={args.iterations}, "
        f"task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print("[Temporal Calibration Screen] NO row sampling.")
    print("[Temporal Calibration Screen] protocol for eval Y:")
    print("  calibrator model: seasons <= Y-2 -> predict Y-1 -> fit calibrator on Y-1 labels")
    print("  evaluation model: seasons <= Y-1 -> predict Y -> apply frozen calibrator")
    print("[Temporal Calibration Screen] raw_score is UNCLIPPED; clipped_score preserves prior display semantics.")

    fit_rows: list[dict] = []
    eval_rows: list[dict] = []

    for eval_year in folds:
        cal_year = eval_year - 1
        cal_train = frame.loc[frame[season] < cal_year]
        cal_frame = frame.loc[frame[season] == cal_year]
        eval_train = frame.loc[frame[season] < eval_year]
        eval_frame = frame.loc[frame[season] == eval_year]
        if cal_train.empty or cal_frame.empty or eval_train.empty or eval_frame.empty:
            raise ValueError(
                f"Fold {eval_year}: missing split; need train<{cal_year}, cal={cal_year}, eval={eval_year}"
            )

        y_cal = pd.to_numeric(cal_frame[target], errors="raise").to_numpy(np.float64)
        y_eval = pd.to_numeric(eval_frame[target], errors="raise").to_numpy(np.float64)
        print(
            f"\n[Fold {eval_year}] cal_year={cal_year}, cal_train={len(cal_train):,}, "
            f"cal={len(cal_frame):,}, eval_train={len(eval_train):,}, eval={len(eval_frame):,}"
        )

        for base_variant in base_variants:
            features = feature_sets[base_variant]
            print(f"  [Base] {base_variant} features={len(features)}", flush=True)
            cal_prediction = fit_predict(
                cal_train, cal_frame, target, features, params, args.verbose
            )
            eval_prediction = fit_predict(
                eval_train, eval_frame, target, features, params, args.verbose
            )
            raw_cal_metric = probability_metrics(y_cal, cal_prediction)
            raw_eval_metric = probability_metrics(y_eval, eval_prediction)
            print(
                f"    raw eval: brier={raw_eval_metric['brier']:.8f} "
                f"raw_score={raw_eval_metric['raw_score']:+.2f} "
                f"clipped={raw_eval_metric['clipped_score']:.2f} "
                f"auc={raw_eval_metric['auc']:.5f}"
            )

            for method in calibrators:
                fitted = fit_calibrator(method, cal_prediction, y_cal)
                cal_calibrated = apply_calibrator(method, cal_prediction, fitted)
                eval_calibrated = apply_calibrator(method, eval_prediction, fitted)
                cal_metric = probability_metrics(y_cal, cal_calibrated)
                eval_metric = probability_metrics(y_eval, eval_calibrated)

                fit_rows.append(
                    {
                        "evaluation_year": int(eval_year),
                        "calibration_year": int(cal_year),
                        "base_variant": base_variant,
                        "calibrator": method,
                        "cal_train_rows": int(len(cal_train)),
                        "cal_rows": int(len(cal_frame)),
                        "raw_cal_brier": raw_cal_metric["brier"],
                        "calibrated_cal_brier": cal_metric["brier"],
                        "delta_cal_brier": cal_metric["brier"] - raw_cal_metric["brier"],
                        "raw_cal_prediction_mean": raw_cal_metric["prediction_mean"],
                        "calibrated_cal_prediction_mean": cal_metric["prediction_mean"],
                        "cal_target_mean": cal_metric["target_mean"],
                        "parameters": json.dumps(fitted, sort_keys=True),
                    }
                )
                eval_rows.append(
                    {
                        "evaluation_year": int(eval_year),
                        "calibration_year": int(cal_year),
                        "base_variant": base_variant,
                        "calibrator": method,
                        "eval_train_rows": int(len(eval_train)),
                        "eval_rows": int(len(eval_frame)),
                        "feature_count": int(len(features)),
                        **eval_metric,
                        "delta_brier_vs_raw": eval_metric["brier"] - raw_eval_metric["brier"],
                        "delta_raw_score_vs_raw": eval_metric["raw_score"] - raw_eval_metric["raw_score"],
                    }
                )
                print(
                    f"    {method:<18s} brier={eval_metric['brier']:.8f} "
                    f"delta={eval_metric['brier'] - raw_eval_metric['brier']:+.8f} "
                    f"raw_score={eval_metric['raw_score']:+.2f} "
                    f"p_mean={eval_metric['prediction_mean']:.5f}"
                )

            del cal_prediction, eval_prediction
            gc.collect()

        del y_cal, y_eval
        gc.collect()

    fit_results = pd.DataFrame(fit_rows)
    results = pd.DataFrame(eval_rows)
    fit_results.to_csv(output_dir / "calibration_fit.csv", index=False)
    results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        results.groupby(["base_variant", "calibrator"], as_index=False)
        .agg(
            folds=("evaluation_year", "count"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_delta_brier=("delta_brier_vs_raw", "mean"),
            worst_delta_brier=("delta_brier_vs_raw", "max"),
            best_delta_brier=("delta_brier_vs_raw", "min"),
            mean_raw_score=("raw_score", "mean"),
            worst_raw_score=("raw_score", "min"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["base_variant", "mean_delta_brier", "worst_delta_brier"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "folds": folds,
            "base_variants": base_variants,
            "calibrators": calibrators,
            "iterations": int(args.iterations),
            "catboost_params": params,
            "canonical_invariants": invariant_check,
            "sampling": "none",
            "protocol": (
                "for eval Y, fit calibrator on Y-1 predictions from model trained on seasons < Y-1; "
                "apply frozen calibrator to Y predictions from fresh model trained on seasons < Y"
            ),
            "score_reporting": "raw_score is unclipped; clipped_score/competition_score clamp negative values to zero",
        },
        output_dir / "run_config.json",
    )

    print("\n[Temporal Calibration Summary: lower delta is better]")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_brier": "{:.8f}".format,
                "worst_brier": "{:.8f}".format,
                "mean_delta_brier": "{:+.8f}".format,
                "worst_delta_brier": "{:+.8f}".format,
                "best_delta_brier": "{:+.8f}".format,
                "mean_raw_score": "{:+.2f}".format,
                "worst_raw_score": "{:+.2f}".format,
                "mean_auc": "{:.5f}".format,
            },
        )
    )
    print("\n[Per-fold results]")
    print(
        results[
            [
                "evaluation_year", "base_variant", "calibrator", "brier",
                "delta_brier_vs_raw", "raw_score", "clipped_score", "auc",
                "prediction_mean", "target_mean",
            ]
        ].sort_values(["evaluation_year", "base_variant", "brier"]).to_string(
            index=False,
            formatters={
                "brier": "{:.8f}".format,
                "delta_brier_vs_raw": "{:+.8f}".format,
                "raw_score": "{:+.2f}".format,
                "clipped_score": "{:.2f}".format,
                "auc": "{:.5f}".format,
                "prediction_mean": "{:.5f}".format,
                "target_mean": "{:.5f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
