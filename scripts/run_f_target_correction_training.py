from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_game_type_f_offset as offset_diag
import run_catboost_ablation as core
from src.canonical_features import (
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.utils import load_config, save_json, seed_everything


FEATURES = list(CANONICAL_FEATURES)


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one correction scale is required")
    if any(value < 0 for value in values):
        raise ValueError("Correction scales must be >= 0")
    return values


def normalize_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def catboost_params(
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
    loss_function: str,
) -> dict:
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
        "loss_function": loss_function,
        "has_time": True,
        "one_hot_max_size": 10,
        "allow_writing_files": False,
        "task_type": task_type,
        "verbose": verbose,
    }
    if task_type == "GPU":
        params["devices"] = devices
    return params


def estimate_subgroup_delta(
    frame: pd.DataFrame,
    old_years: list[int],
    calibration_year: int,
    season_col: str,
    target_col: str,
    min_group_rows: int,
) -> tuple[float, pd.DataFrame]:
    work = frame.copy()
    offset_diag.add_experience_bucket(work)
    pairs = offset_diag.build_subgroup_pairs(
        work,
        old_years=old_years,
        new_years=[calibration_year],
        season_col=season_col,
        target_col=target_col,
        game_type_col="game_type",
        min_group_rows=min_group_rows,
    )
    if pairs.empty:
        raise ValueError("No matched subgroups were available to estimate the F correction")
    residual = pairs["raw_residual"].to_numpy(np.float64)
    weight = pairs["weight"].to_numpy(np.float64)
    # Minimize weighted RMSE of (new F-R + delta) - old F-R.
    delta = -float(np.average(residual, weights=weight))
    return delta, pairs


def overall_fr_delta(
    frame: pd.DataFrame,
    old_years: list[int],
    calibration_year: int,
    season_col: str,
    target_col: str,
) -> float:
    def fr_effect(part: pd.DataFrame) -> float:
        rate = part.groupby("game_type", observed=True)[target_col].mean()
        if not {"F", "R"}.issubset(rate.index):
            raise ValueError("Both F and R are required")
        return float(rate.loc["F"] - rate.loc["R"])

    old_effects = []
    for year in old_years:
        old_effects.append(fr_effect(frame.loc[frame[season_col].eq(year)]))
    old_reference = float(np.mean(old_effects))
    new_effect = fr_effect(frame.loc[frame[season_col].eq(calibration_year)])
    return old_reference - new_effect


def build_soft_targets(
    train: pd.DataFrame,
    raw_y: np.ndarray,
    season_col: str,
    calibration_year: int,
    delta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.asarray(raw_y, dtype=np.float64).copy()
    mask = (
        train[season_col].eq(calibration_year).to_numpy()
        & normalize_category(train["game_type"]).eq("F").to_numpy()
    )
    if not mask.any():
        raise ValueError("No calibration-year F rows found")

    q = float(y[mask].mean())
    q_target = float(np.clip(q + delta, 1e-6, 1.0 - 1e-6))
    if q >= 1.0 - 1e-12:
        raise ValueError("Calibration F success rate is already 1")

    # Keep observed positives at 1.0 and lift observed negatives to c. This is the
    # unique one-parameter affine soft-label transform that preserves positives and
    # makes the calibration-year F mean equal q_target exactly.
    c = float((q_target - q) / (1.0 - q))
    c = float(np.clip(c, 0.0, 1.0 - 1e-6))
    y[mask] = c + (1.0 - c) * y[mask]

    return y, {
        "raw_f_rate": q,
        "target_f_rate": q_target,
        "soft_negative_target": c,
        "achieved_soft_f_rate": float(y[mask].mean()),
    }


def invert_soft_prediction(
    prediction: np.ndarray,
    frame: pd.DataFrame,
    soft_negative_target: float,
) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64).copy()
    f_mask = normalize_category(frame["game_type"]).eq("F").to_numpy()
    c = float(soft_negative_target)
    p[f_mask] = (p[f_mask] - c) / max(1e-12, 1.0 - c)
    return np.clip(p, 0.0, 1.0)


def build_reweighting(
    train: pd.DataFrame,
    raw_y: np.ndarray,
    season_col: str,
    calibration_year: int,
    delta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.asarray(raw_y, dtype=np.float64)
    mask = (
        train[season_col].eq(calibration_year).to_numpy()
        & normalize_category(train["game_type"]).eq("F").to_numpy()
    )
    if not mask.any():
        raise ValueError("No calibration-year F rows found")

    q = float(y[mask].mean())
    q_target = float(np.clip(q + delta, 1e-6, 1.0 - 1e-6))
    if q <= 0.0 or q >= 1.0:
        raise ValueError("Calibration F success rate must lie strictly between 0 and 1")

    # Preserve total expected weight inside the calibration-year F group while
    # changing its weighted positive prevalence from q to q_target.
    w_pos = q_target / q
    w_neg = (1.0 - q_target) / (1.0 - q)
    odds_ratio = w_pos / w_neg

    weight = np.ones(len(train), dtype=np.float64)
    positive = mask & (y > 0.5)
    negative = mask & ~positive
    weight[positive] = w_pos
    weight[negative] = w_neg

    weighted_rate = float(np.average(y[mask], weights=weight[mask]))
    group_mean_weight = float(weight[mask].mean())
    # Global normalization is loss-equivalent and keeps numerical scale familiar.
    weight /= float(weight.mean())

    return weight, {
        "raw_f_rate": q,
        "target_f_rate": q_target,
        "positive_weight_before_global_norm": float(w_pos),
        "negative_weight_before_global_norm": float(w_neg),
        "positive_to_negative_odds_ratio": float(odds_ratio),
        "weighted_f_rate": weighted_rate,
        "f_group_mean_weight_before_global_norm": group_mean_weight,
    }


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def invert_weighted_prediction(
    prediction: np.ndarray,
    frame: pd.DataFrame,
    positive_to_negative_odds_ratio: float,
) -> np.ndarray:
    p = np.asarray(prediction, dtype=np.float64).copy()
    f_mask = normalize_category(frame["game_type"]).eq("F").to_numpy()
    p[f_mask] = sigmoid(
        safe_logit(p[f_mask]) - math.log(float(positive_to_negative_odds_ratio))
    )
    return np.clip(p, 0.0, 1.0)


def train_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    label: np.ndarray,
    params: dict,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = core.prepare_x(train, FEATURES)
    x_valid, _ = core.prepare_x(valid, FEATURES)
    train_pool = Pool(
        x_train,
        label=np.asarray(label, dtype=np.float64),
        weight=sample_weight,
        cat_features=categorical,
        feature_names=FEATURES,
    )
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=FEATURES)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=params.get("verbose", 0))
    prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
    del model, train_pool, valid_pool, x_train, x_valid
    gc.collect()
    return prediction


def detailed_metrics(y: np.ndarray, p: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    result = dict(core.metrics(y, p))
    gt = normalize_category(frame["game_type"])
    for label in ("F", "R"):
        mask = gt.eq(label).to_numpy()
        yy = np.asarray(y, dtype=np.float64)[mask]
        pp = np.asarray(p, dtype=np.float64)[mask]
        prefix = label.lower()
        result[f"{prefix}_rows"] = int(mask.sum())
        result[f"{prefix}_rate"] = float(yy.mean())
        result[f"{prefix}_pred_mean"] = float(pp.mean())
        result[f"{prefix}_calibration_gap"] = float(pp.mean() - yy.mean())
        result[f"{prefix}_brier"] = float(np.mean(np.square(pp - yy)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain CatBoost after normalizing calibration-year game_type=F targets "
            "toward the old measurement scale, then invert predictions back to the "
            "observed scale for next-year validation."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--scales", default="0.50,0.75,1.00,1.25")
    parser.add_argument("--min-group-rows", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if args.validation_year <= args.calibration_year:
        raise ValueError("validation-year must be after calibration-year")
    scales = parse_floats(args.scales)

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame = load_frame(config).copy()
    raw_canonical = [feature for feature in CANONICAL_FEATURES if feature != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype(np.float64)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    old_years = sorted(int(year) for year in frame.loc[frame[season] < args.calibration_year, season].unique())
    if not old_years:
        raise ValueError("No old-era seasons found")

    train = frame.loc[frame[season] < args.validation_year].copy()
    valid = frame.loc[frame[season].eq(args.validation_year)].copy()
    if train.empty or valid.empty:
        raise ValueError("Training or validation partition is empty")
    if args.calibration_year not in set(train[season].unique()):
        raise ValueError("Calibration year is not in the training partition")

    y_train = train[target].to_numpy(np.float64)
    y_valid = valid[target].to_numpy(np.float64)

    learned_delta, subgroup_pairs = estimate_subgroup_delta(
        frame,
        old_years=old_years,
        calibration_year=args.calibration_year,
        season_col=season,
        target_col=target,
        min_group_rows=args.min_group_rows,
    )
    direct_delta = overall_fr_delta(
        frame,
        old_years=old_years,
        calibration_year=args.calibration_year,
        season_col=season,
        target_col=target,
    )

    output_dir = Path(config["paths"]["output_dir"]) / "f_target_correction_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    subgroup_pairs.to_csv(output_dir / "calibration_subgroup_pairs.csv", index=False)

    print(
        f"[F Target-Correction Training] old={old_years}, calibration={args.calibration_year}, "
        f"validation={args.validation_year}, scales={scales}, iterations={args.iterations}, "
        f"task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print(f"  subgroup-derived old-scale F correction = {learned_delta:+.6f}")
    print(f"  direct overall F-R old-scale correction = {direct_delta:+.6f}")
    print(f"  {args.validation_year} labels are used ONLY for scoring")

    logloss_params = catboost_params(
        config, args.iterations, args.task_type, args.devices, args.verbose, "Logloss"
    )
    crossentropy_params = catboost_params(
        config, args.iterations, args.task_type, args.devices, args.verbose, "CrossEntropy"
    )

    rows: list[dict] = []
    parameter_rows: list[dict] = []

    def record(name: str, family: str, scale: float | None, prediction: np.ndarray, detail: str) -> None:
        metric = detailed_metrics(y_valid, prediction, valid)
        rows.append(
            {
                "name": name,
                "family": family,
                "correction_scale": scale,
                "detail": detail,
                **metric,
            }
        )

    print("\n[1] Raw Logloss baseline")
    p_raw = train_predict(train, valid, y_train, logloss_params)
    record("raw_logloss", "baseline", None, p_raw, "raw binary targets; Logloss")

    print("[2] Raw CrossEntropy control")
    p_raw_ce = train_predict(train, valid, y_train, crossentropy_params)
    record(
        "raw_crossentropy",
        "baseline",
        None,
        p_raw_ce,
        "raw binary targets; CrossEntropy control for soft-label comparison",
    )

    for index, scale in enumerate(scales, start=1):
        delta = float(scale * learned_delta)
        print(f"[{index + 2}/{len(scales) + 2}] correction scale={scale:.3f}, delta={delta:+.6f}")

        soft_y, soft_info = build_soft_targets(
            train,
            y_train,
            season_col=season,
            calibration_year=args.calibration_year,
            delta=delta,
        )
        p_soft_latent = train_predict(train, valid, soft_y, crossentropy_params)
        p_soft_inverse = invert_soft_prediction(
            p_soft_latent,
            valid,
            soft_negative_target=soft_info["soft_negative_target"],
        )
        record(
            f"soft_s{scale:.2f}_latent",
            "soft_target",
            scale,
            p_soft_latent,
            "CrossEntropy soft targets; prediction left on normalized/old scale",
        )
        record(
            f"soft_s{scale:.2f}_inverse",
            "soft_target",
            scale,
            p_soft_inverse,
            "CrossEntropy soft targets; F prediction inverse-affine transformed to observed scale",
        )
        parameter_rows.append(
            {
                "family": "soft_target",
                "correction_scale": scale,
                "delta": delta,
                **soft_info,
            }
        )
        del soft_y, p_soft_latent, p_soft_inverse
        gc.collect()

        weight, weight_info = build_reweighting(
            train,
            y_train,
            season_col=season,
            calibration_year=args.calibration_year,
            delta=delta,
        )
        p_weight_latent = train_predict(
            train,
            valid,
            y_train,
            logloss_params,
            sample_weight=weight,
        )
        p_weight_inverse = invert_weighted_prediction(
            p_weight_latent,
            valid,
            positive_to_negative_odds_ratio=weight_info["positive_to_negative_odds_ratio"],
        )
        record(
            f"weight_s{scale:.2f}_latent",
            "class_reweight",
            scale,
            p_weight_latent,
            "binary Logloss; rebalance calibration-year F positives/negatives; latent scale",
        )
        record(
            f"weight_s{scale:.2f}_inverse",
            "class_reweight",
            scale,
            p_weight_inverse,
            "binary Logloss; rebalanced targets; inverse-odds transform on validation F",
        )
        parameter_rows.append(
            {
                "family": "class_reweight",
                "correction_scale": scale,
                "delta": delta,
                **weight_info,
            }
        )
        del weight, p_weight_latent, p_weight_inverse
        gc.collect()

    results = pd.DataFrame(rows)
    baseline_brier = float(results.loc[results["name"].eq("raw_logloss"), "brier"].iloc[0])
    baseline_score = float(
        results.loc[results["name"].eq("raw_logloss"), "competition_score"].iloc[0]
    )
    results["delta_brier_vs_raw"] = results["brier"] - baseline_brier
    results["delta_score_vs_raw"] = results["competition_score"] - baseline_score
    results = results.sort_values(["brier", "name"]).reset_index(drop=True)

    params_df = pd.DataFrame(parameter_rows)
    results.to_csv(output_dir / "results.csv", index=False)
    params_df.to_csv(output_dir / "correction_parameters.csv", index=False)
    save_json(
        {
            "old_years": old_years,
            "calibration_year": int(args.calibration_year),
            "validation_year": int(args.validation_year),
            "correction_scales": scales,
            "subgroup_derived_delta": float(learned_delta),
            "direct_overall_fr_delta": float(direct_delta),
            "min_group_rows": int(args.min_group_rows),
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices,
            "features": FEATURES,
            "canonical_invariants": invariant_check,
            "logloss_params": logloss_params,
            "crossentropy_params": crossentropy_params,
            "selection_warning": (
                "2024 is a validation fold. Do not automatically choose the single best scale "
                "for 2025; first judge whether a correction family improves robustly around scale=1."
            ),
        },
        output_dir / "run_config.json",
    )

    display_cols = [
        "name",
        "brier",
        "competition_score",
        "auc",
        "prediction_std",
        "f_brier",
        "f_calibration_gap",
        "r_brier",
        "delta_brier_vs_raw",
    ]
    print("\n[Validation results: lower Brier is better]")
    print(
        results[display_cols].to_string(
            index=False,
            formatters={
                "brier": "{:.8f}".format,
                "competition_score": "{:.2f}".format,
                "auc": "{:.5f}".format,
                "prediction_std": "{:.5f}".format,
                "f_brier": "{:.8f}".format,
                "f_calibration_gap": "{:+.6f}".format,
                "r_brier": "{:.8f}".format,
                "delta_brier_vs_raw": "{:+.8f}".format,
            },
        )
    )

    print("\n[Correction parameters]")
    keep = [
        column
        for column in [
            "family",
            "correction_scale",
            "delta",
            "raw_f_rate",
            "target_f_rate",
            "soft_negative_target",
            "positive_weight_before_global_norm",
            "negative_weight_before_global_norm",
            "positive_to_negative_odds_ratio",
        ]
        if column in params_df.columns
    ]
    print(params_df[keep].to_string(index=False))

    best = results.iloc[0]
    print(
        f"\nBest={best['name']} brier={best['brier']:.8f} "
        f"score={best['competition_score']:.2f} "
        f"delta_vs_raw={best['delta_brier_vs_raw']:+.8f}"
    )
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
