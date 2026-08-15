from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_catboost_ablation as core
import run_f_target_correction_training as corr
from src.canonical_features import (
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.utils import load_config, save_json, seed_everything


FULL_FEATURES = list(CANONICAL_FEATURES)
F_FEATURES = [feature for feature in CANONICAL_FEATURES if feature != "game_type"]


def parse_floats(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one float value is required")
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


def train_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    label: np.ndarray,
    features: list[str],
    params: dict,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = core.prepare_x(train, features)
    x_valid, _ = core.prepare_x(valid, features)
    train_pool = Pool(
        x_train,
        label=np.asarray(label, dtype=np.float64),
        weight=sample_weight,
        cat_features=categorical,
        feature_names=features,
    )
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=params.get("verbose", 0))
    prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)

    del model, train_pool, valid_pool, x_train, x_valid
    gc.collect()
    return prediction


def replace_f(
    raw_prediction: np.ndarray,
    validation: pd.DataFrame,
    f_prediction: np.ndarray,
) -> np.ndarray:
    out = np.asarray(raw_prediction, dtype=np.float64).copy()
    f_mask = normalize_category(validation["game_type"]).eq("F").to_numpy()
    if int(f_mask.sum()) != len(f_prediction):
        raise ValueError(
            f"F prediction length mismatch: mask={int(f_mask.sum())}, prediction={len(f_prediction)}"
        )
    out[f_mask] = np.asarray(f_prediction, dtype=np.float64)
    return np.clip(out, 0.0, 1.0)


def blend_f(
    raw_prediction: np.ndarray,
    validation: pd.DataFrame,
    candidate_f_prediction: np.ndarray,
    weight: float,
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("blend weight must be in [0, 1]")
    f_mask = normalize_category(validation["game_type"]).eq("F").to_numpy()
    raw_f = np.asarray(raw_prediction, dtype=np.float64)[f_mask]
    candidate_f = np.asarray(candidate_f_prediction, dtype=np.float64)
    mixed_f = (1.0 - weight) * raw_f + weight * candidate_f
    return replace_f(raw_prediction, validation, mixed_f)


def detailed_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    validation: pd.DataFrame,
) -> dict[str, float]:
    result = dict(core.metrics(y, prediction))
    game_type = normalize_category(validation["game_type"])
    for label in ("F", "R"):
        mask = game_type.eq(label).to_numpy()
        yy = np.asarray(y, dtype=np.float64)[mask]
        pp = np.asarray(prediction, dtype=np.float64)[mask]
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
            "Test whether F-only target correction is useful when R predictions are protected by "
            "the ordinary full-history model. Also compare an F-only specialist against a corrected "
            "full model, using leakage-safe 2023->2024 validation."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--scales", default="0.50,0.75,1.00,1.25")
    parser.add_argument("--blend-weights", default="0.25,0.50,0.75,1.00")
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
    blend_weights = parse_floats(args.blend_weights)
    if any(value < 0 for value in scales):
        raise ValueError("correction scales must be >= 0")
    if any(value < 0 or value > 1 for value in blend_weights):
        raise ValueError("blend weights must lie in [0,1]")

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame = load_frame(config).copy()
    raw_canonical = [
        feature for feature in CANONICAL_FEATURES if feature != PITCHER_TEAM_WIN_EXPECTANCY
    ]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype(np.float64)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    old_years = sorted(
        int(year)
        for year in frame.loc[frame[season] < args.calibration_year, season].unique()
    )
    if not old_years:
        raise ValueError("No old-era seasons found")

    train = frame.loc[frame[season] < args.validation_year].copy()
    validation = frame.loc[frame[season].eq(args.validation_year)].copy()
    if train.empty or validation.empty:
        raise ValueError("Training or validation partition is empty")

    f_train_mask = normalize_category(train["game_type"]).eq("F").to_numpy()
    f_valid_mask = normalize_category(validation["game_type"]).eq("F").to_numpy()
    train_f = train.loc[f_train_mask].copy()
    valid_f = validation.loc[f_valid_mask].copy()
    if train_f.empty or valid_f.empty:
        raise ValueError("F train/validation rows are required")

    y_train = train[target].to_numpy(np.float64)
    y_train_f = train_f[target].to_numpy(np.float64)
    y_valid = validation[target].to_numpy(np.float64)

    learned_delta, subgroup_pairs = corr.estimate_subgroup_delta(
        frame,
        old_years=old_years,
        calibration_year=args.calibration_year,
        season_col=season,
        target_col=target,
        min_group_rows=args.min_group_rows,
    )
    direct_delta = corr.overall_fr_delta(
        frame,
        old_years=old_years,
        calibration_year=args.calibration_year,
        season_col=season,
        target_col=target,
    )

    output_dir = Path(config["paths"]["output_dir"]) / "f_specialist_hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    subgroup_pairs.to_csv(output_dir / "calibration_subgroup_pairs.csv", index=False)

    logloss_params = catboost_params(
        config,
        args.iterations,
        args.task_type,
        args.devices,
        args.verbose,
        "Logloss",
    )
    crossentropy_params = catboost_params(
        config,
        args.iterations,
        args.task_type,
        args.devices,
        args.verbose,
        "CrossEntropy",
    )

    print(
        f"[F Specialist Hybrid] old={old_years}, calibration={args.calibration_year}, "
        f"validation={args.validation_year}, scales={scales}, blend_weights={blend_weights}, "
        f"iterations={args.iterations}, task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print(
        f"  train={len(train):,}, train_F={len(train_f):,}, "
        f"validation={len(validation):,}, validation_F={len(valid_f):,}"
    )
    print(f"  subgroup-derived old-scale F correction={learned_delta:+.6f}")
    print(f"  direct overall F-R correction={direct_delta:+.6f}")
    print(f"  {args.validation_year} labels are used ONLY for final scoring")

    rows: list[dict] = []
    parameter_rows: list[dict] = []

    def record(
        name: str,
        family: str,
        scale: float | None,
        blend_weight: float | None,
        prediction: np.ndarray,
        detail: str,
    ) -> None:
        metric = detailed_metrics(y_valid, prediction, validation)
        rows.append(
            {
                "name": name,
                "family": family,
                "correction_scale": scale,
                "blend_weight": blend_weight,
                "detail": detail,
                **metric,
            }
        )

    print("\n[1] Full-history raw baseline")
    p_raw = train_predict(
        train,
        validation,
        y_train,
        FULL_FEATURES,
        logloss_params,
    )
    record(
        "full_raw",
        "baseline",
        None,
        None,
        p_raw,
        "2019-2023 full model with raw binary targets",
    )

    print("[2] F-only raw specialist")
    p_f_raw = train_predict(
        train_f,
        valid_f,
        y_train_f,
        F_FEATURES,
        logloss_params,
    )
    for weight in blend_weights:
        hybrid = blend_f(p_raw, validation, p_f_raw, weight)
        record(
            f"hybrid_Fraw_w{weight:.2f}",
            "f_specialist_raw",
            None,
            weight,
            hybrid,
            "R=full_raw; F=blend(full_raw, F-only raw specialist)",
        )

    total_scale_steps = len(scales)
    for index, scale in enumerate(scales, start=1):
        delta = float(scale * learned_delta)
        print(
            f"[{index + 2}/{total_scale_steps + 2}] corrected training "
            f"scale={scale:.3f}, delta={delta:+.6f}"
        )

        # A) Corrected full model. Only its F predictions are allowed to replace the
        # raw baseline, so any collateral movement in R is explicitly blocked.
        soft_y_full, soft_info_full = corr.build_soft_targets(
            train,
            y_train,
            season_col=season,
            calibration_year=args.calibration_year,
            delta=delta,
        )
        p_full_soft_latent = train_predict(
            train,
            validation,
            soft_y_full,
            FULL_FEATURES,
            crossentropy_params,
        )
        p_full_soft_inverse = corr.invert_soft_prediction(
            p_full_soft_latent,
            validation,
            soft_negative_target=soft_info_full["soft_negative_target"],
        )
        p_full_soft_f = p_full_soft_inverse[f_valid_mask]
        for weight in blend_weights:
            hybrid = blend_f(p_raw, validation, p_full_soft_f, weight)
            record(
                f"hybrid_fullsoft_s{scale:.2f}_w{weight:.2f}",
                "corrected_full_F_only",
                scale,
                weight,
                hybrid,
                "R=full_raw; F=blend(full_raw, full-model soft-corrected inverse F)",
            )

        parameter_rows.append(
            {
                "model_scope": "full",
                "family": "soft_target",
                "correction_scale": scale,
                "delta": delta,
                **soft_info_full,
            }
        )
        del soft_y_full, p_full_soft_latent, p_full_soft_inverse, p_full_soft_f
        gc.collect()

        # B) F-only specialist with soft-target normalization.
        soft_y_f, soft_info_f = corr.build_soft_targets(
            train_f,
            y_train_f,
            season_col=season,
            calibration_year=args.calibration_year,
            delta=delta,
        )
        p_f_soft_latent = train_predict(
            train_f,
            valid_f,
            soft_y_f,
            F_FEATURES,
            crossentropy_params,
        )
        p_f_soft_inverse = corr.invert_soft_prediction(
            p_f_soft_latent,
            valid_f,
            soft_negative_target=soft_info_f["soft_negative_target"],
        )
        for weight in blend_weights:
            hybrid = blend_f(p_raw, validation, p_f_soft_inverse, weight)
            record(
                f"hybrid_Fsoft_s{scale:.2f}_w{weight:.2f}",
                "f_specialist_soft",
                scale,
                weight,
                hybrid,
                "R=full_raw; F=blend(full_raw, F-only soft-corrected inverse specialist)",
            )

        parameter_rows.append(
            {
                "model_scope": "F_only",
                "family": "soft_target",
                "correction_scale": scale,
                "delta": delta,
                **soft_info_f,
            }
        )
        del soft_y_f, p_f_soft_latent, p_f_soft_inverse
        gc.collect()

        # C) F-only specialist with binary labels and class reweighting.
        weight_f, weight_info_f = corr.build_reweighting(
            train_f,
            y_train_f,
            season_col=season,
            calibration_year=args.calibration_year,
            delta=delta,
        )
        p_f_weight_latent = train_predict(
            train_f,
            valid_f,
            y_train_f,
            F_FEATURES,
            logloss_params,
            sample_weight=weight_f,
        )
        p_f_weight_inverse = corr.invert_weighted_prediction(
            p_f_weight_latent,
            valid_f,
            positive_to_negative_odds_ratio=weight_info_f[
                "positive_to_negative_odds_ratio"
            ],
        )
        for blend_weight in blend_weights:
            hybrid = blend_f(
                p_raw,
                validation,
                p_f_weight_inverse,
                blend_weight,
            )
            record(
                f"hybrid_Fweight_s{scale:.2f}_w{blend_weight:.2f}",
                "f_specialist_weight",
                scale,
                blend_weight,
                hybrid,
                "R=full_raw; F=blend(full_raw, F-only reweighted inverse specialist)",
            )

        parameter_rows.append(
            {
                "model_scope": "F_only",
                "family": "class_reweight",
                "correction_scale": scale,
                "delta": delta,
                **weight_info_f,
            }
        )
        del weight_f, p_f_weight_latent, p_f_weight_inverse
        gc.collect()

    results = pd.DataFrame(rows)
    baseline_brier = float(results.loc[results["name"].eq("full_raw"), "brier"].iloc[0])
    baseline_score = float(
        results.loc[results["name"].eq("full_raw"), "competition_score"].iloc[0]
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
            "blend_weights": blend_weights,
            "subgroup_derived_delta": float(learned_delta),
            "direct_overall_fr_delta": float(direct_delta),
            "min_group_rows": int(args.min_group_rows),
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices,
            "full_features": FULL_FEATURES,
            "f_specialist_features": F_FEATURES,
            "canonical_invariants": invariant_check,
            "selection_warning": (
                "2024 is a validation fold. Multiple correction scales and F blend weights are "
                "reported for diagnosis; do not automatically use the single best 2024 setting "
                "for 2025 without a fixed/conservative selection rule."
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

    family_summary = (
        results.groupby("family", as_index=False)
        .agg(
            best_brier=("brier", "min"),
            best_delta_brier=("delta_brier_vs_raw", "min"),
            mean_brier=("brier", "mean"),
        )
        .sort_values("best_brier")
    )
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)

    print("\n[Best by family]")
    print(
        family_summary.to_string(
            index=False,
            formatters={
                "best_brier": "{:.8f}".format,
                "best_delta_brier": "{:+.8f}".format,
                "mean_brier": "{:.8f}".format,
            },
        )
    )

    best = results.iloc[0]
    print(
        f"\nBest={best['name']} brier={best['brier']:.8f} "
        f"score={best['competition_score']:.2f} "
        f"delta_vs_raw={best['delta_brier_vs_raw']:+.8f}"
    )
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
