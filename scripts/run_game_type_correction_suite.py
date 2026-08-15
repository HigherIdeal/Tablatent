from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

BASE_FEATURES = list(CANONICAL_FEATURES)
ERA_FEATURE = "game_type_regime"


@dataclass
class PredictionSet:
    name: str
    prediction: np.ndarray
    source: str
    correction_family: str
    alpha: float | None = None


def parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one float value is required")
    return values


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


def normalize_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def add_game_type_regime(frame: pd.DataFrame, regime_start_year: int, season_col: str) -> None:
    game_type = normalize_category(frame["game_type"])
    era = np.where(frame[season_col].to_numpy() >= regime_start_year, "new", "old")
    frame[ERA_FEATURE] = game_type + "_" + pd.Series(era, index=frame.index, dtype="string").astype(str)


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, target_col: str, features: list[str], params: dict):
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = core.prepare_x(train, features)
    x_valid, _ = core.prepare_x(valid, features)
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=params.get("verbose", 0))
    prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
    return prediction, model


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def find_logit_intercept(y: np.ndarray, p: np.ndarray) -> float:
    y_mean = float(np.mean(y))
    z = safe_logit(p)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(sigmoid(z + mid).mean()) < y_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def apply_f_additive(prediction: np.ndarray, frame: pd.DataFrame, global_delta: float, team_delta: dict[str, float] | None = None) -> np.ndarray:
    out = np.asarray(prediction, dtype=np.float64).copy()
    f_mask = normalize_category(frame["game_type"]).eq("F").to_numpy()
    if team_delta is None:
        correction = np.full(len(frame), global_delta, dtype=np.float64)
    else:
        teams = normalize_category(frame["pitcher_team_id"])
        correction = teams.map(team_delta).fillna(global_delta).to_numpy(np.float64)
    out[f_mask] += correction[f_mask]
    return np.clip(out, 0.0, 1.0)


def apply_f_logit(prediction: np.ndarray, frame: pd.DataFrame, global_intercept: float, team_intercept: dict[str, float] | None = None) -> np.ndarray:
    out = np.asarray(prediction, dtype=np.float64).copy()
    f_mask = normalize_category(frame["game_type"]).eq("F").to_numpy()
    if team_intercept is None:
        correction = np.full(len(frame), global_intercept, dtype=np.float64)
    else:
        teams = normalize_category(frame["pitcher_team_id"])
        correction = teams.map(team_intercept).fillna(global_intercept).to_numpy(np.float64)
    out[f_mask] = sigmoid(safe_logit(out[f_mask]) + correction[f_mask])
    return np.clip(out, 0.0, 1.0)


def shrink_map(raw_values: pd.Series, counts: pd.Series, global_value: float, alpha: float) -> dict[str, float]:
    counts = counts.astype(float)
    if alpha == 0:
        reliability = pd.Series(1.0, index=counts.index)
    else:
        reliability = counts / (counts + float(alpha))
    shrunk = global_value + reliability * (raw_values - global_value)
    return {str(key): float(value) for key, value in shrunk.items()}


def effect_offsets(old: pd.DataFrame, calibration: pd.DataFrame, target_col: str) -> tuple[float, pd.Series, pd.Series]:
    def effect(frame: pd.DataFrame) -> float:
        rates = frame.groupby("game_type", observed=True)[target_col].mean()
        return float(rates["F"] - rates["R"])

    global_delta = effect(calibration) - effect(old)

    def team_effect(frame: pd.DataFrame) -> pd.DataFrame:
        grouped = frame.groupby(["pitcher_team_id", "game_type"], observed=True).agg(rate=(target_col, "mean"), rows=(target_col, "size")).reset_index()
        rate = grouped.pivot(index="pitcher_team_id", columns="game_type", values="rate")
        rows = grouped.pivot(index="pitcher_team_id", columns="game_type", values="rows")
        if not {"F", "R"}.issubset(rate.columns):
            return pd.DataFrame(columns=["effect", "rows"])
        common = rate.index[rate[["F", "R"]].notna().all(axis=1)]
        out = pd.DataFrame(index=common)
        out["effect"] = rate.loc[common, "F"] - rate.loc[common, "R"]
        out["rows"] = rows.loc[common, ["F", "R"]].min(axis=1)
        return out

    old_team = team_effect(old)
    new_team = team_effect(calibration)
    joined = old_team.join(new_team, lsuffix="_old", rsuffix="_new", how="inner")
    raw = joined["effect_new"] - joined["effect_old"]
    count = joined[["rows_old", "rows_new"]].min(axis=1)
    raw.index = raw.index.map(str)
    count.index = count.index.map(str)
    return global_delta, raw, count


def residual_offsets(calibration: pd.DataFrame, y: np.ndarray, prediction: np.ndarray) -> tuple[float, pd.Series, pd.Series, float, pd.Series]:
    work = pd.DataFrame({
        "game_type": normalize_category(calibration["game_type"]).to_numpy(),
        "team": normalize_category(calibration["pitcher_team_id"]).to_numpy(),
        "y": np.asarray(y, dtype=np.float64),
        "p": np.asarray(prediction, dtype=np.float64),
    })
    f = work.loc[work["game_type"].eq("F")].copy()
    global_add = float((f["y"] - f["p"]).mean())
    team = f.groupby("team", observed=True).agg(y=("y", "mean"), p=("p", "mean"), rows=("y", "size"))
    team_add = team["y"] - team["p"]
    team_count = team["rows"]
    global_logit = find_logit_intercept(f["y"].to_numpy(), f["p"].to_numpy())
    team_logit_values: dict[str, float] = {}
    for team_id, group in f.groupby("team", observed=True):
        team_logit_values[str(team_id)] = find_logit_intercept(group["y"].to_numpy(np.float64), group["p"].to_numpy(np.float64))
    team_logit = pd.Series(team_logit_values, dtype=np.float64)
    team_add.index = team_add.index.map(str)
    team_count.index = team_count.index.map(str)
    return global_add, team_add, team_count, global_logit, team_logit


def detailed_metrics(y: np.ndarray, p: np.ndarray, frame: pd.DataFrame) -> dict:
    result = dict(core.metrics(y, p))
    game_type = normalize_category(frame["game_type"])
    for label in ("F", "R"):
        mask = game_type.eq(label).to_numpy()
        if not mask.any():
            continue
        y_part = np.asarray(y, dtype=np.float64)[mask]
        p_part = np.asarray(p, dtype=np.float64)[mask]
        prefix = label.lower()
        result[f"{prefix}_rows"] = int(mask.sum())
        result[f"{prefix}_rate"] = float(y_part.mean())
        result[f"{prefix}_pred_mean"] = float(p_part.mean())
        result[f"{prefix}_calibration_gap"] = float(p_part.mean() - y_part.mean())
        result[f"{prefix}_brier"] = float(np.mean(np.square(p_part - y_part)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe 2023->2024 suite for F measurement-shift corrections and explicit regime features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration-year", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--alphas", default="0,100,500,2000,10000")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    import catboost
    from catboost import Pool

    if args.validation_year <= args.calibration_year:
        raise ValueError("validation-year must be after calibration-year")
    alphas = parse_floats(args.alphas)
    if any(alpha < 0 for alpha in alphas):
        raise ValueError("alphas must be >= 0")

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    frame = load_frame(config).copy()

    raw_canonical = [feature for feature in CANONICAL_FEATURES if feature != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id, "pitcher_team_id"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype(np.float64)
    add_game_type_regime(frame, args.regime_start_year, season)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    old = frame.loc[frame[season] < args.calibration_year].copy()
    calibration = frame.loc[frame[season].eq(args.calibration_year)].copy()
    full = frame.loc[frame[season] < args.validation_year].copy()
    validation = frame.loc[frame[season].eq(args.validation_year)].copy()
    if any(part.empty for part in (old, calibration, full, validation)):
        raise ValueError("One of old/calibration/full/validation partitions is empty")

    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)
    output_dir = Path(config["paths"]["output_dir"]) / "game_type_correction_suite"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Game-Type Correction Suite] calibration={args.calibration_year}, validation={args.validation_year}, alphas={alphas}, iterations={args.iterations}, task_type={args.task_type}, catboost={catboost.__version__}")
    print(f"  old={len(old):,} ({int(old[season].min())}-{int(old[season].max())}), calibration={len(calibration):,}, full={len(full):,}, validation={len(validation):,}")
    print(f"  {args.validation_year} labels are used ONLY for final scoring, never for estimating corrections")

    print("\n[1/4] old-only raw model: train old era, predict calibration and validation")
    p_old_cal, old_model = fit_predict(old, calibration, target, BASE_FEATURES, params)
    x_old_valid, old_cat = core.prepare_x(validation, BASE_FEATURES)
    old_valid_pool = Pool(x_old_valid, cat_features=old_cat, feature_names=BASE_FEATURES)
    p_old_val = old_model.predict_proba(old_valid_pool)[:, 1].astype(np.float64)
    y_cal = calibration[target].to_numpy(np.float64)
    y_val = validation[target].to_numpy(np.float64)

    effect_global, effect_team_raw, effect_team_count = effect_offsets(old, calibration, target)
    residual_global, residual_team_raw, residual_team_count, logit_global, logit_team_raw = residual_offsets(calibration, y_cal, p_old_cal)

    print(f"  learned global corrections from <= {args.calibration_year}: effect_add={effect_global:+.6f}, residual_add={residual_global:+.6f}, logit_intercept={logit_global:+.6f}")

    predictions: list[PredictionSet] = [
        PredictionSet("old_only_raw", p_old_val, f"train<{args.calibration_year}", "none"),
        PredictionSet("old_only_effect_global", apply_f_additive(p_old_val, validation, effect_global), "old model + F-R shift learned on calibration year", "effect_additive_global"),
        PredictionSet("old_only_residual_global", apply_f_additive(p_old_val, validation, residual_global), "old model + F forward residual learned on calibration year", "residual_additive_global"),
        PredictionSet("old_only_logit_global", apply_f_logit(p_old_val, validation, logit_global), "old model + F logit intercept learned on calibration year", "residual_logit_global"),
    ]

    team_export_rows: list[dict] = []
    for alpha in alphas:
        effect_team = shrink_map(effect_team_raw, effect_team_count, effect_global, alpha)
        residual_team = shrink_map(residual_team_raw, residual_team_count, residual_global, alpha)
        common_logit = logit_team_raw.reindex(residual_team_count.index).dropna()
        common_counts = residual_team_count.reindex(common_logit.index)
        logit_team = shrink_map(common_logit, common_counts, logit_global, alpha)
        predictions.extend([
            PredictionSet(f"old_only_effect_team_a{alpha:g}", apply_f_additive(p_old_val, validation, effect_global, effect_team), "old model + shrunk team F-R shift", "effect_additive_team", alpha),
            PredictionSet(f"old_only_residual_team_a{alpha:g}", apply_f_additive(p_old_val, validation, residual_global, residual_team), "old model + shrunk team forward residual", "residual_additive_team", alpha),
            PredictionSet(f"old_only_logit_team_a{alpha:g}", apply_f_logit(p_old_val, validation, logit_global, logit_team), "old model + shrunk team logit intercept", "residual_logit_team", alpha),
        ])
        for team_id in sorted(set(effect_team) | set(residual_team) | set(logit_team)):
            team_export_rows.append({
                "alpha": alpha,
                "pitcher_team_id": team_id,
                "effect_add": effect_team.get(team_id, np.nan),
                "residual_add": residual_team.get(team_id, np.nan),
                "residual_logit": logit_team.get(team_id, np.nan),
            })

    print("[2/4] full-history raw model")
    p_full_raw, full_raw_model = fit_predict(full, validation, target, BASE_FEATURES, params)
    predictions.append(PredictionSet("full_history_raw", p_full_raw, f"train<{args.validation_year}", "none"))
    del full_raw_model
    gc.collect()

    print("[3/4] full-history + explicit game_type_regime")
    features_add_era = [*BASE_FEATURES, ERA_FEATURE]
    p_era_add, era_add_model = fit_predict(full, validation, target, features_add_era, params)
    predictions.append(PredictionSet("full_history_add_game_type_regime", p_era_add, f"train<{args.validation_year}", "explicit_regime_feature"))
    del era_add_model
    gc.collect()

    print("[4/4] full-history replace raw game_type with game_type_regime")
    features_replace_era = [feature for feature in BASE_FEATURES if feature != "game_type"] + [ERA_FEATURE]
    p_era_replace, era_replace_model = fit_predict(full, validation, target, features_replace_era, params)
    predictions.append(PredictionSet("full_history_replace_game_type_regime", p_era_replace, f"train<{args.validation_year}", "explicit_regime_feature"))
    del era_replace_model
    gc.collect()

    # Diagnostic blends: useful only to see whether corrected old-model signal is complementary.
    blend_sources = {
        "effect_global": predictions[1].prediction,
        "residual_global": predictions[2].prediction,
        "logit_global": predictions[3].prediction,
    }
    for label, corrected in blend_sources.items():
        for weight in (0.25, 0.50, 0.75):
            blend = (1.0 - weight) * p_full_raw + weight * corrected
            predictions.append(PredictionSet(f"blend_full_{label}_w{weight:.2f}", np.clip(blend, 0.0, 1.0), "diagnostic blend", "blend", weight))

    raw_reference = detailed_metrics(y_val, p_full_raw, validation)
    result_rows: list[dict] = []
    for pred_set in predictions:
        metric = detailed_metrics(y_val, pred_set.prediction, validation)
        result_rows.append({
            "name": pred_set.name,
            "source": pred_set.source,
            "correction_family": pred_set.correction_family,
            "alpha_or_weight": pred_set.alpha,
            **metric,
            "delta_brier_vs_full_raw": metric["brier"] - raw_reference["brier"],
            "delta_score_vs_full_raw": metric["competition_score"] - raw_reference["competition_score"],
        })

    results = pd.DataFrame(result_rows).sort_values(["brier", "name"]).reset_index(drop=True)
    results.to_csv(output_dir / "results.csv", index=False)
    pd.DataFrame([
        {"family": "effect_additive", "global_value": effect_global, "description": "(calibration F-R) - (old F-R); add to old-scale F probability"},
        {"family": "model_residual_additive", "global_value": residual_global, "description": "mean(y-p) on calibration-year F from old-only model"},
        {"family": "model_residual_logit", "global_value": logit_global, "description": "logit intercept matching calibration-year F mean from old-only model"},
    ]).to_csv(output_dir / "global_corrections.csv", index=False)
    pd.DataFrame(team_export_rows).to_csv(output_dir / "team_corrections_by_alpha.csv", index=False)

    raw_team_rows: list[dict] = []
    team_ids = sorted(set(effect_team_raw.index) | set(residual_team_raw.index) | set(logit_team_raw.index))
    for team_id in team_ids:
        raw_team_rows.append({
            "pitcher_team_id": team_id,
            "effect_add_raw": float(effect_team_raw.get(team_id, np.nan)),
            "effect_support": float(effect_team_count.get(team_id, np.nan)),
            "residual_add_raw": float(residual_team_raw.get(team_id, np.nan)),
            "residual_support": float(residual_team_count.get(team_id, np.nan)),
            "residual_logit_raw": float(logit_team_raw.get(team_id, np.nan)),
        })
    pd.DataFrame(raw_team_rows).to_csv(output_dir / "team_corrections_raw.csv", index=False)

    cal_metric = detailed_metrics(y_cal, p_old_cal, calibration)
    save_json({
        "calibration_year": int(args.calibration_year),
        "validation_year": int(args.validation_year),
        "regime_start_year": int(args.regime_start_year),
        "alphas": alphas,
        "iterations": int(args.iterations),
        "task_type": args.task_type,
        "devices": args.devices,
        "canonical_invariants": invariant_check,
        "catboost_params": params,
        "calibration_forward_metrics_raw_old_model": cal_metric,
        "learned_global_corrections": {
            "effect_additive": effect_global,
            "model_residual_additive": residual_global,
            "model_residual_logit": logit_global,
        },
        "leakage_rule": f"All correction parameters are fit using years <= {args.calibration_year}; {args.validation_year} labels are used only for final scoring.",
        "warning": "Team-shrinkage alphas and blend weights are swept on validation for diagnosis; do not blindly copy the validation-best setting into final 2025 inference.",
    }, output_dir / "run_config.json")

    display_cols = ["name", "brier", "competition_score", "auc", "prediction_std", "f_brier", "f_calibration_gap", "r_brier", "delta_brier_vs_full_raw"]
    print("\n[Validation results] lower Brier is better")
    print(results[display_cols].to_string(index=False, formatters={
        "brier": "{:.8f}".format,
        "competition_score": "{:.2f}".format,
        "auc": "{:.5f}".format,
        "prediction_std": "{:.5f}".format,
        "f_brier": "{:.8f}".format,
        "f_calibration_gap": "{:+.6f}".format,
        "r_brier": "{:.8f}".format,
        "delta_brier_vs_full_raw": "{:+.8f}".format,
    }))
    best = results.iloc[0]
    print(f"\nBest: {best['name']} brier={best['brier']:.8f} score={best['competition_score']:.2f} delta_vs_full_raw={best['delta_brier_vs_full_raw']:+.8f}")
    print(f"Saved: {output_dir}")

    del old_model, old_valid_pool, x_old_valid
    gc.collect()


if __name__ == "__main__":
    main()
