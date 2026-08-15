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

BASE_FEATURES = list(CANONICAL_FEATURES)
ERA_FEATURE = "game_type_regime"
core.CATEGORICAL.add(ERA_FEATURE)


@dataclass
class Pred:
    name: str
    values: np.ndarray
    family: str
    detail: str
    parameter: float | None = None


def parse_floats(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values or any(v < 0 for v in values):
        raise ValueError("--alphas must contain non-negative numbers")
    return values


def normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def catboost_params(config: dict, args: argparse.Namespace) -> dict:
    p = {
        "iterations": int(args.iterations),
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
        "task_type": args.task_type,
        "verbose": args.verbose,
    }
    if args.task_type == "GPU":
        p["devices"] = args.devices
    return p


def train_model(train: pd.DataFrame, target: str, features: list[str], params: dict):
    from catboost import CatBoostClassifier, Pool

    x, cat = core.prepare_x(train, features)
    y = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
    pool = Pool(x, label=y, cat_features=cat, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(pool, verbose=params.get("verbose", 0))
    return model


def predict_model(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    from catboost import Pool

    x, cat = core.prepare_x(frame, features)
    pool = Pool(x, cat_features=cat, feature_names=features)
    return model.predict_proba(pool)[:, 1].astype(np.float64)


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def fit_logit_intercept(y: np.ndarray, p: np.ndarray) -> float:
    target_mean = float(np.mean(y))
    z = safe_logit(p)
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(sigmoid(z + mid).mean()) < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def shrink(raw: pd.Series, support: pd.Series, global_value: float, alpha: float) -> dict[str, float]:
    support = support.astype(float)
    rel = pd.Series(1.0, index=support.index) if alpha == 0 else support / (support + alpha)
    out = global_value + rel * (raw - global_value)
    return {str(k): float(v) for k, v in out.items()}


def apply_f_additive(p: np.ndarray, frame: pd.DataFrame, global_value: float, by_team: dict[str, float] | None = None, scale: float = 1.0) -> np.ndarray:
    out = np.asarray(p, dtype=np.float64).copy()
    f = normalize(frame["game_type"]).eq("F").to_numpy()
    if by_team is None:
        delta = np.full(len(frame), global_value)
    else:
        delta = normalize(frame["pitcher_team_id"]).map(by_team).fillna(global_value).to_numpy(float)
    out[f] += scale * delta[f]
    return np.clip(out, 0, 1)


def apply_f_logit(p: np.ndarray, frame: pd.DataFrame, global_value: float, by_team: dict[str, float] | None = None, scale: float = 1.0) -> np.ndarray:
    out = np.asarray(p, dtype=np.float64).copy()
    f = normalize(frame["game_type"]).eq("F").to_numpy()
    if by_team is None:
        delta = np.full(len(frame), global_value)
    else:
        delta = normalize(frame["pitcher_team_id"]).map(by_team).fillna(global_value).to_numpy(float)
    out[f] = sigmoid(safe_logit(out[f]) + scale * delta[f])
    return np.clip(out, 0, 1)


def residual_corrections(cal: pd.DataFrame, y: np.ndarray, p: np.ndarray):
    work = pd.DataFrame({
        "game_type": normalize(cal["game_type"]).to_numpy(),
        "team": normalize(cal["pitcher_team_id"]).to_numpy(),
        "y": y,
        "p": p,
    })
    f = work.loc[work["game_type"].eq("F")].copy()
    global_add = float((f["y"] - f["p"]).mean())
    global_logit = fit_logit_intercept(f["y"].to_numpy(), f["p"].to_numpy())
    grouped = f.groupby("team", observed=True).agg(y=("y", "mean"), p=("p", "mean"), rows=("y", "size"))
    team_add = grouped["y"] - grouped["p"]
    team_logit = pd.Series({
        str(team): fit_logit_intercept(g["y"].to_numpy(), g["p"].to_numpy())
        for team, g in f.groupby("team", observed=True)
    }, dtype=float)
    team_add.index = team_add.index.map(str)
    support = grouped["rows"].copy()
    support.index = support.index.map(str)
    return global_add, global_logit, team_add, team_logit, support


def metrics(y: np.ndarray, p: np.ndarray, frame: pd.DataFrame) -> dict:
    out = dict(core.metrics(y, p))
    gt = normalize(frame["game_type"])
    for name in ("F", "R"):
        m = gt.eq(name).to_numpy()
        yy, pp = y[m], p[m]
        key = name.lower()
        out[f"{key}_rows"] = int(m.sum())
        out[f"{key}_rate"] = float(yy.mean())
        out[f"{key}_pred_mean"] = float(pp.mean())
        out[f"{key}_calibration_gap"] = float(pp.mean() - yy.mean())
        out[f"{key}_brier"] = float(np.mean((pp - yy) ** 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward F-measurement correction experiments. Defaults to learn on 2023 and score on 2024.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--calibration-year", type=int, default=2023)
    ap.add_argument("--validation-year", type=int, default=2024)
    ap.add_argument("--regime-start-year", type=int, default=2023)
    ap.add_argument("--min-group-rows", type=int, default=100)
    ap.add_argument("--alphas", default="0,100,500,2000,10000")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args()

    import catboost

    if args.validation_year <= args.calibration_year:
        raise ValueError("validation year must be later than calibration year")
    alphas = parse_floats(args.alphas)
    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame = load_frame(config).copy()
    raw_features = [x for x in CANONICAL_FEATURES if x != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_features + CANONICAL_SOURCE_COLUMNS + [target, season, row_id, "pitcher_team_id"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariants = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype(float)
    frame[ERA_FEATURE] = normalize(frame["game_type"]) + "_" + np.where(frame[season] >= args.regime_start_year, "new", "old")
    offset_diag.add_experience_bucket(frame)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    old = frame.loc[frame[season] < args.calibration_year].copy()
    cal = frame.loc[frame[season].eq(args.calibration_year)].copy()
    full = frame.loc[frame[season] < args.validation_year].copy()
    val = frame.loc[frame[season].eq(args.validation_year)].copy()
    recent = cal.copy()
    if min(len(old), len(cal), len(full), len(val)) == 0:
        raise ValueError("empty temporal partition")

    params = catboost_params(config, args)
    out_dir = Path(config["paths"]["output_dir"]) / "game_type_correction_suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    old_years = sorted(old[season].unique().astype(int).tolist())

    print(f"[Correction Suite] old={old_years}, calibration={args.calibration_year}, validation={args.validation_year}")
    print(f"[Correction Suite] alphas={alphas}, iterations={args.iterations}, task_type={args.task_type}, catboost={catboost.__version__}")
    print(f"[Correction Suite] {args.validation_year} target is used only for final scoring")

    # 1) Old model: cleanest test of whether a correction learned on 2023 can rescue an old-regime model on 2024.
    print("\n[1/5] old-only model")
    old_model = train_model(old, target, BASE_FEATURES, params)
    p_old_cal = predict_model(old_model, cal, BASE_FEATURES)
    p_old_val = predict_model(old_model, val, BASE_FEATURES)
    y_cal = cal[target].to_numpy(float)
    y_val = val[target].to_numpy(float)

    # Measurement-effect correction estimated using 2023 only. This reuses the exact subgroup logic from the offset diagnostic.
    pairs = offset_diag.build_subgroup_pairs(
        frame,
        old_years=old_years,
        new_years=[args.calibration_year],
        season_col=season,
        target_col=target,
        game_type_col="game_type",
        min_group_rows=args.min_group_rows,
    )
    if pairs.empty:
        raise ValueError("No matched subgroup pairs; lower --min-group-rows")
    common_measurement_delta = float(np.average(pairs["raw_residual"], weights=pairs["weight"]))
    team_pairs = pairs.loc[pairs["family"].eq("pitcher_team")].copy()
    team_effect = team_pairs.set_index("group_key")["raw_residual"]
    team_effect_support = team_pairs.set_index("group_key")["weight"]

    # Simple global F-R correction as a separate, less structured baseline.
    old_year_effects = []
    for year in old_years:
        rates = old.loc[old[season].eq(year)].groupby("game_type", observed=True)[target].mean()
        if {"F", "R"}.issubset(rates.index):
            old_year_effects.append(float(rates["F"] - rates["R"]))
    old_macro_effect = float(np.mean(old_year_effects))
    cal_rates = cal.groupby("game_type", observed=True)[target].mean()
    direct_effect_delta = float((cal_rates["F"] - cal_rates["R"]) - old_macro_effect)

    residual_add, residual_logit, team_residual_add, team_residual_logit, team_residual_support = residual_corrections(cal, y_cal, p_old_cal)
    print(f"  correction to NEW observed F scale: subgroup={common_measurement_delta:+.6f}, direct_F-R={direct_effect_delta:+.6f}")
    print(f"  old-model 2023 residual: additive={residual_add:+.6f}, logit={residual_logit:+.6f}")

    preds: list[Pred] = [Pred("old_only_raw", p_old_val, "baseline", "train old years only")]
    preds += [
        Pred("old_only_measurement_common", apply_f_additive(p_old_val, val, common_measurement_delta), "measurement", "2023-only subgroup common offset"),
        Pred("old_only_direct_FR", apply_f_additive(p_old_val, val, direct_effect_delta), "measurement", "2023-only global F-R offset"),
        Pred("old_only_residual_add", apply_f_additive(p_old_val, val, residual_add), "calibration", "2023 F mean residual"),
        Pred("old_only_residual_logit", apply_f_logit(p_old_val, val, residual_logit), "calibration", "2023 F logit intercept"),
    ]

    team_export = []
    for alpha in alphas:
        team_m = shrink(team_effect, team_effect_support, common_measurement_delta, alpha)
        team_a = shrink(team_residual_add, team_residual_support, residual_add, alpha)
        common_idx = team_residual_logit.index.intersection(team_residual_support.index)
        team_l = shrink(team_residual_logit.loc[common_idx], team_residual_support.loc[common_idx], residual_logit, alpha)
        preds += [
            Pred(f"old_only_measurement_team_a{alpha:g}", apply_f_additive(p_old_val, val, common_measurement_delta, team_m), "measurement_team", "shrunk team measurement offset", alpha),
            Pred(f"old_only_residual_team_a{alpha:g}", apply_f_additive(p_old_val, val, residual_add, team_a), "calibration_team", "shrunk team additive residual", alpha),
            Pred(f"old_only_logit_team_a{alpha:g}", apply_f_logit(p_old_val, val, residual_logit, team_l), "calibration_team", "shrunk team logit residual", alpha),
        ]
        for team in sorted(set(team_m) | set(team_a) | set(team_l)):
            team_export.append({"alpha": alpha, "team": team, "measurement": team_m.get(team, np.nan), "residual_add": team_a.get(team, np.nan), "residual_logit": team_l.get(team, np.nan)})

    # 2) Recent-only model for a known comparison and possible complementarity.
    print("[2/5] recent-only model")
    recent_model = train_model(recent, target, BASE_FEATURES, params)
    p_recent = predict_model(recent_model, val, BASE_FEATURES)
    preds.append(Pred("recent_only_raw", p_recent, "baseline", f"train {args.calibration_year} only"))
    del recent_model
    gc.collect()

    # 3) Standard full-history reference.
    print("[3/5] full-history raw model")
    full_model = train_model(full, target, BASE_FEATURES, params)
    p_full = predict_model(full_model, val, BASE_FEATURES)
    preds.append(Pred("full_history_raw", p_full, "baseline", "all prior years"))
    del full_model
    gc.collect()

    # 4/5) Let CatBoost explicitly distinguish F_old from F_new instead of forcing one game_type meaning.
    print("[4/5] full-history + game_type_regime")
    add_features = [*BASE_FEATURES, ERA_FEATURE]
    add_model = train_model(full, target, add_features, params)
    p_add_era = predict_model(add_model, val, add_features)
    preds.append(Pred("full_add_game_type_regime", p_add_era, "regime_feature", "raw game_type plus F_old/F_new interaction"))
    del add_model
    gc.collect()

    print("[5/5] full-history replace game_type with game_type_regime")
    replace_features = [x for x in BASE_FEATURES if x != "game_type"] + [ERA_FEATURE]
    replace_model = train_model(full, target, replace_features, params)
    p_replace_era = predict_model(replace_model, val, replace_features)
    preds.append(Pred("full_replace_game_type_regime", p_replace_era, "regime_feature", "replace raw game_type by F_old/F_new interaction"))
    del replace_model
    gc.collect()

    # Fixed diagnostic blends. These are not automatically selected for final 2025 use.
    for w in (0.25, 0.50, 0.75):
        preds.append(Pred(f"blend_full_recent_w{w:.2f}", (1-w)*p_full + w*p_recent, "blend", "full + recent-only", w))
        preds.append(Pred(f"blend_full_old_corrected_w{w:.2f}", (1-w)*p_full + w*apply_f_additive(p_old_val, val, common_measurement_delta), "blend", "full + measurement-corrected old model", w))
        preds.append(Pred(f"blend_full_add_era_w{w:.2f}", (1-w)*p_full + w*p_add_era, "blend", "full + explicit regime model", w))

    reference = metrics(y_val, p_full, val)
    rows = []
    for item in preds:
        m = metrics(y_val, np.clip(item.values, 0, 1), val)
        rows.append({
            "name": item.name,
            "family": item.family,
            "detail": item.detail,
            "parameter": item.parameter,
            **m,
            "delta_brier_vs_full": m["brier"] - reference["brier"],
            "delta_score_vs_full": m["competition_score"] - reference["competition_score"],
        })
    results = pd.DataFrame(rows).sort_values(["brier", "name"]).reset_index(drop=True)
    results.to_csv(out_dir / "results.csv", index=False)
    pairs.to_csv(out_dir / "calibration_subgroup_pairs.csv", index=False)
    pd.DataFrame(team_export).to_csv(out_dir / "team_corrections_by_alpha.csv", index=False)

    save_json({
        "old_years": old_years,
        "calibration_year": args.calibration_year,
        "validation_year": args.validation_year,
        "regime_start_year": args.regime_start_year,
        "min_group_rows": args.min_group_rows,
        "alphas": alphas,
        "iterations": args.iterations,
        "task_type": args.task_type,
        "canonical_invariants": invariants,
        "learned_from_calibration_only": {
            "common_measurement_delta_to_new_scale": common_measurement_delta,
            "direct_FR_delta_to_new_scale": direct_effect_delta,
            "old_model_F_residual_additive": residual_add,
            "old_model_F_residual_logit": residual_logit,
        },
        "leakage_rule": f"No {args.validation_year} target is used to estimate offsets, team corrections, or regime features. Validation labels are used only to score the predefined sweep.",
        "selection_warning": "Alpha and blend sweeps are diagnostic on the validation year. Do not blindly transfer the validation-best hyperparameter to 2025.",
    }, out_dir / "run_config.json")

    cols = ["name", "brier", "competition_score", "auc", "prediction_std", "f_brier", "f_calibration_gap", "r_brier", "delta_brier_vs_full"]
    print("\n[Validation results: lower Brier is better]")
    print(results[cols].to_string(index=False, formatters={
        "brier": "{:.8f}".format,
        "competition_score": "{:.2f}".format,
        "auc": "{:.5f}".format,
        "prediction_std": "{:.5f}".format,
        "f_brier": "{:.8f}".format,
        "f_calibration_gap": "{:+.6f}".format,
        "r_brier": "{:.8f}".format,
        "delta_brier_vs_full": "{:+.8f}".format,
    }))
    best = results.iloc[0]
    print(f"\nBest={best['name']} brier={best['brier']:.8f} score={best['competition_score']:.2f} delta_vs_full={best['delta_brier_vs_full']:+.8f}")
    print(f"Saved: {out_dir}")

    del old_model
    gc.collect()


if __name__ == "__main__":
    main()
