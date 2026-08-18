from __future__ import annotations

import argparse
import gc
import json
import sys
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


SUCCESS_STATE = [
    "eng_ps_prev1_minus_long",
    "eng_ps_prev3_minus_long",
    "eng_ps_prev5_minus_long",
    "eng_ps_prev1_minus_prev3",
    "eng_ps_prev3_minus_prev5",
    "eng_ps_prev1_minus_prev5",
    "eng_ps_recent_mean_135",
    "eng_ps_recent_mean_minus_long",
    "eng_ps_recent_range_135",
]

MIDDLE_STATE = [
    "eng_pm_prev1_minus_long",
    "eng_pm_prev3_minus_long",
    "eng_pm_prev5_minus_long",
    "eng_pm_prev1_minus_prev3",
    "eng_pm_prev3_minus_prev5",
    "eng_pm_prev1_minus_prev5",
    "eng_pm_recent_mean_135",
    "eng_pm_recent_mean_minus_long",
    "eng_pm_recent_range_135",
]

MATCHUP_STATE = [
    "eng_long_pitcher_minus_batter_success",
    "eng_prev1_pitcher_minus_batter_success",
    "eng_prev3_pitcher_minus_batter_success",
    "eng_prev5_pitcher_minus_batter_success",
    "eng_recent_mean_pitcher_minus_batter_success",
    "eng_long_pitcher_minus_batter_middle",
    "eng_prev1_pitcher_minus_batter_middle",
    "eng_prev3_pitcher_minus_batter_middle",
    "eng_prev5_pitcher_minus_batter_middle",
]

EXPERIENCE_MODULATION = [
    "eng_pitcher_stale_weight_2000",
    "eng_ps_prev1_minus_long_x_stale2000",
    "eng_ps_prev3_minus_long_x_stale2000",
    "eng_ps_prev5_minus_long_x_stale2000",
    "eng_ps_recent_mean_minus_long_x_stale2000",
    "eng_ps_prev1_minus_prev5_x_stale2000",
    "eng_pm_prev1_minus_long_x_stale2000",
    "eng_pm_prev3_minus_long_x_stale2000",
    "eng_pm_prev5_minus_long_x_stale2000",
    "eng_pm_recent_mean_minus_long_x_stale2000",
    "eng_pm_prev1_minus_prev5_x_stale2000",
]

VARIANT_GROUPS = {
    "reference_canonical": [],
    "add_success_state": SUCCESS_STATE,
    "add_middle_state": MIDDLE_STATE,
    "add_matchup_state": MATCHUP_STATE,
    "add_recent_state": SUCCESS_STATE + MIDDLE_STATE,
    "add_recent_experience": SUCCESS_STATE + MIDDLE_STATE + EXPERIENCE_MODULATION,
    "add_all_state": SUCCESS_STATE + MIDDLE_STATE + MATCHUP_STATE + EXPERIENCE_MODULATION,
}


def parse_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("at least one fold is required")
    return result


def parse_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype(np.float32)


def add_asof_state_features(frame: pd.DataFrame) -> None:
    ps_long = numeric(frame, "asof_pitcher_success_rate")
    ps1 = numeric(frame, "asof_pitcher_prev1_game_success_rate")
    ps3 = numeric(frame, "asof_pitcher_prev3_game_success_rate")
    ps5 = numeric(frame, "asof_pitcher_prev5_game_success_rate")

    pm_long = numeric(frame, "asof_pitcher_middle_rate")
    pm1 = numeric(frame, "asof_pitcher_prev1_game_middle_rate")
    pm3 = numeric(frame, "asof_pitcher_prev3_game_middle_rate")
    pm5 = numeric(frame, "asof_pitcher_prev5_game_middle_rate")

    bs_long = numeric(frame, "asof_batter_success_rate")
    bm_long = numeric(frame, "asof_batter_middle_rate")

    # Success: current/recent state relative to long-run pitcher history.
    frame["eng_ps_prev1_minus_long"] = ps1 - ps_long
    frame["eng_ps_prev3_minus_long"] = ps3 - ps_long
    frame["eng_ps_prev5_minus_long"] = ps5 - ps_long
    frame["eng_ps_prev1_minus_prev3"] = ps1 - ps3
    frame["eng_ps_prev3_minus_prev5"] = ps3 - ps5
    frame["eng_ps_prev1_minus_prev5"] = ps1 - ps5
    ps_stack = pd.concat([ps1, ps3, ps5], axis=1)
    frame["eng_ps_recent_mean_135"] = ps_stack.mean(axis=1, skipna=False)
    frame["eng_ps_recent_mean_minus_long"] = frame["eng_ps_recent_mean_135"] - ps_long
    frame["eng_ps_recent_range_135"] = ps_stack.max(axis=1, skipna=False) - ps_stack.min(axis=1, skipna=False)

    # Middle-rate state: same construction, kept separate for attribution.
    frame["eng_pm_prev1_minus_long"] = pm1 - pm_long
    frame["eng_pm_prev3_minus_long"] = pm3 - pm_long
    frame["eng_pm_prev5_minus_long"] = pm5 - pm_long
    frame["eng_pm_prev1_minus_prev3"] = pm1 - pm3
    frame["eng_pm_prev3_minus_prev5"] = pm3 - pm5
    frame["eng_pm_prev1_minus_prev5"] = pm1 - pm5
    pm_stack = pd.concat([pm1, pm3, pm5], axis=1)
    frame["eng_pm_recent_mean_135"] = pm_stack.mean(axis=1, skipna=False)
    frame["eng_pm_recent_mean_minus_long"] = frame["eng_pm_recent_mean_135"] - pm_long
    frame["eng_pm_recent_range_135"] = pm_stack.max(axis=1, skipna=False) - pm_stack.min(axis=1, skipna=False)

    # Pitcher-vs-batter state gaps. These contain no ID lookup and are available at inference.
    frame["eng_long_pitcher_minus_batter_success"] = ps_long - bs_long
    frame["eng_prev1_pitcher_minus_batter_success"] = ps1 - bs_long
    frame["eng_prev3_pitcher_minus_batter_success"] = ps3 - bs_long
    frame["eng_prev5_pitcher_minus_batter_success"] = ps5 - bs_long
    frame["eng_recent_mean_pitcher_minus_batter_success"] = frame["eng_ps_recent_mean_135"] - bs_long
    frame["eng_long_pitcher_minus_batter_middle"] = pm_long - bm_long
    frame["eng_prev1_pitcher_minus_batter_middle"] = pm1 - bm_long
    frame["eng_prev3_pitcher_minus_batter_middle"] = pm3 - bm_long
    frame["eng_prev5_pitcher_minus_batter_middle"] = pm5 - bm_long

    # A bounded experience gate. The gate itself is monotonic in asof_pitcher_n;
    # the useful new information is primarily in the delta x gate products.
    n = numeric(frame, "asof_pitcher_n").clip(lower=0)
    stale_weight = n / (n + np.float32(2000.0))
    frame["eng_pitcher_stale_weight_2000"] = stale_weight
    for base_name in [
        "eng_ps_prev1_minus_long",
        "eng_ps_prev3_minus_long",
        "eng_ps_prev5_minus_long",
        "eng_ps_recent_mean_minus_long",
        "eng_ps_prev1_minus_prev5",
        "eng_pm_prev1_minus_long",
        "eng_pm_prev3_minus_long",
        "eng_pm_prev5_minus_long",
        "eng_pm_recent_mean_minus_long",
        "eng_pm_prev1_minus_prev5",
    ]:
        frame[f"{base_name}_x_stale2000"] = numeric(frame, base_name) * stale_weight


def feature_set(variant: str) -> list[str]:
    if variant not in VARIANT_GROUPS:
        raise ValueError(f"Unknown variant: {variant}")
    return list(CANONICAL_FEATURES) + list(VARIANT_GROUPS[variant])


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Screen inference-safe asof_* state features: recent-vs-long deltas, "
            "recent shape, pitcher-batter gaps, and experience-modulated deltas."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    try:
        import catboost
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    folds = parse_ints(args.folds)
    variants = list(VARIANT_GROUPS) if args.variants == "all" else parse_strings(args.variants)
    unknown = [v for v in variants if v not in VARIANT_GROUPS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    frame = load_frame(config).copy()
    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(
        raw_canonical
        + CANONICAL_SOURCE_COLUMNS
        + [
            target,
            season,
            row_id,
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
        ]
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    add_asof_state_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    output_dir = Path(config["paths"]["output_dir"]) / "asof_state_engineering"
    output_dir.mkdir(parents=True, exist_ok=True)
    sets = {v: feature_set(v) for v in variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)
    print(
        f"[ASOF State Engineering] folds={folds}, variants={variants}, iterations={args.iterations}, "
        f"task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print("[ASOF State Engineering] NO row sampling: every eligible temporal train/validation row is used.")
    print("[ASOF State Engineering] all engineered features use inference-available asof/context columns only.")

    rows: list[dict] = []
    for val_year in folds:
        train = frame.loc[frame[season] < val_year]
        valid = frame.loc[frame[season] == val_year]
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train or validation")

        y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
        y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
        print(
            f"\n[Fold {val_year}] train={len(train):,}, val={len(valid):,}, "
            f"train_rate={float(y_train.mean()):.6f}, val_rate={float(y_valid.mean()):.6f}"
        )

        baseline_brier = None
        for idx, variant in enumerate(variants, start=1):
            features = sets[variant]
            x_train, categorical = core.prepare_x(train, features)
            x_valid, _ = core.prepare_x(valid, features)
            train_pool = Pool(
                x_train,
                label=y_train,
                cat_features=categorical,
                feature_names=features,
            )
            valid_pool = Pool(
                x_valid,
                cat_features=categorical,
                feature_names=features,
            )
            model = CatBoostClassifier(**params)
            print(f"  [{idx:02d}/{len(variants):02d}] {variant:<24s} features={len(features):2d}", flush=True)
            model.fit(train_pool, verbose=args.verbose)
            prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
            metric = core.metrics(y_valid, prediction)
            if variant == "reference_canonical":
                baseline_brier = float(metric["brier"])
            rows.append(
                {
                    "validation_year": int(val_year),
                    "variant": variant,
                    "train_rows": int(len(train)),
                    "val_rows": int(len(valid)),
                    "feature_count": int(len(features)),
                    **metric,
                }
            )
            print(
                f"       brier={metric['brier']:.8f} score={metric['competition_score']:.2f} "
                f"auc={metric['auc']:.5f} p_std={metric['prediction_std']:.5f}"
            )
            del model, train_pool, valid_pool, x_train, x_valid, prediction
            gc.collect()

        if baseline_brier is None and "reference_canonical" not in variants:
            print("  note: reference_canonical not requested; deltas will be merged only where available")
        del y_train, y_valid
        gc.collect()

    results = pd.DataFrame(rows)
    reference = (
        results.loc[results["variant"].eq("reference_canonical"), ["validation_year", "brier"]]
        .rename(columns={"brier": "reference_brier"})
    )
    results = results.merge(reference, on="validation_year", how="left")
    results["delta_brier_vs_reference"] = results["brier"] - results["reference_brier"]
    results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        results.groupby("variant", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            feature_count=("feature_count", "first"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_delta_brier=("delta_brier_vs_reference", "mean"),
            worst_delta_brier=("delta_brier_vs_reference", "max"),
            best_delta_brier=("delta_brier_vs_reference", "min"),
            mean_score=("competition_score", "mean"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["mean_delta_brier", "worst_delta_brier"], na_position="last")
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "folds": folds,
            "variants": variants,
            "variant_groups": VARIANT_GROUPS,
            "iterations": int(args.iterations),
            "catboost_params": params,
            "canonical_invariants": invariant_check,
            "sampling": "none",
            "goal": "screen inference-safe state representations before broader Road-to-1500 model work",
        },
        output_dir / "run_config.json",
    )

    print("\n[ASOF State Summary: lower delta is better]")
    print(
        summary[
            [
                "variant",
                "feature_count",
                "mean_brier",
                "mean_delta_brier",
                "worst_delta_brier",
                "best_delta_brier",
                "mean_score",
                "mean_auc",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_brier": "{:.8f}".format,
                "mean_delta_brier": "{:+.8f}".format,
                "worst_delta_brier": "{:+.8f}".format,
                "best_delta_brier": "{:+.8f}".format,
                "mean_score": "{:.2f}".format,
                "mean_auc": "{:.5f}".format,
            },
        )
    )
    print("\n[Per-fold deltas]")
    print(
        results[
            ["validation_year", "variant", "brier", "competition_score", "auc", "delta_brier_vs_reference"]
        ].sort_values(["validation_year", "brier"]).to_string(
            index=False,
            formatters={
                "brier": "{:.8f}".format,
                "competition_score": "{:.2f}".format,
                "auc": "{:.5f}".format,
                "delta_brier_vs_reference": "{:+.8f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
