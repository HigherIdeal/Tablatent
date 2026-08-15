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


RAW_FEATURES = list(CANONICAL_FEATURES)
DROP_GAME_TYPE_FEATURES = [feature for feature in CANONICAL_FEATURES if feature != "game_type"]
VARIANTS = {
    "raw_game_type": RAW_FEATURES,
    "drop_game_type": DROP_GAME_TYPE_FEATURES,
}


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_windows(value: str) -> list[str]:
    windows = [item.strip().lower() for item in value.split(",") if item.strip()]
    for window in windows:
        if window == "all":
            continue
        if not window.endswith("y") or not window[:-1].isdigit() or int(window[:-1]) <= 0:
            raise ValueError(f"Invalid window '{window}'. Use all,3y,2y,1y style values.")
    return windows


def window_train_years(all_years: list[int], val_year: int, window: str) -> tuple[int, ...]:
    prior_years = [year for year in all_years if year < val_year]
    if window == "all":
        selected = prior_years
    else:
        n_years = int(window[:-1])
        selected = prior_years[-n_years:]
    return tuple(selected)


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
        description="Compare recent training windows with raw vs removed game_type."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--windows", default="all,3y,2y,1y")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    try:
        import catboost
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError("catboost가 없습니다.") from exc

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))

    folds = parse_ints(args.folds)
    windows = parse_windows(args.windows)

    frame = load_frame(config).copy()
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    raw_canonical = [feature for feature in CANONICAL_FEATURES if feature != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    all_years = sorted(int(year) for year in frame[season].unique())
    output_dir = Path(config["paths"]["output_dir"]) / "training_window_game_type"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Training Window x Game-Type] folds={folds}, windows={windows}, "
        f"variants={list(VARIANTS)}, iterations={args.iterations}, "
        f"task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print("[Training Window x Game-Type] canonical de-duplication unchanged")
    print("[Training Window x Game-Type] identical season windows within a fold are trained once and reused")

    rows: list[dict] = []
    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)

    for val_year in folds:
        valid = frame.loc[frame[season] == val_year]
        if valid.empty:
            raise ValueError(f"Fold {val_year}: no validation rows")

        y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float32)
        x_valid_raw, raw_categorical = core.prepare_x(valid, RAW_FEATURES)
        x_valid_drop = x_valid_raw.loc[:, DROP_GAME_TYPE_FEATURES]
        drop_categorical = [feature for feature in raw_categorical if feature != "game_type"]

        window_years = {window: window_train_years(all_years, val_year, window) for window in windows}
        if any(not years for years in window_years.values()):
            raise ValueError(f"Fold {val_year}: at least one requested window has no training seasons")

        aliases_by_years: dict[tuple[int, ...], list[str]] = {}
        for window, years in window_years.items():
            aliases_by_years.setdefault(years, []).append(window)

        print(f"\n[Fold {val_year}] val={len(valid):,}, val_rate={float(y_valid.mean()):.6f}")

        for unique_index, (train_years, aliases) in enumerate(aliases_by_years.items(), start=1):
            train = frame.loc[frame[season].isin(train_years)]
            y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
            prior = float(y_train.mean())
            alias_text = "/".join(aliases)
            print(
                f"  [Window {unique_index:02d}/{len(aliases_by_years):02d}] {alias_text:<12s} "
                f"years={train_years[0]}-{train_years[-1]} rows={len(train):,} prior={prior:.6f}"
            )

            x_train_raw, _ = core.prepare_x(train, RAW_FEATURES)
            x_train_drop = x_train_raw.loc[:, DROP_GAME_TYPE_FEATURES]

            prepared = {
                "raw_game_type": (x_train_raw, x_valid_raw, raw_categorical),
                "drop_game_type": (x_train_drop, x_valid_drop, drop_categorical),
            }

            computed: dict[str, dict[str, float]] = {}
            for variant_index, variant in enumerate(VARIANTS, start=1):
                features = VARIANTS[variant]
                x_train, x_valid, categorical = prepared[variant]
                train_pool = Pool(
                    x_train,
                    label=y_train,
                    cat_features=categorical,
                    feature_names=features,
                )
                valid_pool = Pool(
                    x_valid,
                    label=y_valid,
                    cat_features=categorical,
                    feature_names=features,
                )

                model = CatBoostClassifier(**params)
                model.fit(train_pool, verbose=args.verbose)
                prediction = model.predict_proba(valid_pool)[:, 1]
                metric = core.metrics(y_valid, prediction)
                computed[variant] = metric

                print(
                    f"      [{variant_index}/2] {variant:<15s} features={len(features):2d} "
                    f"brier={metric['brier']:.8f} score={metric['competition_score']:.2f} "
                    f"auc={metric['auc']:.5f} p_std={metric['prediction_std']:.5f}"
                )

                del model, train_pool, valid_pool, prediction
                gc.collect()

            for alias in aliases:
                for variant, metric in computed.items():
                    rows.append(
                        {
                            "validation_year": int(val_year),
                            "window": alias,
                            "train_years": ",".join(str(year) for year in train_years),
                            "train_start_year": int(train_years[0]),
                            "train_end_year": int(train_years[-1]),
                            "train_seasons": int(len(train_years)),
                            "train_rows": int(len(train)),
                            "val_rows": int(len(valid)),
                            "train_prior": prior,
                            "variant": variant,
                            "feature_count": int(len(VARIANTS[variant])),
                            **metric,
                        }
                    )

            del train, y_train, x_train_raw, x_train_drop
            gc.collect()

        del valid, y_valid, x_valid_raw, x_valid_drop
        gc.collect()

    results = pd.DataFrame(rows)

    all_raw = (
        results.loc[(results["window"] == "all") & (results["variant"] == "raw_game_type"),
                    ["validation_year", "brier"]]
        .rename(columns={"brier": "all_raw_brier"})
    )
    results = results.merge(all_raw, on="validation_year", how="left")
    results["delta_vs_all_raw"] = results["brier"] - results["all_raw_brier"]

    raw_same_window = (
        results.loc[results["variant"] == "raw_game_type", ["validation_year", "window", "brier"]]
        .rename(columns={"brier": "same_window_raw_brier"})
    )
    results = results.merge(raw_same_window, on=["validation_year", "window"], how="left")
    results["delta_drop_vs_same_window_raw"] = np.where(
        results["variant"].eq("drop_game_type"),
        results["brier"] - results["same_window_raw_brier"],
        0.0,
    )

    results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        results.groupby(["window", "variant"], as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_score=("competition_score", "mean"),
            mean_auc=("auc", "mean"),
            mean_p_std=("prediction_std", "mean"),
            mean_delta_vs_all_raw=("delta_vs_all_raw", "mean"),
        )
        .sort_values(["mean_brier", "worst_brier"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    best_by_fold = (
        results.sort_values(["validation_year", "brier"])
        .groupby("validation_year", as_index=False)
        .first()[
            [
                "validation_year", "window", "variant", "train_years", "brier",
                "competition_score", "auc", "prediction_std", "delta_vs_all_raw",
            ]
        ]
    )
    best_by_fold.to_csv(output_dir / "best_by_fold.csv", index=False)

    run_config = {
        "folds": folds,
        "windows": windows,
        "variants": list(VARIANTS),
        "iterations": int(args.iterations),
        "canonical_features": CANONICAL_FEATURES,
        "invariant_check": invariant_check,
        "note": "Window aliases with identical train seasons are trained once and reused.",
    }
    save_json(run_config, output_dir / "run_config.json")

    print("\n[Training Window x Game-Type summary] lower mean_brier is better")
    print(
        summary[
            [
                "window", "variant", "mean_brier", "mean_score", "mean_auc",
                "mean_p_std", "mean_delta_vs_all_raw",
            ]
        ].to_string(index=False)
    )

    print("\n[Best configuration by validation fold]")
    print(best_by_fold.to_string(index=False))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
