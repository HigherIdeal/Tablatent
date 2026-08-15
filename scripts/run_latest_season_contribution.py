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
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def catboost_params(
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
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


def select_train_years(
    available_years: list[int],
    validation_year: int,
    exclude_latest: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prior_years = [year for year in available_years if year < validation_year]
    if not prior_years:
        raise ValueError(f"Fold {validation_year}: no prior training seasons")
    if exclude_latest < 0:
        raise ValueError("exclude_latest must be >= 0")
    if exclude_latest >= len(prior_years):
        raise ValueError(
            f"Fold {validation_year}: exclude_latest={exclude_latest} removes all "
            f"{len(prior_years)} prior seasons"
        )

    if exclude_latest == 0:
        return tuple(prior_years), tuple()
    return tuple(prior_years[:-exclude_latest]), tuple(prior_years[-exclude_latest:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the most recent training season helps the next-year fold. "
            "For example, fold 2024 with exclude_latest=1 trains on 2019-2022 and "
            "directly measures the contribution of 2023."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument(
        "--exclude-latest",
        default="0,1",
        help="Comma-separated number of latest prior seasons to exclude. Default: 0,1",
    )
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

    folds = parse_ints(args.folds)
    exclusions = parse_ints(args.exclude_latest)
    if 0 not in exclusions:
        exclusions = [0, *exclusions]
    exclusions = list(dict.fromkeys(exclusions))
    if any(value < 0 for value in exclusions):
        raise ValueError("--exclude-latest values must be >= 0")

    frame = load_frame(config).copy()
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

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
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    all_years = sorted(int(year) for year in frame[season].unique())
    output_dir = Path(config["paths"]["output_dir"]) / "latest_season_contribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = catboost_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
    )

    print(
        f"[Latest-Season Contribution] folds={folds}, exclude_latest={exclusions}, "
        f"variants={list(VARIANTS)}, iterations={args.iterations}, "
        f"task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print("[Latest-Season Contribution] exclude_latest=0: use every prior season")
    print("[Latest-Season Contribution] exclude_latest=1: remove only the immediately previous season")
    print("[Latest-Season Contribution] all older seasons are retained; this is NOT a recent-window test")

    rows: list[dict] = []

    for val_year in folds:
        valid = frame.loc[frame[season] == val_year]
        if valid.empty:
            raise ValueError(f"Fold {val_year}: no validation rows")

        y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float32)
        x_valid_raw, raw_categorical = core.prepare_x(valid, RAW_FEATURES)
        x_valid_drop = x_valid_raw.loc[:, DROP_GAME_TYPE_FEATURES]
        drop_categorical = [feature for feature in raw_categorical if feature != "game_type"]
        prepared_valid = {
            "raw_game_type": (x_valid_raw, raw_categorical),
            "drop_game_type": (x_valid_drop, drop_categorical),
        }

        print(
            f"\n[Fold {val_year}] val={len(valid):,}, val_rate={float(y_valid.mean()):.6f}"
        )

        for exclusion_index, exclude_latest in enumerate(exclusions, start=1):
            train_years, removed_years = select_train_years(
                all_years,
                validation_year=val_year,
                exclude_latest=exclude_latest,
            )
            train = frame.loc[frame[season].isin(train_years)]
            y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)

            train_year_text = ",".join(str(year) for year in train_years)
            removed_year_text = ",".join(str(year) for year in removed_years) or "none"
            print(
                f"  [Train {exclusion_index:02d}/{len(exclusions):02d}] "
                f"exclude_latest={exclude_latest} years={train_years[0]}-{train_years[-1]} "
                f"rows={len(train):,} prior={float(y_train.mean()):.6f} "
                f"removed={removed_year_text}"
            )

            x_train_raw, _ = core.prepare_x(train, RAW_FEATURES)
            x_train_drop = x_train_raw.loc[:, DROP_GAME_TYPE_FEATURES]
            prepared_train = {
                "raw_game_type": x_train_raw,
                "drop_game_type": x_train_drop,
            }

            for variant_index, (variant, features) in enumerate(VARIANTS.items(), start=1):
                x_train = prepared_train[variant]
                x_valid, categorical = prepared_valid[variant]

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

                print(
                    f"      [{variant_index}/2] {variant:<15s} features={len(features):2d} "
                    f"brier={metric['brier']:.8f} score={metric['competition_score']:.2f} "
                    f"auc={metric['auc']:.5f} p_std={metric['prediction_std']:.5f}"
                )

                rows.append(
                    {
                        "validation_year": int(val_year),
                        "exclude_latest": int(exclude_latest),
                        "train_years": train_year_text,
                        "removed_years": removed_year_text,
                        "train_start_year": int(train_years[0]),
                        "train_end_year": int(train_years[-1]),
                        "train_seasons": int(len(train_years)),
                        "train_rows": int(len(train)),
                        "val_rows": int(len(valid)),
                        "train_prior": float(y_train.mean()),
                        "val_rate": float(y_valid.mean()),
                        "variant": variant,
                        "feature_count": int(len(features)),
                        **metric,
                    }
                )

                del model, train_pool, valid_pool, prediction
                gc.collect()

            del train, y_train, x_train_raw, x_train_drop
            gc.collect()

        del valid, y_valid, x_valid_raw, x_valid_drop
        gc.collect()

    results = pd.DataFrame(rows)

    baseline = (
        results.loc[
            results["exclude_latest"].eq(0),
            ["validation_year", "variant", "brier", "competition_score"],
        ]
        .rename(
            columns={
                "brier": "full_history_brier",
                "competition_score": "full_history_score",
            }
        )
    )
    results = results.merge(baseline, on=["validation_year", "variant"], how="left")
    results["delta_brier_vs_full_history"] = results["brier"] - results["full_history_brier"]
    results["delta_score_vs_full_history"] = (
        results["competition_score"] - results["full_history_score"]
    )

    summary = (
        results.groupby(["exclude_latest", "variant"], as_index=False)
        .agg(
            mean_brier=("brier", "mean"),
            mean_score=("competition_score", "mean"),
            mean_auc=("auc", "mean"),
            mean_p_std=("prediction_std", "mean"),
            mean_delta_brier_vs_full=("delta_brier_vs_full_history", "mean"),
            worst_delta_brier_vs_full=("delta_brier_vs_full_history", "max"),
        )
        .sort_values(["mean_brier", "exclude_latest", "variant"])
        .reset_index(drop=True)
    )

    latest_one = results.loc[results["exclude_latest"].eq(1)].copy()
    latest_one["latest_season_contribution_brier"] = latest_one["delta_brier_vs_full_history"]
    latest_one["latest_season_contribution_score"] = -latest_one["delta_score_vs_full_history"]
    contribution = latest_one[
        [
            "validation_year",
            "variant",
            "removed_years",
            "train_years",
            "brier",
            "full_history_brier",
            "latest_season_contribution_brier",
            "competition_score",
            "full_history_score",
            "latest_season_contribution_score",
            "auc",
            "prediction_std",
        ]
    ].sort_values(["validation_year", "variant"])

    results.to_csv(output_dir / "fold_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    contribution.to_csv(output_dir / "latest_season_contribution.csv", index=False)
    save_json(
        {
            "folds": folds,
            "exclude_latest": exclusions,
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices,
            "variants": VARIANTS,
            "canonical_invariants": invariant_check,
            "catboost_params": params,
            "interpretation": {
                "delta_brier_vs_full_history_positive": "removing latest season hurts; latest season was useful",
                "delta_brier_vs_full_history_negative": "removing latest season helps; latest season was harmful",
            },
        },
        output_dir / "run_config.json",
    )

    print("\n[Latest-Season Contribution summary] lower mean_brier is better")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_brier": "{:.6f}".format,
                "mean_score": "{:.2f}".format,
                "mean_auc": "{:.6f}".format,
                "mean_p_std": "{:.6f}".format,
                "mean_delta_brier_vs_full": "{:+.6f}".format,
                "worst_delta_brier_vs_full": "{:+.6f}".format,
            },
        )
    )

    print("\n[Effect of removing only the immediately previous season]")
    if contribution.empty:
        print("No exclude_latest=1 rows were requested.")
    else:
        display = contribution[
            [
                "validation_year",
                "variant",
                "removed_years",
                "train_years",
                "latest_season_contribution_brier",
                "latest_season_contribution_score",
            ]
        ]
        print(
            display.to_string(
                index=False,
                formatters={
                    "latest_season_contribution_brier": "{:+.8f}".format,
                    "latest_season_contribution_score": "{:+.2f}".format,
                },
            )
        )
        print("  positive Brier contribution: removing latest season made Brier worse -> latest season helped")
        print("  negative Brier contribution: removing latest season improved Brier -> latest season hurt")

    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
