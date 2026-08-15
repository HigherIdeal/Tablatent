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
DROP_GAME_TYPE_FEATURES = [f for f in CANONICAL_FEATURES if f != "game_type"]
VARIANTS = {
    "raw_game_type": RAW_FEATURES,
    "drop_game_type": DROP_GAME_TYPE_FEATURES,
}


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_floats(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one decay is required")
    for value in values:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Decay must be in (0, 1], got {value}")
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


def make_recency_weights(
    train_season: pd.Series,
    latest_train_year: int,
    decay: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return mean-normalized exponential season weights.

    Raw season weight is decay ** age, where age=0 for the latest training season.
    Mean normalization keeps total weight comparable across decay settings, so the
    experiment changes relative recency emphasis without trivially changing the
    overall regularization scale.
    """
    years = pd.to_numeric(train_season, errors="raise").astype(int).to_numpy()
    age = latest_train_year - years
    if np.any(age < 0):
        raise ValueError("Training data contains a season newer than latest_train_year")

    raw = np.power(float(decay), age, dtype=np.float64)
    normalized = raw / raw.mean()

    audit_rows = []
    for year in sorted(np.unique(years)):
        mask = years == year
        audit_rows.append(
            {
                "season": int(year),
                "age": int(latest_train_year - year),
                "rows": int(mask.sum()),
                "raw_weight": float(raw[mask][0]),
                "normalized_weight": float(normalized[mask][0]),
            }
        )
    return normalized.astype(np.float32), pd.DataFrame(audit_rows)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    return float((w.sum() ** 2) / np.square(w).sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all-history CatBoost with exponential season recency weighting."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--decays", default="1.0,0.9,0.75,0.5")
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
    decays = parse_floats(args.decays)

    frame = load_frame(config).copy()
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    output_dir = Path(config["paths"]["output_dir"]) / "recency_weighting"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = catboost_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
    )

    print(
        f"[Recency Weighting] folds={folds}, decays={decays}, variants={list(VARIANTS)}, "
        f"iterations={args.iterations}, task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print("[Recency Weighting] all prior seasons are retained for every fold")
    print("[Recency Weighting] weight(season) = decay ** age; latest train season has raw weight 1")
    print("[Recency Weighting] weights are mean-normalized before CatBoost")
    print("[Recency Weighting] decay=1.0 is the exact no-recency-weight baseline")

    result_rows: list[dict] = []
    weight_rows: list[dict] = []

    for val_year in folds:
        train = frame.loc[frame[season] < val_year]
        valid = frame.loc[frame[season] == val_year]
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: missing train or validation rows")

        latest_train_year = int(train[season].max())
        y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
        y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float32)

        x_train_raw, raw_categorical = core.prepare_x(train, RAW_FEATURES)
        x_valid_raw, _ = core.prepare_x(valid, RAW_FEATURES)
        x_train_drop = x_train_raw.loc[:, DROP_GAME_TYPE_FEATURES]
        x_valid_drop = x_valid_raw.loc[:, DROP_GAME_TYPE_FEATURES]
        drop_categorical = [f for f in raw_categorical if f != "game_type"]

        prepared = {
            "raw_game_type": (x_train_raw, x_valid_raw, raw_categorical),
            "drop_game_type": (x_train_drop, x_valid_drop, drop_categorical),
        }

        print(
            f"\n[Fold {val_year}] train={len(train):,} ({int(train[season].min())}-{latest_train_year}), "
            f"val={len(valid):,}, train_rate={float(y_train.mean()):.6f}, val_rate={float(y_valid.mean()):.6f}"
        )

        for decay_index, decay in enumerate(decays, start=1):
            weights, audit = make_recency_weights(train[season], latest_train_year, decay)
            weighted_prior = float(np.average(y_train.astype(np.float64), weights=weights.astype(np.float64)))
            ess = effective_sample_size(weights)

            print(
                f"  [Decay {decay_index:02d}/{len(decays):02d}] decay={decay:.3f} "
                f"weighted_prior={weighted_prior:.6f} ess={ess:,.0f}/{len(train):,} "
                f"min_w={float(weights.min()):.4f} max_w={float(weights.max()):.4f}"
            )

            for _, audit_row in audit.iterrows():
                weight_rows.append(
                    {
                        "validation_year": int(val_year),
                        "decay": float(decay),
                        **audit_row.to_dict(),
                    }
                )

            for variant_index, (variant, features) in enumerate(VARIANTS.items(), start=1):
                x_train, x_valid, categorical = prepared[variant]
                train_pool = Pool(
                    x_train,
                    label=y_train,
                    weight=weights,
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

                result_rows.append(
                    {
                        "validation_year": int(val_year),
                        "decay": float(decay),
                        "variant": variant,
                        "feature_count": int(len(features)),
                        "train_start_year": int(train[season].min()),
                        "train_end_year": latest_train_year,
                        "train_rows": int(len(train)),
                        "val_rows": int(len(valid)),
                        "train_prior": float(y_train.mean()),
                        "weighted_prior": weighted_prior,
                        "effective_sample_size": ess,
                        "min_weight": float(weights.min()),
                        "max_weight": float(weights.max()),
                        **metric,
                    }
                )

                del model, train_pool, valid_pool, prediction
                gc.collect()

        del train, valid, y_train, y_valid
        del x_train_raw, x_valid_raw, x_train_drop, x_valid_drop
        gc.collect()

    results = pd.DataFrame(result_rows)
    baseline = (
        results.loc[np.isclose(results["decay"], 1.0), ["validation_year", "variant", "brier"]]
        .rename(columns={"brier": "baseline_brier"})
    )
    results = results.merge(baseline, on=["validation_year", "variant"], how="left")
    results["delta_vs_decay1"] = results["brier"] - results["baseline_brier"]

    summary = (
        results.groupby(["decay", "variant"], as_index=False)
        .agg(
            mean_brier=("brier", "mean"),
            mean_score=("competition_score", "mean"),
            mean_auc=("auc", "mean"),
            mean_p_std=("prediction_std", "mean"),
            mean_delta_vs_decay1=("delta_vs_decay1", "mean"),
            worst_delta_vs_decay1=("delta_vs_decay1", "max"),
        )
        .sort_values(["mean_brier", "decay", "variant"])
        .reset_index(drop=True)
    )

    best_by_fold = (
        results.sort_values(["validation_year", "brier"])
        .groupby("validation_year", as_index=False)
        .first()
    )

    results.to_csv(output_dir / "fold_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    best_by_fold.to_csv(output_dir / "best_by_fold.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output_dir / "weights_by_fold.csv", index=False)
    save_json(
        output_dir / "run_config.json",
        {
            "folds": folds,
            "decays": decays,
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices,
            "variants": {key: value for key, value in VARIANTS.items()},
            "weight_formula": "raw_weight = decay ** (latest_train_year - season); normalize mean to 1",
            "canonical_invariants": invariant_check,
            "catboost_params": params,
        },
    )

    print("\n[Recency Weighting summary] lower mean_brier is better")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_brier": "{:.6f}".format,
                "mean_score": "{:.2f}".format,
                "mean_auc": "{:.6f}".format,
                "mean_p_std": "{:.6f}".format,
                "mean_delta_vs_decay1": "{:+.6f}".format,
                "worst_delta_vs_decay1": "{:+.6f}".format,
            },
        )
    )

    print("\n[Best configuration by validation fold]")
    cols = [
        "validation_year",
        "decay",
        "variant",
        "brier",
        "competition_score",
        "auc",
        "prediction_std",
        "weighted_prior",
        "effective_sample_size",
        "delta_vs_decay1",
    ]
    print(best_by_fold[cols].to_string(index=False))
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
