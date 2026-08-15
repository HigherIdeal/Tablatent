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


PREV1_EFFECT = "game_type_prev1_effect"
PREV12_EWMA_EFFECT = "game_type_prev12_ewma_effect"
BASE_NO_GAME_TYPE = [feature for feature in CANONICAL_FEATURES if feature != "game_type"]

VARIANT_FEATURES = {
    "raw_game_type": list(CANONICAL_FEATURES),
    "drop_game_type": list(BASE_NO_GAME_TYPE),
    "prev1_effect": [*BASE_NO_GAME_TYPE, PREV1_EFFECT],
    "prev12_ewma_effect": [*BASE_NO_GAME_TYPE, PREV12_EWMA_EFFECT],
}


def _season_effect_map(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    game_type_col: str,
    source_year: int,
) -> pd.Series:
    source = frame.loc[frame[season_col] == source_year]
    if source.empty:
        return pd.Series(dtype=np.float64)
    global_rate = float(source[target_col].mean())
    return source.groupby(game_type_col, observed=True)[target_col].mean() - global_rate


def add_temporal_game_type_features(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    game_type_col: str = "game_type",
) -> dict[str, dict[str, float | int | None]]:
    """Add leakage-safe season-lagged game_type effects in-place.

    For season t, only labels from seasons before t are used.
    prev1 uses the immediately previous observed season.
    prev12_ewma uses 2/3 * prev1 + 1/3 * prev2 when both exist.
    """
    frame[PREV1_EFFECT] = np.float32(0.0)
    frame[PREV12_EWMA_EFFECT] = np.float32(0.0)

    years = sorted(int(year) for year in frame[season_col].dropna().unique())
    effect_maps = {
        year: _season_effect_map(frame, season_col, target_col, game_type_col, year)
        for year in years
    }
    audit: dict[str, dict[str, float | int | None]] = {}

    for index, year in enumerate(years):
        mask = frame[season_col] == year
        current_types = frame.loc[mask, game_type_col]
        prev1_year = years[index - 1] if index >= 1 else None
        prev2_year = years[index - 2] if index >= 2 else None

        if prev1_year is None:
            prev1_values = np.zeros(int(mask.sum()), dtype=np.float64)
        else:
            prev1_values = (
                current_types.map(effect_maps[prev1_year]).fillna(0.0).to_numpy(np.float64)
            )

        if prev2_year is None:
            ewma_values = prev1_values.copy()
        else:
            prev2_values = (
                current_types.map(effect_maps[prev2_year]).fillna(0.0).to_numpy(np.float64)
            )
            ewma_values = (2.0 / 3.0) * prev1_values + (1.0 / 3.0) * prev2_values

        frame.loc[mask, PREV1_EFFECT] = prev1_values.astype(np.float32)
        frame.loc[mask, PREV12_EWMA_EFFECT] = ewma_values.astype(np.float32)

        audit[str(year)] = {
            "prev1_year": prev1_year,
            "prev2_year": prev2_year,
            "prev1_mean": float(prev1_values.mean()) if len(prev1_values) else 0.0,
            "prev1_std": float(prev1_values.std()) if len(prev1_values) else 0.0,
            "ewma_mean": float(ewma_values.mean()) if len(ewma_values) else 0.0,
            "ewma_std": float(ewma_values.std()) if len(ewma_values) else 0.0,
        }

    return audit


def run_experiment(
    config: dict,
    folds: list[int],
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> None:
    try:
        import catboost
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise RuntimeError("catboost가 없습니다.") from exc

    seed_everything(int(config["seed"]))
    frame = load_frame(config).copy()
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    validate_canonical_schema(frame)
    add_canonical_derived_features(frame)

    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame[target] = pd.to_numeric(frame[target], errors="raise").astype(np.float64)
    frame["game_type"] = frame["game_type"].astype("string").fillna("<MISSING>").astype(str)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    temporal_audit = add_temporal_game_type_features(frame, season, target)

    output_dir = Path(config["paths"]["output_dir"]) / "game_type_temporal_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_sets.json").write_text(
        json.dumps(VARIANT_FEATURES, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[Game-Type Temporal] folds={folds}, variants={len(VARIANT_FEATURES)}, "
        f"iterations={iterations}, task_type={task_type}, catboost={catboost.__version__}"
    )
    print("[Game-Type Temporal] A raw_game_type")
    print("[Game-Type Temporal] B drop_game_type")
    print("[Game-Type Temporal] C prev1_effect: previous-season game_type effect")
    print("[Game-Type Temporal] D prev12_ewma_effect: 2/3 prev1 + 1/3 prev2")
    print("[Game-Type Temporal] temporal features are strict season-lagged; no current-season labels")

    rows: list[dict] = []
    for val_year in folds:
        fold = frame.loc[frame[season] <= val_year]
        train = fold.loc[fold[season] < val_year]
        valid = fold.loc[fold[season] == val_year]
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train or validation split")

        y_train = train[target].to_numpy(np.float32)
        y_valid = valid[target].to_numpy(np.float32)
        print(
            f"\n[Fold {val_year}] train={len(train):,}, val={len(valid):,}, "
            f"prior={float(y_train.mean()):.6f}"
        )

        for index, (variant, features) in enumerate(VARIANT_FEATURES.items(), start=1):
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
                label=y_valid,
                cat_features=categorical,
                feature_names=features,
            )

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

            print(
                f"  [{index:02d}/{len(VARIANT_FEATURES):02d}] {variant:<22s} "
                f"features={len(features):2d}",
                flush=True,
            )
            model = CatBoostClassifier(**params)
            model.fit(train_pool, verbose=verbose)
            prediction = model.predict_proba(valid_pool)[:, 1]
            metric = core.metrics(y_valid, prediction)
            rows.append(
                {
                    "variant": variant,
                    "validation_year": int(val_year),
                    "train_rows": int(len(train)),
                    "val_rows": int(len(valid)),
                    "feature_count": int(len(features)),
                    **metric,
                }
            )
            print(
                f"       brier={metric['brier']:.8f} "
                f"score={metric['competition_score']:.2f} "
                f"auc={metric['auc']:.5f} "
                f"p_std={metric['prediction_std']:.5f}"
            )

            del model, train_pool, valid_pool, x_train, x_valid, prediction
            gc.collect()

    fold_results = pd.DataFrame(rows)
    raw_reference = (
        fold_results.loc[
            fold_results["variant"] == "raw_game_type",
            ["validation_year", "brier"],
        ]
        .rename(columns={"brier": "raw_game_type_brier"})
    )
    drop_reference = (
        fold_results.loc[
            fold_results["variant"] == "drop_game_type",
            ["validation_year", "brier"],
        ]
        .rename(columns={"brier": "drop_game_type_brier"})
    )
    fold_results = fold_results.merge(raw_reference, on="validation_year", how="left")
    fold_results = fold_results.merge(drop_reference, on="validation_year", how="left")
    fold_results["delta_brier_vs_raw"] = fold_results["brier"] - fold_results["raw_game_type_brier"]
    fold_results["delta_brier_vs_drop"] = fold_results["brier"] - fold_results["drop_game_type_brier"]
    fold_results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        fold_results.groupby("variant", as_index=False)
        .agg(
            feature_count=("feature_count", "first"),
            mean_brier=("brier", "mean"),
            mean_competition_score=("competition_score", "mean"),
            mean_delta_vs_raw=("delta_brier_vs_raw", "mean"),
            worst_delta_vs_raw=("delta_brier_vs_raw", "max"),
            mean_delta_vs_drop=("delta_brier_vs_drop", "mean"),
            mean_auc=("auc", "mean"),
            mean_prediction_std=("prediction_std", "mean"),
        )
        .sort_values(["mean_brier", "worst_delta_vs_raw"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "folds": folds,
            "iterations": int(iterations),
            "variants": VARIANT_FEATURES,
            "temporal_feature_policy": {
                PREV1_EFFECT: "previous observed season game_type success rate minus that season global success rate",
                PREV12_EWMA_EFFECT: "2/3 previous-season effect + 1/3 two-seasons-back effect",
            },
            "temporal_audit": temporal_audit,
            "output_dir": str(output_dir),
        },
        output_dir / "run_config.json",
    )

    print("\n[Game-Type Temporal summary] lower Brier is better")
    print(
        summary[
            [
                "variant",
                "feature_count",
                "mean_brier",
                "mean_competition_score",
                "mean_delta_vs_raw",
                "worst_delta_vs_raw",
                "mean_delta_vs_drop",
                "mean_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {output_dir}")


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw game_type against leakage-safe season-lagged game_type effects."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    run_experiment(
        config=config,
        folds=parse_ints(args.folds),
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
