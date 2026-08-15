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

import run_asof_state_engineering as asof_core
from src.canonical_features import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


INTERACTION_SPECS: dict[str, list[str]] = {
    "ctx_count_state": ["balls_before", "strikes_before"],
    "ctx_hand_matchup": ["pitcher_hand", "batter_hand"],
    "ctx_count_hand": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
    "ctx_count_base": ["balls_before", "strikes_before", "base_state"],
    "ctx_count_outs": ["balls_before", "strikes_before", "outs_before"],
    "ctx_count_pressure": ["balls_before", "strikes_before", "base_state", "outs_before"],
}

INTERACTION_COLUMNS = list(INTERACTION_SPECS)
ENGINEERED_CATEGORICAL = set(INTERACTION_COLUMNS)

VARIANT_GROUPS: dict[str, list[str]] = {
    "reference_canonical": [],
    "success_state": list(asof_core.SUCCESS_STATE),
    "success_plus_count_state": list(asof_core.SUCCESS_STATE) + ["ctx_count_state"],
    "success_plus_hand_matchup": list(asof_core.SUCCESS_STATE) + ["ctx_hand_matchup"],
    "success_plus_count_hand": list(asof_core.SUCCESS_STATE) + ["ctx_count_hand"],
    "success_plus_count_base": list(asof_core.SUCCESS_STATE) + ["ctx_count_base"],
    "success_plus_count_outs": list(asof_core.SUCCESS_STATE) + ["ctx_count_outs"],
    "success_plus_context_all": list(asof_core.SUCCESS_STATE) + INTERACTION_COLUMNS,
}


def parse_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("at least one fold is required")
    return result


def parse_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _tokenize(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def add_context_interactions(frame: pd.DataFrame) -> None:
    """Add inference-safe categorical context crosses in-place.

    Only pre-pitch context is used. Target labels, pitcher/batter IDs, and
    game_type crosses are deliberately excluded.
    """
    required = {source for sources in INTERACTION_SPECS.values() for source in sources}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing context-interaction source columns: {missing}")

    token_cache = {column: _tokenize(frame[column]) for column in required}
    for output, sources in INTERACTION_SPECS.items():
        value = token_cache[sources[0]].copy()
        for source in sources[1:]:
            value = value.str.cat(token_cache[source], sep="|")
        frame[output] = value


def feature_set(variant: str) -> list[str]:
    if variant not in VARIANT_GROUPS:
        raise ValueError(f"Unknown variant: {variant}")
    features = list(CANONICAL_FEATURES) + list(VARIANT_GROUPS[variant])
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate features in variant {variant}")
    return features


def prepare_x(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Prepare CatBoost input and keep engineered crosses categorical."""
    x = frame.loc[:, features].copy()
    categorical_set = set(CANONICAL_CATEGORICAL) | ENGINEERED_CATEGORICAL
    categorical = [feature for feature in features if feature in categorical_set]
    cat_lookup = set(categorical)
    for column in features:
        if column in cat_lookup:
            x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype(np.float32)
            x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x, categorical


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Screen explicit inference-safe context crosses on top of the retained "
            "success-state features. No target encoding and no row sampling."
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
    unknown = [variant for variant in variants if variant not in VARIANT_GROUPS]
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
    required.update(source for sources in INTERACTION_SPECS.values() for source in sources)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    asof_core.add_asof_state_features(frame)
    add_context_interactions(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    output_dir = Path(config["paths"]["output_dir"]) / "context_interaction_screen"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = {variant: feature_set(variant) for variant in variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cardinalities = {
        column: int(frame[column].nunique(dropna=False)) for column in INTERACTION_COLUMNS
    }
    save_json(cardinalities, output_dir / "interaction_cardinalities.json")

    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)
    print(
        f"[Context Interaction Screen] folds={folds}, variants={variants}, "
        f"iterations={args.iterations}, task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print("[Context Interaction Screen] NO row sampling; all eligible temporal rows are used.")
    print("[Context Interaction Screen] target labels and pitcher/batter IDs are NOT used in crosses.")
    print("[Context Interaction Screen] game_type crosses are deliberately excluded due to regime sensitivity.")
    print(f"[Context Interaction Screen] cardinalities={cardinalities}")

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

        for index, variant in enumerate(variants, start=1):
            features = feature_sets[variant]
            x_train, categorical = prepare_x(train, features)
            x_valid, _ = prepare_x(valid, features)
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
            print(
                f"  [{index:02d}/{len(variants):02d}] {variant:<28s} "
                f"features={len(features):2d} cat={len(categorical):2d}",
                flush=True,
            )
            model.fit(train_pool, verbose=args.verbose)
            prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
            metric = probability_metrics(y_valid, prediction)
            rows.append(
                {
                    "validation_year": int(val_year),
                    "variant": variant,
                    "train_rows": int(len(train)),
                    "val_rows": int(len(valid)),
                    "feature_count": int(len(features)),
                    "categorical_count": int(len(categorical)),
                    **metric,
                }
            )
            print(
                f"       brier={metric['brier']:.8f} "
                f"raw_score={metric['raw_score']:+.2f} clipped={metric['clipped_score']:.2f} "
                f"auc={metric['auc']:.5f} p_std={metric['prediction_std']:.5f}"
            )
            del model, train_pool, valid_pool, x_train, x_valid, prediction
            gc.collect()

        del y_train, y_valid
        gc.collect()

    results = pd.DataFrame(rows)
    reference = (
        results.loc[
            results["variant"].eq("reference_canonical"),
            ["validation_year", "brier"],
        ]
        .rename(columns={"brier": "reference_variant_brier"})
    )
    results = results.merge(reference, on="validation_year", how="left")
    results["delta_brier_vs_reference"] = results["brier"] - results["reference_variant_brier"]

    success_reference = (
        results.loc[
            results["variant"].eq("success_state"),
            ["validation_year", "brier"],
        ]
        .rename(columns={"brier": "success_state_brier"})
    )
    results = results.merge(success_reference, on="validation_year", how="left")
    results["delta_brier_vs_success_state"] = results["brier"] - results["success_state_brier"]
    results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        results.groupby("variant", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            feature_count=("feature_count", "first"),
            categorical_count=("categorical_count", "first"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_delta_vs_reference=("delta_brier_vs_reference", "mean"),
            worst_delta_vs_reference=("delta_brier_vs_reference", "max"),
            mean_delta_vs_success=("delta_brier_vs_success_state", "mean"),
            worst_delta_vs_success=("delta_brier_vs_success_state", "max"),
            best_delta_vs_success=("delta_brier_vs_success_state", "min"),
            mean_raw_score=("raw_score", "mean"),
            worst_raw_score=("raw_score", "min"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["mean_delta_vs_success", "worst_delta_vs_success"], na_position="last")
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "folds": folds,
            "variants": variants,
            "variant_groups": VARIANT_GROUPS,
            "interaction_specs": INTERACTION_SPECS,
            "interaction_cardinalities": cardinalities,
            "iterations": int(args.iterations),
            "catboost_params": params,
            "canonical_invariants": invariant_check,
            "sampling": "none",
            "game_type_crosses": "excluded",
            "goal": "test whether explicit pre-pitch context combinations add discrimination beyond success-state deltas",
        },
        output_dir / "run_config.json",
    )

    print("\n[Context Interaction Summary: lower delta is better]")
    print(
        summary[
            [
                "variant",
                "feature_count",
                "mean_brier",
                "mean_delta_vs_reference",
                "mean_delta_vs_success",
                "worst_delta_vs_success",
                "best_delta_vs_success",
                "mean_raw_score",
                "mean_auc",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_brier": "{:.8f}".format,
                "mean_delta_vs_reference": "{:+.8f}".format,
                "mean_delta_vs_success": "{:+.8f}".format,
                "worst_delta_vs_success": "{:+.8f}".format,
                "best_delta_vs_success": "{:+.8f}".format,
                "mean_raw_score": "{:+.2f}".format,
                "mean_auc": "{:.5f}".format,
            },
        )
    )

    print("\n[Per-fold deltas vs success_state]")
    print(
        results[
            [
                "validation_year",
                "variant",
                "brier",
                "raw_score",
                "auc",
                "delta_brier_vs_reference",
                "delta_brier_vs_success_state",
            ]
        ].sort_values(["validation_year", "brier"]).to_string(
            index=False,
            formatters={
                "brier": "{:.8f}".format,
                "raw_score": "{:+.2f}".format,
                "auc": "{:.5f}".format,
                "delta_brier_vs_reference": "{:+.8f}".format,
                "delta_brier_vs_success_state": "{:+.8f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
