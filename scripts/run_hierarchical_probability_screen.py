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


HP_SPECS = {
    "count": {
        "keys": ["balls_before", "strikes_before"],
        "parent": [],
    },
    "count_game_type": {
        "keys": ["balls_before", "strikes_before", "game_type"],
        "parent": ["balls_before", "strikes_before"],
    },
    "count_handedness": {
        "keys": ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"],
        "parent": ["balls_before", "strikes_before"],
    },
    "count_base": {
        "keys": ["balls_before", "strikes_before", "base_state"],
        "parent": ["balls_before", "strikes_before"],
    },
    "experience_count": {
        "keys": ["balls_before", "strikes_before", "eng_pitcher_experience_bucket"],
        "parent": ["balls_before", "strikes_before"],
    },
}

HP_COLUMNS = {name: f"hp_{name}_prob" for name in HP_SPECS}

VARIANT_GROUPS = {
    "reference_canonical": [],
    "add_success_state": list(asof_core.SUCCESS_STATE),
    "add_hp_count": [HP_COLUMNS["count"]],
    "add_hp_count_game_type": [HP_COLUMNS["count_game_type"]],
    "add_hp_count_handedness": [HP_COLUMNS["count_handedness"]],
    "add_hp_count_base": [HP_COLUMNS["count_base"]],
    "add_hp_experience_count": [HP_COLUMNS["experience_count"]],
    "add_hp_all": list(HP_COLUMNS.values()),
    "add_success_plus_hp_all": list(asof_core.SUCCESS_STATE) + list(HP_COLUMNS.values()),
}


def parse_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("at least one fold is required")
    return result


def parse_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def add_experience_bucket(frame: pd.DataFrame) -> None:
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce")
    # Generalizable state only; pitcher_id is never used.
    frame["eng_pitcher_experience_bucket"] = pd.cut(
        n,
        bins=[-np.inf, 100, 500, 1000, 2000, 5000, np.inf],
        labels=["0_100", "101_500", "501_1000", "1001_2000", "2001_5000", "5000_plus"],
        include_lowest=True,
    ).astype("string").fillna("<MISSING>").astype(str)


def aggregate_rate(source: pd.DataFrame, keys: list[str], target: str) -> pd.DataFrame:
    if not keys:
        raise ValueError("aggregate_rate requires at least one key")
    grouped = (
        source.groupby(keys, dropna=False, observed=True)[target]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "_sum", "count": "_count"})
    )
    return grouped


def lookup_parent_probability(
    source: pd.DataFrame,
    target_frame: pd.DataFrame,
    parent_keys: list[str],
    target: str,
    alpha: float,
) -> np.ndarray:
    global_prior = float(pd.to_numeric(source[target], errors="raise").mean())
    if not parent_keys:
        return np.full(len(target_frame), global_prior, dtype=np.float64)

    parent = aggregate_rate(source, parent_keys, target)
    parent["_parent_prob"] = (
        parent["_sum"].astype(np.float64) + float(alpha) * global_prior
    ) / (parent["_count"].astype(np.float64) + float(alpha))
    merged = target_frame[parent_keys].merge(
        parent[parent_keys + ["_parent_prob"]],
        on=parent_keys,
        how="left",
        sort=False,
    )
    return merged["_parent_prob"].fillna(global_prior).to_numpy(np.float64)


def apply_smoothed_probability(
    source: pd.DataFrame,
    target_frame: pd.DataFrame,
    keys: list[str],
    parent_keys: list[str],
    target: str,
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if source.empty:
        return (
            np.full(len(target_frame), 0.5, dtype=np.float32),
            {"exact_match_fraction": 0.0, "mean_support": 0.0, "median_support": 0.0},
        )

    global_prior = float(pd.to_numeric(source[target], errors="raise").mean())
    stats = aggregate_rate(source, keys, target)

    if parent_keys:
        parent = aggregate_rate(source, parent_keys, target)
        parent["_parent_prob"] = (
            parent["_sum"].astype(np.float64) + float(alpha) * global_prior
        ) / (parent["_count"].astype(np.float64) + float(alpha))
        stats = stats.merge(
            parent[parent_keys + ["_parent_prob"]],
            on=parent_keys,
            how="left",
            sort=False,
        )
        stats["_parent_prob"] = stats["_parent_prob"].fillna(global_prior)
    else:
        stats["_parent_prob"] = global_prior

    stats["_prob"] = (
        stats["_sum"].astype(np.float64) + float(alpha) * stats["_parent_prob"].astype(np.float64)
    ) / (stats["_count"].astype(np.float64) + float(alpha))

    fallback = lookup_parent_probability(
        source=source,
        target_frame=target_frame,
        parent_keys=parent_keys,
        target=target,
        alpha=alpha,
    )

    merged = target_frame[keys].merge(
        stats[keys + ["_prob", "_count"]],
        on=keys,
        how="left",
        sort=False,
    )
    exact = merged["_prob"].notna().to_numpy()
    # pandas 2.x/3.x may expose a read-only NumPy view here; we mutate missing rows below.
    probability = merged["_prob"].to_numpy(dtype=np.float64, copy=True)
    probability[~exact] = fallback[~exact]
    support = merged["_count"].fillna(0).to_numpy(np.float64)

    diagnostics = {
        "exact_match_fraction": float(exact.mean()) if len(exact) else 0.0,
        "mean_support": float(support.mean()) if len(support) else 0.0,
        "median_support": float(np.median(support)) if len(support) else 0.0,
    }
    return probability.astype(np.float32), diagnostics


def add_temporal_hierarchical_probabilities(
    frame: pd.DataFrame,
    season_col: str,
    target_col: str,
    alpha: float,
) -> pd.DataFrame:
    """Create season-expanding target statistics with no same-season labels.

    Rows in season Y are encoded using target labels from seasons < Y only.
    The earliest season has no history and receives neutral 0.5 values.
    """
    result = pd.DataFrame(index=frame.index)
    diagnostics: list[dict] = []
    years = sorted(int(y) for y in pd.unique(frame[season_col]))

    print(f"[Hierarchical features] years={years}, alpha={alpha:g}")
    for year in years:
        row_mask = frame[season_col].eq(year)
        history_mask = frame[season_col].lt(year)
        current = frame.loc[row_mask]
        history = frame.loc[history_mask]
        print(
            f"  year={year}: rows={len(current):,}, history_rows={len(history):,}",
            flush=True,
        )
        for name, spec in HP_SPECS.items():
            column = HP_COLUMNS[name]
            values, diag = apply_smoothed_probability(
                source=history,
                target_frame=current,
                keys=list(spec["keys"]),
                parent_keys=list(spec["parent"]),
                target=target_col,
                alpha=alpha,
            )
            result.loc[row_mask, column] = values
            diagnostics.append(
                {
                    "season": int(year),
                    "family": name,
                    "rows": int(len(current)),
                    "history_rows": int(len(history)),
                    **diag,
                }
            )

    for column in HP_COLUMNS.values():
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(np.float32)
    return result, pd.DataFrame(diagnostics)


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
            "Screen leakage-safe hierarchical conditional-probability features. "
            "Every season is encoded using target labels from older seasons only."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=200.0)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")

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
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
        ]
    )
    for spec in HP_SPECS.values():
        required.update(k for k in spec["keys"] if not k.startswith("eng_"))
        required.update(k for k in spec["parent"] if not k.startswith("eng_"))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    asof_core.add_asof_state_features(frame)
    add_experience_bucket(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    hp_frame, hp_diagnostics = add_temporal_hierarchical_probabilities(
        frame=frame,
        season_col=season,
        target_col=target,
        alpha=float(args.alpha),
    )
    for column in HP_COLUMNS.values():
        frame[column] = hp_frame[column].to_numpy(np.float32)
    del hp_frame
    gc.collect()

    output_dir = Path(config["paths"]["output_dir"]) / "hierarchical_probability_screen"
    output_dir.mkdir(parents=True, exist_ok=True)
    hp_diagnostics.to_csv(output_dir / "mapping_diagnostics.csv", index=False)

    sets = {v: feature_set(v) for v in variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)
    print(
        f"[Hierarchical Probability Screen] folds={folds}, variants={variants}, "
        f"iterations={args.iterations}, alpha={args.alpha:g}, task_type={args.task_type}, "
        f"catboost={catboost.__version__}"
    )
    print("[Hierarchical Probability Screen] NO row sampling.")
    print("[Hierarchical Probability Screen] training rows use ONLY older-season labels for target encodings.")
    print("[Hierarchical Probability Screen] pitcher_id/batter_id are NOT used in mappings or model features.")

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
            print(f"  [{idx:02d}/{len(variants):02d}] {variant:<27s} features={len(features):2d}", flush=True)
            model.fit(train_pool, verbose=args.verbose)
            prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
            metric = core.metrics(y_valid, prediction)
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

        del y_train, y_valid
        gc.collect()

    results = pd.DataFrame(rows)
    reference_variant = (
        results.loc[
            results["variant"].eq("reference_canonical"),
            ["validation_year", "brier"],
        ]
        .rename(columns={"brier": "reference_variant_brier"})
    )
    results = results.merge(reference_variant, on="validation_year", how="left")
    results["delta_brier_vs_reference"] = (
        results["brier"] - results["reference_variant_brier"]
    )
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
            "hierarchical_specs": HP_SPECS,
            "alpha": float(args.alpha),
            "iterations": int(args.iterations),
            "catboost_params": params,
            "canonical_invariants": invariant_check,
            "sampling": "none",
            "target_encoding_protocol": "row season Y uses labels from seasons < Y only; earliest season falls back to 0.5",
        },
        output_dir / "run_config.json",
    )

    print("\n[Hierarchical Probability Summary: lower delta is better]")
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
            [
                "validation_year",
                "variant",
                "brier",
                "competition_score",
                "auc",
                "delta_brier_vs_reference",
            ]
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
