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


VARIANTS = {
    "canonical": list(CANONICAL_FEATURES),
    "no_team_ids": [
        feature
        for feature in CANONICAL_FEATURES
        if feature not in {"pitcher_team_id", "batter_team_id"}
    ],
}


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


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


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, target: str, features: list[str], params: dict) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = core.prepare_x(train, features)
    x_valid, _ = core.prepare_x(valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)

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
    model.fit(train_pool, verbose=params.get("verbose", 0))
    prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)

    del model, train_pool, valid_pool, x_train, x_valid, y_train
    gc.collect()
    return prediction


def metric_row(
    protocol: str,
    variant: str,
    subset: str,
    y: np.ndarray,
    p: np.ndarray,
    rows: int,
    pitchers: int,
    train_rows: int,
    train_pitchers: int,
) -> dict:
    metric = core.metrics(y, p)
    return {
        "protocol": protocol,
        "variant": variant,
        "subset": subset,
        "rows": int(rows),
        "pitchers": int(pitchers),
        "train_rows": int(train_rows),
        "train_pitchers": int(train_pitchers),
        **metric,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test temporal CatBoost when validation pitcher IDs are unseen. "
            "The strict protocol removes every pitcher appearing in the validation year "
            "from all earlier training seasons."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame = load_frame(config).copy()
    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(raw_canonical + CANONICAL_SOURCE_COLUMNS + [target, season, row_id, "pitcher_id"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["_pitcher_key"] = normalize_id(frame["pitcher_id"])
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    train_full = frame.loc[frame[season] < args.validation_year].copy()
    valid = frame.loc[frame[season].eq(args.validation_year)].copy()
    if train_full.empty or valid.empty:
        raise ValueError("Training or validation partition is empty")

    train_pitcher_set = set(train_full["_pitcher_key"].unique())
    valid_pitcher_set = set(valid["_pitcher_key"].unique())
    natural_unseen_pitchers = valid_pitcher_set - train_pitcher_set
    seen_pitchers = valid_pitcher_set & train_pitcher_set

    natural_unseen_mask = valid["_pitcher_key"].isin(natural_unseen_pitchers).to_numpy()
    seen_mask = valid["_pitcher_key"].isin(seen_pitchers).to_numpy()

    # Strict deployment stress test: pretend every validation pitcher is unseen.
    # Remove all of their historical rows from every earlier season.
    train_purged = train_full.loc[~train_full["_pitcher_key"].isin(valid_pitcher_set)].copy()
    if train_purged.empty:
        raise ValueError("Purging validation pitchers removed all training rows")

    y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)

    output_dir = Path(config["paths"]["output_dir"]) / "unseen_pitcher_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Unseen-Pitcher Validation] validation={args.validation_year}, iterations={args.iterations}, "
        f"task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print(
        f"  train_full={len(train_full):,} rows / {train_full['_pitcher_key'].nunique():,} pitchers"
    )
    print(
        f"  validation={len(valid):,} rows / {valid['_pitcher_key'].nunique():,} pitchers"
    )
    print(
        f"  natural unseen={int(natural_unseen_mask.sum()):,} rows / {len(natural_unseen_pitchers):,} pitchers; "
        f"seen={int(seen_mask.sum()):,} rows / {len(seen_pitchers):,} pitchers"
    )
    print(
        f"  strict-purged train={len(train_purged):,} rows / {train_purged['_pitcher_key'].nunique():,} pitchers "
        f"(all {len(valid_pitcher_set):,} validation pitchers removed from history)"
    )

    rows: list[dict] = []

    for variant, features in VARIANTS.items():
        print(f"\n[Standard temporal] variant={variant} features={len(features)}")
        p_standard = fit_predict(train_full, valid, target, features, params)
        rows.append(
            metric_row(
                "standard_temporal",
                variant,
                "all_2024",
                y_valid,
                p_standard,
                len(valid),
                valid["_pitcher_key"].nunique(),
                len(train_full),
                train_full["_pitcher_key"].nunique(),
            )
        )
        if natural_unseen_mask.any():
            rows.append(
                metric_row(
                    "standard_temporal",
                    variant,
                    "natural_unseen_only",
                    y_valid[natural_unseen_mask],
                    p_standard[natural_unseen_mask],
                    int(natural_unseen_mask.sum()),
                    len(natural_unseen_pitchers),
                    len(train_full),
                    train_full["_pitcher_key"].nunique(),
                )
            )
        if seen_mask.any():
            rows.append(
                metric_row(
                    "standard_temporal",
                    variant,
                    "seen_only",
                    y_valid[seen_mask],
                    p_standard[seen_mask],
                    int(seen_mask.sum()),
                    len(seen_pitchers),
                    len(train_full),
                    train_full["_pitcher_key"].nunique(),
                )
            )

        print(f"[Strict unseen] variant={variant} features={len(features)}")
        p_purged = fit_predict(train_purged, valid, target, features, params)
        rows.append(
            metric_row(
                "strict_all_pitchers_unseen",
                variant,
                "all_2024",
                y_valid,
                p_purged,
                len(valid),
                valid["_pitcher_key"].nunique(),
                len(train_purged),
                train_purged["_pitcher_key"].nunique(),
            )
        )
        del p_standard, p_purged
        gc.collect()

    results = pd.DataFrame(rows)
    standard_ref = float(
        results.loc[
            (results["protocol"].eq("standard_temporal"))
            & (results["variant"].eq("canonical"))
            & (results["subset"].eq("all_2024")),
            "brier",
        ].iloc[0]
    )
    results["delta_brier_vs_standard_canonical"] = results["brier"] - standard_ref
    results = results.sort_values(["protocol", "subset", "brier", "variant"]).reset_index(drop=True)
    results.to_csv(output_dir / "results.csv", index=False)

    pitcher_summary = pd.DataFrame(
        {
            "validation_pitchers": [len(valid_pitcher_set)],
            "seen_pitchers": [len(seen_pitchers)],
            "natural_unseen_pitchers": [len(natural_unseen_pitchers)],
            "validation_rows": [len(valid)],
            "seen_rows": [int(seen_mask.sum())],
            "natural_unseen_rows": [int(natural_unseen_mask.sum())],
            "full_train_rows": [len(train_full)],
            "purged_train_rows": [len(train_purged)],
        }
    )
    pitcher_summary.to_csv(output_dir / "pitcher_coverage.csv", index=False)

    save_json(
        {
            "validation_year": int(args.validation_year),
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices,
            "variants": VARIANTS,
            "catboost_params": params,
            "canonical_invariants": invariant_check,
            "assumption": "strict_all_pitchers_unseen removes all earlier rows for every pitcher appearing in validation",
        },
        output_dir / "run_config.json",
    )

    print("\n[Results: lower Brier is better]")
    print(
        results[
            [
                "protocol",
                "variant",
                "subset",
                "rows",
                "pitchers",
                "train_rows",
                "brier",
                "competition_score",
                "auc",
                "prediction_std",
                "delta_brier_vs_standard_canonical",
            ]
        ].to_string(
            index=False,
            formatters={
                "brier": "{:.8f}".format,
                "competition_score": "{:.2f}".format,
                "auc": "{:.5f}".format,
                "prediction_std": "{:.5f}".format,
                "delta_brier_vs_standard_canonical": "{:+.8f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
