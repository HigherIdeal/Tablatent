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
import run_unseen_pitcher_validation as unseen_core
from src.canonical_features import (
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.utils import load_config, save_json, seed_everything


def parse_ints(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number <= 0:
            raise ValueError("caps must be positive integers")
        result.append(number)
    if not result:
        raise ValueError("at least one cap is required")
    return sorted(set(result))


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


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    params: dict,
) -> np.ndarray:
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


def cap_per_pitcher(train: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    Keep only each pitcher's most recent `cap` rows.

    The caller must sort train chronologically first. The repository does not expose a
    full game timestamp, so chronology follows the existing project convention:
    season -> game_month -> row_id.
    """
    out = (
        train.groupby("_pitcher_key", sort=False, group_keys=False)
        .tail(int(cap))
        .sort_index()
        .copy()
    )
    if out.empty:
        raise ValueError(f"cap={cap} produced an empty training set")
    return out


def season_matched_random_control(
    train_full: pd.DataFrame,
    capped_train: pd.DataFrame,
    season_col: str,
    seed: int,
) -> pd.DataFrame:
    """Randomly retain the same number of rows per season as a capped training set."""
    rng = np.random.default_rng(seed)
    pieces = []
    for year, full_year in train_full.groupby(season_col, sort=True):
        n = int((capped_train[season_col] == year).sum())
        if n > len(full_year):
            raise AssertionError((year, n, len(full_year)))
        if n == len(full_year):
            pieces.append(full_year)
        elif n > 0:
            chosen = rng.choice(full_year.index.to_numpy(), size=n, replace=False)
            pieces.append(full_year.loc[chosen])
    out = pd.concat(pieces, axis=0).sort_index().copy()
    if len(out) != len(capped_train):
        raise AssertionError((len(out), len(capped_train)))
    return out


def training_summary(train: pd.DataFrame, target: str) -> dict:
    counts = train["_pitcher_key"].value_counts()
    return {
        "train_rows": int(len(train)),
        "train_pitchers": int(train["_pitcher_key"].nunique()),
        "train_target_rate": float(pd.to_numeric(train[target], errors="raise").mean()),
        "rows_per_pitcher_mean": float(counts.mean()),
        "rows_per_pitcher_median": float(counts.median()),
        "rows_per_pitcher_p90": float(counts.quantile(0.90)),
        "rows_per_pitcher_max": int(counts.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test per-pitcher training-history caps. All pre-validation seasons remain eligible, "
            "but each training pitcher contributes only their most recent N rows."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--caps", default="500,1000,2000,4000,8000")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--matched-controls",
        action="store_true",
        help=(
            "Also fit season-matched random-size controls for every cap. This separates the "
            "effect of keeping each pitcher's recent rows from the effect of simply using fewer rows."
        ),
    )
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    folds = core.parse_ints(args.folds)
    caps = parse_ints(args.caps)

    frame = load_frame(config).copy()
    raw_canonical = [f for f in CANONICAL_FEATURES if f != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(
        raw_canonical
        + CANONICAL_SOURCE_COLUMNS
        + [target, season, row_id, "pitcher_id"]
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["_pitcher_key"] = unseen_core.normalize_id(frame["pitcher_id"])

    # Existing project chronology convention. This is not a reconstructed exact game timestamp.
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)

    features = list(CANONICAL_FEATURES)
    params = catboost_params(config, args.iterations, args.task_type, args.devices, args.verbose)
    output_dir = Path(config["paths"]["output_dir"]) / "per_pitcher_history_cap"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Per-Pitcher History Cap] folds={folds}, caps={caps}, iterations={args.iterations}, "
        f"matched_controls={args.matched_controls}, task_type={args.task_type}, catboost={catboost.__version__}"
    )
    print(
        "[Per-Pitcher History Cap] ALL prior seasons remain eligible; only each pitcher's "
        "training rows are capped to the most recent N rows."
    )
    print(
        "[Per-Pitcher History Cap] pitcher_id is used ONLY to select training rows; "
        "it is NOT a model feature."
    )
    print(
        "[Per-Pitcher History Cap] note: retained rows still contain official asof_* values, "
        "which may summarize history older than the row cap."
    )

    rows: list[dict] = []
    train_summaries: list[dict] = []

    for val_year in folds:
        train_full = frame.loc[frame[season] < val_year].copy()
        valid = frame.loc[frame[season].eq(val_year)].copy()
        if train_full.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train or validation split")

        y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
        full_summary = training_summary(train_full, target)
        print(
            f"\n[Fold {val_year}] full_train={len(train_full):,}, val={len(valid):,}, "
            f"pitchers={train_full['_pitcher_key'].nunique():,}, target_rate={full_summary['train_target_rate']:.6f}"
        )

        print("  [baseline] all_history", flush=True)
        p_full = fit_predict(train_full, valid, target, features, params)
        m_full = core.metrics(y_valid, p_full)
        rows.append(
            {
                "validation_year": int(val_year),
                "protocol": "all_history",
                "cap": np.nan,
                "retained_fraction": 1.0,
                **full_summary,
                **m_full,
            }
        )
        train_summaries.append(
            {"validation_year": int(val_year), "protocol": "all_history", "cap": np.nan, **full_summary}
        )
        print(
            f"       rows={len(train_full):,} brier={m_full['brier']:.8f} "
            f"score={m_full['competition_score']:.2f} auc={m_full['auc']:.5f}"
        )
        del p_full
        gc.collect()

        for cap in caps:
            capped = cap_per_pitcher(train_full, cap)
            summary = training_summary(capped, target)
            retained_fraction = float(len(capped) / len(train_full))
            print(
                f"  [cap={cap:5d}] recent_per_pitcher rows={len(capped):,} "
                f"({retained_fraction:.1%})",
                flush=True,
            )
            prediction = fit_predict(capped, valid, target, features, params)
            metric = core.metrics(y_valid, prediction)
            rows.append(
                {
                    "validation_year": int(val_year),
                    "protocol": "recent_per_pitcher",
                    "cap": int(cap),
                    "retained_fraction": retained_fraction,
                    **summary,
                    **metric,
                }
            )
            train_summaries.append(
                {
                    "validation_year": int(val_year),
                    "protocol": "recent_per_pitcher",
                    "cap": int(cap),
                    **summary,
                }
            )
            print(
                f"       brier={metric['brier']:.8f} score={metric['competition_score']:.2f} "
                f"auc={metric['auc']:.5f} p_std={metric['prediction_std']:.5f}"
            )
            del prediction

            if args.matched_controls:
                control = season_matched_random_control(
                    train_full=train_full,
                    capped_train=capped,
                    season_col=season,
                    seed=seed + val_year * 100_000 + cap,
                )
                control_summary = training_summary(control, target)
                print(
                    f"             matched_random rows={len(control):,}",
                    flush=True,
                )
                prediction_control = fit_predict(control, valid, target, features, params)
                metric_control = core.metrics(y_valid, prediction_control)
                rows.append(
                    {
                        "validation_year": int(val_year),
                        "protocol": "season_matched_random",
                        "cap": int(cap),
                        "retained_fraction": retained_fraction,
                        **control_summary,
                        **metric_control,
                    }
                )
                train_summaries.append(
                    {
                        "validation_year": int(val_year),
                        "protocol": "season_matched_random",
                        "cap": int(cap),
                        **control_summary,
                    }
                )
                print(
                    f"       control brier={metric_control['brier']:.8f} "
                    f"score={metric_control['competition_score']:.2f} auc={metric_control['auc']:.5f}"
                )
                del prediction_control, control

            del capped
            gc.collect()

        del train_full, valid, y_valid
        gc.collect()

    results = pd.DataFrame(rows)
    baseline = (
        results.loc[results["protocol"].eq("all_history"), ["validation_year", "brier"]]
        .rename(columns={"brier": "all_history_brier"})
    )
    results = results.merge(baseline, on="validation_year", how="left")
    results["delta_brier_vs_all_history"] = results["brier"] - results["all_history_brier"]

    if args.matched_controls:
        controls = (
            results.loc[
                results["protocol"].eq("season_matched_random"),
                ["validation_year", "cap", "brier"],
            ]
            .rename(columns={"brier": "matched_random_brier"})
        )
        results = results.merge(controls, on=["validation_year", "cap"], how="left")
        results["delta_brier_vs_matched_random"] = results["brier"] - results["matched_random_brier"]

    results.to_csv(output_dir / "fold_results.csv", index=False)
    pd.DataFrame(train_summaries).to_csv(output_dir / "training_set_summary.csv", index=False)

    cap_only = results.loc[results["protocol"].eq("recent_per_pitcher")].copy()
    summary = (
        cap_only.groupby("cap", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_train_rows=("train_rows", "mean"),
            mean_retained_fraction=("retained_fraction", "mean"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_delta_brier=("delta_brier_vs_all_history", "mean"),
            worst_delta_brier=("delta_brier_vs_all_history", "max"),
            mean_score=("competition_score", "mean"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["mean_delta_brier", "worst_delta_brier"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "folds": folds,
            "caps": caps,
            "iterations": int(args.iterations),
            "matched_controls": bool(args.matched_controls),
            "features": features,
            "catboost_params": params,
            "canonical_invariants": invariant_check,
            "chronology": "season -> game_month -> row_id",
            "important_note": (
                "The cap changes which training rows are retained. Official asof_* values inside "
                "retained rows are not recomputed and may summarize older pitcher history."
            ),
        },
        output_dir / "run_config.json",
    )

    print("\n[Per-Pitcher Cap Summary: lower delta is better]")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_retained_fraction": "{:.3f}".format,
                "mean_brier": "{:.8f}".format,
                "worst_brier": "{:.8f}".format,
                "mean_delta_brier": "{:+.8f}".format,
                "worst_delta_brier": "{:+.8f}".format,
                "mean_score": "{:.2f}".format,
                "mean_auc": "{:.5f}".format,
            },
        )
    )

    print("\n[Per-fold results]")
    display = results[
        [
            "validation_year",
            "protocol",
            "cap",
            "train_rows",
            "retained_fraction",
            "brier",
            "competition_score",
            "auc",
            "prediction_std",
            "delta_brier_vs_all_history",
        ]
        + (["delta_brier_vs_matched_random"] if args.matched_controls else [])
    ].copy()
    print(
        display.to_string(
            index=False,
            formatters={
                "retained_fraction": "{:.3f}".format,
                "brier": "{:.8f}".format,
                "competition_score": "{:.2f}".format,
                "auc": "{:.5f}".format,
                "prediction_std": "{:.5f}".format,
                "delta_brier_vs_all_history": "{:+.8f}".format,
                **(
                    {"delta_brier_vs_matched_random": "{:+.8f}".format}
                    if args.matched_controls
                    else {}
                ),
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
