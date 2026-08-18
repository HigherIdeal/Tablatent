from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


VARIANTS = (
    "A0_FULL",
    "A1_DROP_SEASON",
    "A2_ADD_RECENT_F",
    "A3_ADD_GT_REGIME",
)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    metric = probability_metrics(y, p)
    loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    return {
        "score": float(metric["raw_score"]),
        "brier": float(metric["brier"]),
        "loss": loss,
    }


def metric_line(
    name: str,
    metric: dict[str, float],
    delta_brier: float | None = None,
) -> str:
    text = (
        f"{name:<25s} "
        f"score={metric['score']:+9.2f}  "
        f"brier={metric['brier']:.8f}  "
        f"loss={metric['loss']:.8f}"
    )
    if delta_brier is not None:
        text += f"  dB={delta_brier:+.8f}"
    return text


def build_params(
    *,
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    gpu_ram_part: float,
    pinned_memory_size: str,
) -> dict:
    params = context_core.catboost_params(
        config=config,
        iterations=iterations,
        task_type=task_type,
        devices=devices,
        verbose=0,
    )
    params["thread_count"] = -1
    params["metric_period"] = max(50, int(iterations))
    if task_type == "GPU":
        params["gpu_ram_part"] = float(gpu_ram_part)
        params["pinned_memory_size"] = str(pinned_memory_size)
        params["gpu_cat_features_storage"] = "GpuRam"
    return params


def add_regime_features(
    frame: pd.DataFrame,
    *,
    season_col: str,
    regime_start_year: int,
) -> None:
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    game_type = frame["game_type"].astype("string").str.strip().str.upper()
    recent = season.ge(int(regime_start_year))

    # Binary feature aimed specifically at the observed F-domain break.
    frame["eng_recent_f"] = (recent & game_type.eq("F")).astype(np.float32)

    # Explicit interaction. This makes the tree see R_old/R_recent/F_old/F_recent
    # directly rather than discovering the season x game_type cross by itself.
    frame["eng_gt_regime"] = (
        game_type.fillna("<MISSING>").astype(str)
        + "_"
        + np.where(recent, "RECENT", "OLD")
    )


def prepare_x(
    frame: pd.DataFrame,
    features: list[str],
    *,
    extra_categorical: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    extra_categorical = extra_categorical or set()
    categorical_set = set(CANONICAL_CATEGORICAL) | set(extra_categorical)
    categorical = [feature for feature in features if feature in categorical_set]
    cat_lookup = set(categorical)

    x = frame.loc[:, features].copy()
    for column in features:
        if column in cat_lookup:
            x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype(np.float32)
            x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x, categorical


def fit_predict(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    features: list[str],
    extra_categorical: set[str],
    params: dict,
) -> np.ndarray:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    x_train, categorical = prepare_x(
        train,
        features,
        extra_categorical=extra_categorical,
    )
    x_valid, valid_categorical = prepare_x(
        valid,
        features,
        extra_categorical=extra_categorical,
    )
    if categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")

    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float32)

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

    params = dict(params)
    regression = bool(params.pop("_regression", False))
    if regression:
        params["loss_function"] = "RMSE"
        params.pop("eval_metric", None)
    model = (CatBoostRegressor if regression else CatBoostClassifier)(**params)
    model.fit(train_pool, verbose=False)
    pred = np.asarray(model.predict(valid_pool) if regression else model.predict_proba(valid_pool)[:, 1], dtype=np.float64)
    pred = np.clip(pred, 0.0, 1.0)

    del model, train_pool, valid_pool, x_train, x_valid, y_train, y_valid
    gc.collect()
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether explicit temporal regime features improve the pooled R/F "
            "CatBoost beyond season and game_type alone. Train 2019-2023 and validate 2024."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument(
        "--output-dir",
        default="outputs/game_type_temporal_regime_ablation",
    )
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05, 1.0]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)

    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)

    unexpected = sorted(set(frame["game_type"].dropna().unique()) - {"R", "F"})
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")

    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    train = frame.loc[frame[season_col].between(2019, 2023)].copy()
    valid = frame.loc[frame[season_col].eq(2024)].copy()
    if train.empty or valid.empty:
        raise ValueError("train/validation split is empty")

    add_regime_features(
        train,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )
    add_regime_features(
        valid,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )

    base_features = recent_core.feature_set("recent_raw_game_type")
    feature_sets: dict[str, tuple[list[str], set[str]]] = {
        "A0_FULL": (list(base_features), set()),
        "A1_DROP_SEASON": (
            [feature for feature in base_features if feature != season_col],
            set(),
        ),
        "A2_ADD_RECENT_F": (
            [*base_features, "eng_recent_f"],
            set(),
        ),
        "A3_ADD_GT_REGIME": (
            [*base_features, "eng_gt_regime"],
            {"eng_gt_regime"},
        ),
    }

    params = build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    gt_valid = valid["game_type"].astype(str).to_numpy()
    masks = {
        "ALL": np.ones(len(valid), dtype=bool),
        "R": gt_valid == "R",
        "F": gt_valid == "F",
    }

    tqdm.write(
        f"GT temporal regime | train=2019-2023 ({len(train):,}) | valid=2024 ({len(valid):,}) | "
        f"R={int(masks['R'].sum()):,} | F={int(masks['F'].sum()):,} | "
        f"regime_start={args.regime_start_year} | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | "
        f"iterations={args.iterations} | catboost={catboost.__version__}"
    )

    progress = tqdm(total=len(VARIANTS), desc="GT temporal regime", unit="model", dynamic_ncols=True)
    predictions: dict[str, np.ndarray] = {}

    for variant in VARIANTS:
        features, extra_categorical = feature_sets[variant]
        seed_everything(seed)
        pred = fit_predict(
            train=train,
            valid=valid,
            target_col=target_col,
            features=features,
            extra_categorical=extra_categorical,
            params=params,
        )
        predictions[variant] = pred
        progress.update(1)

    progress.close()

    # A0 is the control. Compare every variant on exactly the same ALL/R/F rows.
    baseline_metrics = {
        group: binary_metrics(y_valid[mask], predictions["A0_FULL"][mask])
        for group, mask in masks.items()
    }

    rows: list[dict] = []
    for variant in VARIANTS:
        for group, mask in masks.items():
            metric = binary_metrics(y_valid[mask], predictions[variant][mask])
            delta = metric["brier"] - baseline_metrics[group]["brier"]
            rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "rows": int(mask.sum()),
                    "score": metric["score"],
                    "brier": metric["brier"],
                    "loss": metric["loss"],
                    "delta_brier_vs_A0_same_group": float(delta),
                    "feature_count": len(feature_sets[variant][0]),
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "metrics.csv", index=False)

    pred_frame = pd.DataFrame(
        {
            "target": y_valid,
            "game_type": gt_valid,
            **{f"{name}_probability": pred for name, pred in predictions.items()},
        }
    )
    if row_id_col in valid.columns:
        pred_frame.insert(0, row_id_col, valid[row_id_col].to_numpy())
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    save_json(
        {
            "experiment": "pooled game_type x temporal-regime ablation",
            "train_seasons": [2019, 2020, 2021, 2022, 2023],
            "validation_seasons": [2024],
            "regime_start_year": int(args.regime_start_year),
            "variants": {
                name: {
                    "features": features,
                    "extra_categorical": sorted(extra_categorical),
                }
                for name, (features, extra_categorical) in feature_sets.items()
            },
            "semantics": {
                "A0_FULL": "pooled model with season and game_type",
                "A1_DROP_SEASON": "A0 with season removed",
                "A2_ADD_RECENT_F": "A0 plus 1[game_type=F and season>=regime_start_year]",
                "A3_ADD_GT_REGIME": "A0 plus categorical cross {R,F} x {OLD,RECENT}",
            },
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "gpu_ram_part": float(args.gpu_ram_part) if args.task_type == "GPU" else None,
            "pinned_memory_size": args.pinned_memory_size if args.task_type == "GPU" else None,
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
            "score_definition": "100000 * (1 - Brier/[target_mean*(1-target_mean)]), unclipped raw score",
            "loss_definition": "binary logloss",
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Overall]")
    overall = result_df.loc[result_df["group"].eq("ALL")].sort_values("brier")
    for _, row in overall.iterrows():
        metric = {
            "score": float(row["score"]),
            "brier": float(row["brier"]),
            "loss": float(row["loss"]),
        }
        tqdm.write(
            metric_line(
                str(row["variant"]),
                metric,
                float(row["delta_brier_vs_A0_same_group"]),
            )
        )

    tqdm.write("\n[2024 R]")
    r_rows = result_df.loc[result_df["group"].eq("R")].sort_values("brier")
    for _, row in r_rows.iterrows():
        metric = {
            "score": float(row["score"]),
            "brier": float(row["brier"]),
            "loss": float(row["loss"]),
        }
        tqdm.write(
            metric_line(
                str(row["variant"]),
                metric,
                float(row["delta_brier_vs_A0_same_group"]),
            )
        )

    tqdm.write("\n[2024 F]")
    f_rows = result_df.loc[result_df["group"].eq("F")].sort_values("brier")
    for _, row in f_rows.iterrows():
        metric = {
            "score": float(row["score"]),
            "brier": float(row["brier"]),
            "loss": float(row["loss"]),
        }
        tqdm.write(
            metric_line(
                str(row["variant"]),
                metric,
                float(row["delta_brier_vs_A0_same_group"]),
            )
        )

    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
