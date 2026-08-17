from __future__ import annotations

import argparse
import gc
import json
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
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("P0", 0, 0),
    ("P1", 1, 10),
    ("P2", 11, 50),
    ("P3", 51, 200),
    ("P4", 201, 1000),
    ("P5", 1001, 4000),
    ("P6", 4001, None),
)


def band_mask(values: np.ndarray, low: int, high: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if high is None:
        return values >= float(low)
    return (values >= float(low)) & (values <= float(high))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    metric = probability_metrics(y, p)
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    return {
        "score": float(metric["raw_score"]),
        "brier": float(metric["brier"]),
        "loss": logloss,
        "target_rate": float(y.mean()),
        "prediction_mean": float(p.mean()),
    }


def metric_line(name: str, metric: dict[str, float]) -> str:
    return (
        f"{name:<18s} "
        f"score={metric['score']:+9.2f}  "
        f"brier={metric['brier']:.8f}  "
        f"loss={metric['loss']:.8f}"
    )


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
        # CatBoost does not expose a neural-style batch_size knob. For a single
        # RTX 4090, throughput is instead maximized by keeping categorical data
        # on VRAM and allowing CatBoost to use nearly all available GPU memory.
        params["gpu_ram_part"] = float(gpu_ram_part)
        params["pinned_memory_size"] = str(pinned_memory_size)
        params["gpu_cat_features_storage"] = "GpuRam"
    return params


def fit_predict(
    *,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
    categorical: list[str],
    features: list[str],
    params: dict,
) -> tuple[np.ndarray, str]:
    from catboost import CatBoostClassifier, Pool

    if len(train_y) == 0:
        raise ValueError("empty training subset")
    if len(valid_y) == 0:
        raise ValueError("empty validation subset")

    unique = np.unique(train_y)
    if unique.size < 2:
        prior = float(np.clip(train_y.mean(), 1e-6, 1.0 - 1e-6))
        return np.full(len(valid_y), prior, dtype=np.float64), "constant_single_class"

    train_pool = Pool(
        train_x,
        label=train_y,
        cat_features=categorical,
        feature_names=features,
    )
    valid_pool = Pool(
        valid_x,
        label=valid_y,
        cat_features=categorical,
        feature_names=features,
    )
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=False)
    pred = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)

    del model, train_pool, valid_pool
    gc.collect()
    return pred, "catboost"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R-only hard pitcher-experience specialization. Train 2019-2023 R rows, "
            "validate on 2024 R, and compare one global CatBoost with completely "
            "independent CatBoost models for P0..P6 experience bands."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument(
        "--devices",
        default="2",
        help="CatBoost GPU device id. Default 2 = the third GPU.",
    )
    parser.add_argument(
        "--gpu-ram-part",
        type=float,
        default=0.95,
        help="Fraction of GPU VRAM CatBoost may use. Default 0.95.",
    )
    parser.add_argument(
        "--pinned-memory-size",
        default="4GB",
        help="Pinned host memory per GPU. Default 4GB.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=100,
        help="Safety threshold. Smaller bands use their own training prior instead of CatBoost.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/r_only_experience_specialists",
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
    frame = frame.loc[frame["game_type"].eq("R")].copy()

    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    train = frame.loc[frame[season_col].between(2019, 2023)].copy()
    valid = frame.loc[frame[season_col].eq(2024)].copy()
    if train.empty or valid.empty:
        raise ValueError("R-only train/validation split is empty")

    # Same feature representation for baseline and every specialist. game_type is
    # removed because this experiment is already conditioned on R and it is constant.
    features = recent_core.feature_set("recent_drop_game_type")
    train_x, categorical = context_core.prepare_x(train, features)
    valid_x, valid_categorical = context_core.prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")

    train_y = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    valid_y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float32)
    train_n = pd.to_numeric(train["asof_pitcher_n"], errors="raise").to_numpy(np.float64)
    valid_n = pd.to_numeric(valid["asof_pitcher_n"], errors="raise").to_numpy(np.float64)

    if np.isnan(train_n).any() or np.isnan(valid_n).any():
        raise ValueError("asof_pitcher_n contains NaN")

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

    tqdm.write(
        f"R-only | train=2019-2023 ({len(train):,}) | valid=2024 ({len(valid):,}) | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | iterations={args.iterations} | "
        f"catboost={catboost.__version__}"
    )

    jobs = 1 + len(BANDS)
    progress = tqdm(total=jobs, desc="R experience", unit="model", dynamic_ncols=True)

    # 1) One global R-only baseline.
    global_pred, global_mode = fit_predict(
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        categorical=categorical,
        features=features,
        params=params,
    )
    global_metric = binary_metrics(valid_y, global_pred)
    tqdm.write(metric_line("GLOBAL_R", global_metric))
    progress.update(1)

    routed_pred = np.full(len(valid), np.nan, dtype=np.float64)
    band_rows: list[dict] = []

    for band_name, low, high in BANDS:
        tr_mask = band_mask(train_n, low, high)
        va_mask = band_mask(valid_n, low, high)
        tr_idx = np.flatnonzero(tr_mask)
        va_idx = np.flatnonzero(va_mask)

        if va_idx.size == 0:
            tqdm.write(f"{band_name:<18s} score=      nan  brier=nan       loss=nan")
            progress.update(1)
            band_rows.append(
                {
                    "band": band_name,
                    "low": low,
                    "high": high,
                    "train_rows": int(tr_idx.size),
                    "valid_rows": 0,
                    "mode": "no_validation_rows",
                }
            )
            continue

        baseline_band_metric = binary_metrics(valid_y[va_idx], global_pred[va_idx])

        if tr_idx.size < args.min_train_rows:
            prior = float(np.clip(train_y[tr_idx].mean(), 1e-6, 1.0 - 1e-6)) if tr_idx.size else float(train_y.mean())
            specialist_pred = np.full(va_idx.size, prior, dtype=np.float64)
            specialist_mode = "constant_small_band"
        else:
            seed_everything(seed)
            specialist_pred, specialist_mode = fit_predict(
                train_x=train_x.iloc[tr_idx],
                train_y=train_y[tr_idx],
                valid_x=valid_x.iloc[va_idx],
                valid_y=valid_y[va_idx],
                categorical=categorical,
                features=features,
                params=params,
            )

        routed_pred[va_idx] = specialist_pred
        specialist_metric = binary_metrics(valid_y[va_idx], specialist_pred)
        tqdm.write(metric_line(f"SPECIALIST_{band_name}", specialist_metric))
        progress.update(1)

        band_rows.append(
            {
                "band": band_name,
                "low": int(low),
                "high": None if high is None else int(high),
                "train_rows": int(tr_idx.size),
                "valid_rows": int(va_idx.size),
                "train_target_rate": float(train_y[tr_idx].mean()) if tr_idx.size else np.nan,
                "valid_target_rate": float(valid_y[va_idx].mean()),
                "mode": specialist_mode,
                "baseline_score": baseline_band_metric["score"],
                "baseline_brier": baseline_band_metric["brier"],
                "baseline_loss": baseline_band_metric["loss"],
                "specialist_score": specialist_metric["score"],
                "specialist_brier": specialist_metric["brier"],
                "specialist_loss": specialist_metric["loss"],
                "delta_brier_specialist_minus_baseline": specialist_metric["brier"] - baseline_band_metric["brier"],
                "delta_loss_specialist_minus_baseline": specialist_metric["loss"] - baseline_band_metric["loss"],
            }
        )

        gc.collect()

    progress.close()

    if np.isnan(routed_pred).any():
        missing = int(np.isnan(routed_pred).sum())
        raise RuntimeError(f"{missing} validation rows were not routed to an experience band")

    routed_metric = binary_metrics(valid_y, routed_pred)
    tqdm.write(metric_line("ROUTED_SPECIALISTS", routed_metric))

    band_df = pd.DataFrame(band_rows)
    band_df.to_csv(output_dir / "band_comparison.csv", index=False)

    overall_df = pd.DataFrame(
        [
            {
                "experiment": "global_r",
                "mode": global_mode,
                **global_metric,
            },
            {
                "experiment": "routed_specialists",
                "mode": "hard_pitcher_experience_routing",
                **routed_metric,
            },
        ]
    )
    overall_df["delta_brier_vs_global"] = overall_df["brier"] - global_metric["brier"]
    overall_df["delta_loss_vs_global"] = overall_df["loss"] - global_metric["loss"]
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)

    pred_frame = pd.DataFrame(
        {
            "target": valid_y,
            "asof_pitcher_n": valid_n,
            "global_probability": global_pred,
            "specialist_probability": routed_pred,
        }
    )
    if row_id_col in valid.columns:
        pred_frame.insert(0, row_id_col, valid[row_id_col].to_numpy())
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    save_json(
        {
            "experiment": "R-only independent pitcher-experience CatBoost specialists",
            "train_seasons": [2019, 2020, 2021, 2022, 2023],
            "validation_seasons": [2024],
            "game_type": "R",
            "bands": [
                {"name": name, "low": low, "high": high}
                for name, low, high in BANDS
            ],
            "features": features,
            "feature_count": len(features),
            "categorical": categorical,
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "gpu_ram_part": float(args.gpu_ram_part) if args.task_type == "GPU" else None,
            "pinned_memory_size": args.pinned_memory_size if args.task_type == "GPU" else None,
            "min_train_rows": int(args.min_train_rows),
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
            "score_definition": "100000 * (1 - Brier / [target_mean*(1-target_mean)]), unclipped raw score",
            "loss_definition": "binary logloss on 2024 R validation rows",
            "global": global_metric,
            "routed_specialists": routed_metric,
        },
        output_dir / "run_config.json",
    )

    delta = routed_metric["brier"] - global_metric["brier"]
    tqdm.write(f"delta_brier={delta:+.8f}")
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
