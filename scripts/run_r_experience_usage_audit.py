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
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


PITCHER_HISTORY_RAW = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

# These are deterministic transforms of the retained pitcher success-history columns
# and must be removed together with the raw history in a fair ablation.
PITCHER_HISTORY_DERIVED = [
    "eng_ps_prev1_minus_long",
    "eng_ps_prev3_minus_long",
    "eng_ps_prev5_minus_long",
    "eng_ps_prev1_minus_prev3",
    "eng_ps_prev3_minus_prev5",
    "eng_ps_prev1_minus_prev5",
    "eng_ps_recent_mean_135",
    "eng_ps_recent_mean_minus_long",
    "eng_ps_recent_range_135",
]

BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("N_0_50", 0, 50),
    ("N_51_200", 51, 200),
    ("N_201_1000", 201, 1000),
    ("N_1001_4000", 1001, 4000),
    ("N_4001_PLUS", 4001, None),
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


def metric_line(name: str, metric: dict[str, float], delta_brier: float | None = None) -> str:
    text = (
        f"{name:<20s} "
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


def fit_predict(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    train_y: np.ndarray,
    valid_y: np.ndarray,
    features: list[str],
    params: dict,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    train_x, categorical = context_core.prepare_x(train, features)
    valid_x, valid_categorical = context_core.prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")

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

    del model, train_pool, valid_pool, train_x, valid_x
    gc.collect()
    return pred


def band_mask(values: np.ndarray, low: int, high: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if high is None:
        return values >= float(low)
    return (values >= float(low)) & (values <= float(high))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the R-only CatBoost already uses pitcher experience as reliability. "
            "Train 2019-2023 R and validate on 2024 R with controlled feature removals."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/r_experience_usage_audit")
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

    train_y = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    valid_y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float32)
    valid_n = pd.to_numeric(valid["asof_pitcher_n"], errors="raise").to_numpy(np.float64)
    if np.isnan(valid_n).any():
        raise ValueError("asof_pitcher_n contains NaN")

    full_features = recent_core.feature_set("recent_drop_game_type")
    history_drop = set(PITCHER_HISTORY_RAW + PITCHER_HISTORY_DERIVED)

    missing_raw = sorted(set(PITCHER_HISTORY_RAW) - set(full_features))
    missing_derived = sorted(set(PITCHER_HISTORY_DERIVED) - set(full_features))
    if missing_raw or missing_derived:
        raise ValueError(
            f"Expected pitcher-history features missing: raw={missing_raw}, derived={missing_derived}"
        )

    variants = {
        "A0_FULL": list(full_features),
        "A1_DROP_N": [f for f in full_features if f != "asof_pitcher_n"],
        "A2_DROP_HISTORY": [f for f in full_features if f not in history_drop],
        "A3_DROP_N_HISTORY": [
            f for f in full_features if f != "asof_pitcher_n" and f not in history_drop
        ],
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

    tqdm.write(
        f"R-only experience audit | train=2019-2023 ({len(train):,}) | valid=2024 ({len(valid):,}) | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | iterations={args.iterations} | "
        f"catboost={catboost.__version__}"
    )

    progress = tqdm(total=len(variants), desc="R experience audit", unit="model", dynamic_ncols=True)
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    full_metric: dict[str, float] | None = None

    for name, features in variants.items():
        seed_everything(seed)
        pred = fit_predict(
            train=train,
            valid=valid,
            train_y=train_y,
            valid_y=valid_y,
            features=features,
            params=params,
        )
        metric = binary_metrics(valid_y, pred)
        if full_metric is None:
            full_metric = metric
        delta = metric["brier"] - full_metric["brier"]
        rows.append(
            {
                "experiment": name,
                "feature_count": int(len(features)),
                **metric,
                "delta_brier_vs_full": float(delta),
            }
        )
        predictions[name] = pred
        tqdm.write(metric_line(name, metric, delta))
        progress.update(1)

    progress.close()
    assert full_metric is not None

    result_df = pd.DataFrame(rows).sort_values(["brier", "loss"]).reset_index(drop=True)
    result_df.to_csv(output_dir / "overall_metrics.csv", index=False)

    # Diagnose where the unchanged global model succeeds/fails as pitcher experience grows.
    full_pred = predictions["A0_FULL"]
    band_rows: list[dict] = []
    tqdm.write("\n[GLOBAL_R by pitcher experience]")
    for band_name, low, high in BANDS:
        mask = band_mask(valid_n, low, high)
        if not mask.any():
            continue
        metric = binary_metrics(valid_y[mask], full_pred[mask])
        band_rows.append(
            {
                "band": band_name,
                "low": int(low),
                "high": None if high is None else int(high),
                "rows": int(mask.sum()),
                "target_rate": float(valid_y[mask].mean()),
                **metric,
            }
        )
        tqdm.write(metric_line(band_name, metric))

    pd.DataFrame(band_rows).to_csv(output_dir / "global_by_experience_band.csv", index=False)

    pred_frame = pd.DataFrame(
        {
            "target": valid_y,
            "asof_pitcher_n": valid_n,
            **{f"{name.lower()}_probability": pred for name, pred in predictions.items()},
        }
    )
    if row_id_col in valid.columns:
        pred_frame.insert(0, row_id_col, valid[row_id_col].to_numpy())
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    save_json(
        {
            "experiment": "R-only pitcher experience/history feature usage audit",
            "train_seasons": [2019, 2020, 2021, 2022, 2023],
            "validation_seasons": [2024],
            "game_type": "R",
            "variants": variants,
            "pitcher_history_raw_removed_together": PITCHER_HISTORY_RAW,
            "pitcher_history_derived_removed_together": PITCHER_HISTORY_DERIVED,
            "bands": [
                {"name": name, "low": low, "high": high}
                for name, low, high in BANDS
            ],
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "gpu_ram_part": float(args.gpu_ram_part) if args.task_type == "GPU" else None,
            "pinned_memory_size": args.pinned_memory_size if args.task_type == "GPU" else None,
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
            "score_definition": "100000 * (1 - Brier / [target_mean*(1-target_mean)]), unclipped raw score",
            "loss_definition": "binary logloss on 2024 R validation rows",
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Overall]")
    for _, row in result_df.iterrows():
        metric = {
            "score": float(row["score"]),
            "brier": float(row["brier"]),
            "loss": float(row["loss"]),
        }
        tqdm.write(metric_line(str(row["experiment"]), metric, float(row["delta_brier_vs_full"])))
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
