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


EB_SOURCES = (
    "asof_pitcher_success_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
)


def parse_lambdas(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError("--lambdas must contain positive integers")
    return list(dict.fromkeys(values))


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
        f"{name:<18s} "
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
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
    categorical: list[str],
    features: list[str],
    params: dict,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

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
    return pred


def numeric_rate(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    values[~np.isfinite(values)] = np.nan
    return values


def compute_priors(train: pd.DataFrame, train_y: np.ndarray) -> dict[str, float]:
    priors: dict[str, float] = {}
    # For the historical success rate, shrink toward the actual R-only training
    # target prior. This is the closest population analogue of control success.
    priors["asof_pitcher_success_rate"] = float(np.mean(train_y))

    # The remaining historical rates describe different pitch-state proportions;
    # no direct labels for those events exist here. Their shrink targets are
    # therefore estimated strictly from the R-only training rows themselves.
    for column in EB_SOURCES[1:]:
        values = numeric_rate(train, column)
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError(f"No finite values for {column}")
        priors[column] = float(np.nanmean(values))
    return priors


def add_eb_features(
    frame: pd.DataFrame,
    *,
    prior: dict[str, float],
    strength: int,
) -> list[str]:
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").to_numpy(np.float64)
    if np.isnan(n).any():
        raise ValueError("asof_pitcher_n contains NaN")
    n = np.clip(n, 0.0, None)

    names: list[str] = []
    for source in EB_SOURCES:
        rate = numeric_rate(frame, source)
        p0 = float(prior[source])
        rate = np.where(np.isfinite(rate), rate, p0)
        eb = (n * rate + float(strength) * p0) / (n + float(strength))
        name = f"eng_eb{strength}_{source.removeprefix('asof_pitcher_')}"
        frame[name] = eb.astype(np.float32)
        names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "R-only empirical-Bayes reliability experiment. Keep one global model, "
            "add n-dependent shrinkage features for pitcher historical rates, train on "
            "2019-2023 R and validate on 2024 R."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lambdas", default="50,200,500,1000,2000")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/r_experience_eb_shrinkage")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    strengths = parse_lambdas(args.lambdas)
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

    base_features = recent_core.feature_set("recent_drop_game_type")
    base_train_x, base_categorical = context_core.prepare_x(train, base_features)
    base_valid_x, valid_categorical = context_core.prepare_x(valid, base_features)
    if base_categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")

    priors = compute_priors(train, train_y)
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
        f"R-only EB shrinkage | train=2019-2023 ({len(train):,}) | valid=2024 ({len(valid):,}) | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | iterations={args.iterations} | "
        f"lambdas={strengths} | catboost={catboost.__version__}"
    )

    progress = tqdm(total=1 + len(strengths), desc="R EB shrinkage", unit="model", dynamic_ncols=True)
    rows: list[dict] = []
    pred_columns: dict[str, np.ndarray] = {}

    seed_everything(seed)
    global_pred = fit_predict(
        train_x=base_train_x,
        train_y=train_y,
        valid_x=base_valid_x,
        valid_y=valid_y,
        categorical=base_categorical,
        features=base_features,
        params=params,
    )
    global_metric = binary_metrics(valid_y, global_pred)
    pred_columns["global_probability"] = global_pred
    rows.append({"experiment": "GLOBAL_R", "lambda": None, **global_metric, "delta_brier_vs_global": 0.0})
    tqdm.write(metric_line("GLOBAL_R", global_metric, 0.0))
    progress.update(1)

    del base_train_x, base_valid_x
    gc.collect()

    for strength in strengths:
        train_variant = train.copy()
        valid_variant = valid.copy()
        eb_names_train = add_eb_features(train_variant, prior=priors, strength=strength)
        eb_names_valid = add_eb_features(valid_variant, prior=priors, strength=strength)
        if eb_names_train != eb_names_valid:
            raise RuntimeError("EB feature mismatch")

        features = [*base_features, *eb_names_train]
        train_x, categorical = context_core.prepare_x(train_variant, features)
        valid_x, valid_cat = context_core.prepare_x(valid_variant, features)
        if categorical != valid_cat:
            raise RuntimeError("categorical feature mismatch")

        seed_everything(seed)
        pred = fit_predict(
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            categorical=categorical,
            features=features,
            params=params,
        )
        metric = binary_metrics(valid_y, pred)
        delta = metric["brier"] - global_metric["brier"]
        name = f"EB_{strength}"
        pred_columns[f"eb_{strength}_probability"] = pred
        rows.append(
            {
                "experiment": name,
                "lambda": int(strength),
                **metric,
                "delta_brier_vs_global": float(delta),
            }
        )
        tqdm.write(metric_line(name, metric, delta))
        progress.update(1)

        del train_variant, valid_variant, train_x, valid_x
        gc.collect()

    progress.close()

    result_df = pd.DataFrame(rows).sort_values(["brier", "loss"]).reset_index(drop=True)
    result_df.to_csv(output_dir / "overall_metrics.csv", index=False)

    pred_frame = pd.DataFrame({"target": valid_y, **pred_columns})
    if row_id_col in valid.columns:
        pred_frame.insert(0, row_id_col, valid[row_id_col].to_numpy())
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    save_json(
        {
            "experiment": "R-only n-dependent empirical-Bayes pitcher-rate shrinkage",
            "train_seasons": [2019, 2020, 2021, 2022, 2023],
            "validation_seasons": [2024],
            "game_type": "R",
            "lambdas": strengths,
            "eb_sources": list(EB_SOURCES),
            "priors_from_training_only": priors,
            "formula": "EB=(n*raw_rate + lambda*population_prior)/(n+lambda)",
            "base_features": base_features,
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
        metric = {"score": float(row["score"]), "brier": float(row["brier"]), "loss": float(row["loss"])}
        tqdm.write(metric_line(str(row["experiment"]), metric, float(row["delta_brier_vs_global"])))
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
