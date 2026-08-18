from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_recent_regime_submissions as recent_core
import run_game_type_temporal_regime_ablation as regime_core
import run_regime_feature_prediction_suite as feature_core
from src.utils import load_config, save_json, seed_everything


GROUPS = (
    ("gt", ("game_type",), 20.0),
    ("pitcher_team_gt", ("pitcher_team_id", "game_type"), 200.0),
    ("batter_team_gt", ("batter_team_id", "game_type"), 200.0),
    ("pitcher_gt", ("pitcher_id", "game_type"), 800.0),
    ("batter_gt", ("batter_id", "game_type"), 1200.0),
)
SCALES = (5.0, 20.0, 100.0)


def parse_ints(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values:
        raise ValueError("at least one fold is required")
    return values


def token(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    value = frame[columns[0]].astype("string").fillna("<MISSING>").astype(str)
    for column in columns[1:]:
        value = value.str.cat(frame[column].astype("string").fillna("<MISSING>").astype(str), sep="|")
    return value


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def fit_effects(
    frame: pd.DataFrame,
    y: np.ndarray,
    base_probability: np.ndarray,
    scale: float,
    passes: int,
) -> dict[str, dict[str, float]]:
    eta = logit(base_probability)
    keys = {name: token(frame, columns) for name, columns, _ in GROUPS}
    effects: dict[str, dict[str, float]] = {name: {} for name, _, _ in GROUPS}

    for _ in range(passes):
        for name, _, base_lambda in GROUPS:
            key = keys[name]
            probability = sigmoid(eta)
            work = pd.DataFrame({
                "key": key.to_numpy(),
                "gradient": y - probability,
                "weight": probability * (1.0 - probability),
            })
            grouped = work.groupby("key", sort=False)[["gradient", "weight"]].sum()
            delta = (grouped["gradient"] / (grouped["weight"] + base_lambda * scale)).clip(-0.5, 0.5)
            eta += key.map(delta).fillna(0.0).to_numpy(np.float64)
            current = effects[name]
            for group_key, value in delta.items():
                group_key = str(group_key)
                current[group_key] = current.get(group_key, 0.0) + float(value)
    return effects


def apply_effects(frame: pd.DataFrame, base_probability: np.ndarray, effects: dict[str, dict[str, float]]) -> np.ndarray:
    eta = logit(base_probability)
    for name, columns, _ in GROUPS:
        eta += token(frame, columns).map(effects[name]).fillna(0.0).to_numpy(np.float64)
    return sigmoid(eta)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward hierarchical residual transfer on a shared regime backbone.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--folds", default="2022,2023,2024")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--calibration-month-start", type=int, default=8)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--gpu-ram-part", type=float, default=0.95)
    ap.add_argument("--pinned-memory-size", default="4GB")
    ap.add_argument("--output-dir", default="outputs/hierarchical_residual_transfer")
    args = ap.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    folds = parse_ints(args.folds)
    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]

    frame, invariants = recent_core.prepare_frame(config)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()
    base_features = recent_core.feature_set("recent_raw_game_type")
    model_features = [*base_features, feature_core.RECENT_FLAG, *feature_core.FAST_CONT, *feature_core.RANGE_CONT]
    params = regime_core.build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    progress = tqdm(total=len(folds) * 2, desc="hierarchical transfer", unit="base", dynamic_ncols=True)

    for val_year in folds:
        cal_year = val_year - 1
        adaptation_mask = (frame[season] < cal_year) | (
            frame[season].eq(cal_year)
            & pd.to_numeric(frame["game_month"], errors="raise").lt(args.calibration_month_start)
        )
        calibration_mask = frame[season].eq(cal_year) & pd.to_numeric(
            frame["game_month"], errors="raise"
        ).ge(args.calibration_month_start)
        old = frame.loc[adaptation_mask].copy()
        calibration = frame.loc[calibration_mask].copy()
        full = frame.loc[frame[season] < val_year].copy()
        valid = frame.loc[frame[season].eq(val_year)].copy()
        if min(map(len, (old, calibration, full, valid))) == 0:
            raise ValueError(f"fold {val_year}: empty old/calibration/full/valid partition")

        feature_core.add_regime_features(old, calibration, season_col=season, recent_start=2023)
        feature_core.add_regime_features(full, valid, season_col=season, recent_start=2023)
        seed_everything(seed)
        p_cal = regime_core.fit_predict(
            train=old, valid=calibration, target_col=target, features=model_features,
            extra_categorical=set(), params=params,
        )
        progress.update(1)
        seed_everything(seed)
        p_valid = regime_core.fit_predict(
            train=full, valid=valid, target_col=target, features=model_features,
            extra_categorical=set(), params=params,
        )
        progress.update(1)

        y_cal = pd.to_numeric(calibration[target], errors="raise").to_numpy(np.float64)
        y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
        gt = valid["game_type"].astype(str).to_numpy()
        predictions = {"BASE": p_valid}
        for scale in SCALES:
            effects = fit_effects(calibration, y_cal, p_cal, scale=scale, passes=args.passes)
            predictions[f"HIER_{scale:g}"] = apply_effects(valid, p_valid, effects)

        base_brier = regime_core.binary_metrics(y_valid, p_valid)["brier"]
        for name, pred in predictions.items():
            for group, mask in {"ALL": np.ones(len(valid), bool), "R": gt == "R", "F": gt == "F"}.items():
                metric = regime_core.binary_metrics(y_valid[mask], pred[mask])
                rows.append({
                    "fold": val_year, "calibration_year": cal_year, "variant": name,
                    "group": group, "rows": int(mask.sum()), **metric,
                    "delta_brier_vs_base_all": metric["brier"] - base_brier if group == "ALL" else np.nan,
                })
        if val_year == 2024:
            np.savez_compressed(out / "validation_2024_predictions.npz", y=y_valid, gt=gt, **predictions)
        del old, calibration, full, valid, p_cal, p_valid, predictions
        gc.collect()

    progress.close()
    results = pd.DataFrame(rows)
    results.to_csv(out / "fold_metrics.csv", index=False)
    overall = results.loc[results.group.eq("ALL")]
    summary = overall.groupby("variant", as_index=False).agg(
        folds=("fold", "count"), mean_brier=("brier", "mean"),
        mean_delta=("delta_brier_vs_base_all", "mean"),
        worst_delta=("delta_brier_vs_base_all", "max"),
        wins=("delta_brier_vs_base_all", lambda x: int((x < 0).sum())),
    ).sort_values("mean_delta")
    summary.to_csv(out / "summary.csv", index=False)
    save_json({
        "experiment": "shared nonlinear backbone plus previous-season hierarchical logit residual transfer",
        "folds": folds, "iterations": args.iterations, "passes": args.passes,
        "calibration_month_start": args.calibration_month_start,
        "groups": [{"name": n, "columns": list(c), "lambda": l} for n, c, l in GROUPS],
        "lambda_scales": list(SCALES), "canonical_invariants": invariants,
        "leakage_guard": "for Y, residual base trains through early Y-1 and effects fit only on late Y-1; final base trains through all Y-1; effects freeze for every Y row",
    }, out / "run_config.json")

    latest = results.loc[(results.fold == max(folds)) & (results.group == "ALL")]
    for _, row in latest.sort_values("brier").iterrows():
        print(
            f"{row['variant']}: s={row['score']:.1f} "
            f"b={row['brier']:.3e} d={row['delta_brier_vs_base_all']:+.2e}"
        )
    print(f"out={out}")


if __name__ == "__main__":
    main()
