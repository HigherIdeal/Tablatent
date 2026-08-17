from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_2025_proxy_validation as proxy_core
import run_context_interaction_screen as context_core
import run_gated_r_specialist_suite as gated_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


SYNTHETIC_GATE_FEATURES = [
    "gate_p_full",
    "gate_p_recent",
    "gate_p_diff",
    "gate_p_abs_diff",
    "gate_p_mean",
]


def parse_ints(value: str) -> list[int]:
    out = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("empty integer list")
    return out


def parse_floats(value: str) -> list[float]:
    out = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("empty float list")
    return out


def _token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _fit_expert(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target_col: str,
    config: dict,
    iterations: int,
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
) -> np.ndarray:
    pred, _ = gated_core._fit_prefixes(
        train=train,
        valid=valid,
        features=features,
        target_col=target_col,
        config=config,
        iterations_grid=[iterations],
        task_type=task_type,
        device=device,
        verbose=verbose,
        thread_count=thread_count,
    )
    return np.asarray(pred[iterations], dtype=np.float64)


def _gate_frame(rows: pd.DataFrame, p_full: np.ndarray, p_recent: np.ndarray) -> pd.DataFrame:
    out = rows.copy()
    p_full = np.asarray(p_full, dtype=np.float64)
    p_recent = np.asarray(p_recent, dtype=np.float64)
    if len(out) != len(p_full) or len(out) != len(p_recent):
        raise ValueError("gate frame/prediction row mismatch")
    out["gate_p_full"] = p_full
    out["gate_p_recent"] = p_recent
    out["gate_p_diff"] = p_recent - p_full
    out["gate_p_abs_diff"] = np.abs(p_recent - p_full)
    out["gate_p_mean"] = 0.5 * (p_recent + p_full)
    return out


def _prepare_gate_x(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"missing gate features: {missing}")
    x = frame.loc[:, features].copy()
    categorical = [f for f in features if f in set(CANONICAL_CATEGORICAL)]
    cat = set(categorical)
    for col in features:
        if col in cat:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(np.float32)
            x[col] = x[col].replace([np.inf, -np.inf], np.nan)
    return x, categorical


def _fit_gate_regressor(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
    features: list[str],
    target_col: str,
    seed: int,
    iterations: int,
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
):
    from catboost import CatBoostRegressor, Pool

    x_train, categorical = _prepare_gate_x(train, features)
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    params = {
        "loss_function": "RMSE",
        "iterations": int(iterations),
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": int(verbose),
        "thread_count": int(thread_count),
        "task_type": task_type,
    }
    if task_type == "GPU":
        params["devices"] = str(device)
    model = CatBoostRegressor(**params)
    model.fit(train_pool, verbose=verbose)

    valid_score = None
    if valid is not None:
        x_valid, valid_cat = _prepare_gate_x(valid, features)
        if valid_cat != categorical:
            raise RuntimeError("gate categorical mismatch")
        valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
        valid_score = np.asarray(model.predict(valid_pool), dtype=np.float64)
        del x_valid, valid_pool
    del x_train, train_pool, y_train
    gc.collect()
    return model, valid_score, categorical


def _predict_gate(model, frame: pd.DataFrame, features: list[str], categorical: list[str]) -> np.ndarray:
    from catboost import Pool

    x, cat = _prepare_gate_x(frame, features)
    if cat != categorical:
        raise RuntimeError("gate categorical mismatch at prediction")
    pool = Pool(x, cat_features=categorical, feature_names=features)
    score = np.asarray(model.predict(pool), dtype=np.float64)
    del x, pool
    return score


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(math.log(p / (1.0 - p)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def dynamic_alpha(
    score: np.ndarray,
    *,
    center: float,
    scale: float,
    base_alpha: float,
    strength: float,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    z = (np.asarray(score, dtype=np.float64) - float(center)) / max(float(scale), 1e-8)
    alpha = _sigmoid(_logit(base_alpha) + float(strength) * z)
    return np.clip(alpha, alpha_min, alpha_max)


def _blend_with_r(
    p_full: np.ndarray,
    p_recent: np.ndarray,
    p_r: np.ndarray,
    is_r: np.ndarray,
    alpha: np.ndarray | float,
    beta_r: float,
) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64)
    if a.ndim == 0:
        a = np.full(len(p_full), float(a), dtype=np.float64)
    base = (1.0 - a) * np.asarray(p_full) + a * np.asarray(p_recent)
    out = base.copy()
    mask = np.asarray(is_r, dtype=bool)
    out[mask] = (1.0 - beta_r) * base[mask] + beta_r * np.asarray(p_r)[mask]
    return np.clip(out, 0.0, 1.0)


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def _subset_brier(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    return _brier(y[mask], p[mask]) if mask.any() else float("nan")


def build_gate_oof(
    *,
    frame: pd.DataFrame,
    years: list[int],
    row_features: list[str],
    expert_features: list[str],
    target_col: str,
    config: dict,
    iterations: int,
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
) -> pd.DataFrame:
    """Create strictly forward OOF expert predictions used to train the gate.

    For validation season Y:
      full expert   = all seasons < Y
      recent expert = season Y-1 only
      gate label    = squared-error(full) - squared-error(recent)

    Positive gate labels mean the recent expert was better on that row.  No gate
    training row is predicted by an expert that saw that row's label.
    """
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    blocks: list[pd.DataFrame] = []
    keep_cols = list(dict.fromkeys(row_features + [target_col]))
    for year in years:
        full_mask = season.lt(year)
        recent_mask = season.eq(year - 1)
        valid_mask = season.eq(year)
        if not full_mask.any() or not recent_mask.any() or not valid_mask.any():
            raise RuntimeError(f"cannot build gate OOF year={year}")
        valid = frame.loc[valid_mask].copy()
        p_full = _fit_expert(
            train=frame.loc[full_mask].copy(), valid=valid, features=expert_features,
            target_col=target_col, config=config, iterations=iterations, task_type=task_type,
            device=device, verbose=verbose, thread_count=thread_count,
        )
        p_recent = _fit_expert(
            train=frame.loc[recent_mask].copy(), valid=valid, features=expert_features,
            target_col=target_col, config=config, iterations=iterations, task_type=task_type,
            device=device, verbose=verbose, thread_count=thread_count,
        )
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        block = _gate_frame(valid[keep_cols], p_full, p_recent)
        block["gate_advantage_recent"] = (y - p_full) ** 2 - (y - p_recent) ** 2
        block["gate_oof_year"] = int(year)
        blocks.append(block)
        print(
            f"  gate OOF {year}: rows={len(block):,} "
            f"full_brier={_brier(y, p_full):.8f} recent_brier={_brier(y, p_recent):.8f} "
            f"mean_adv={block['gate_advantage_recent'].mean():+.8f}"
        )
        del valid, p_full, p_recent, y, block
        gc.collect()
    return pd.concat(blocks, ignore_index=True)


def calibrate_gate(
    *,
    oof: pd.DataFrame,
    features: list[str],
    target_col: str,
    calibration_year: int,
    strengths: list[float],
    base_alpha: float,
    alpha_min: float,
    alpha_max: float,
    seed: int,
    gate_iterations: int,
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
) -> tuple[float, pd.DataFrame]:
    train = oof.loc[oof["gate_oof_year"].lt(calibration_year)].copy()
    valid = oof.loc[oof["gate_oof_year"].eq(calibration_year)].copy()
    if train.empty or valid.empty:
        raise RuntimeError("empty gate train/calibration split")
    model, score, _ = _fit_gate_regressor(
        train=train, valid=valid, features=features, target_col="gate_advantage_recent",
        seed=seed, iterations=gate_iterations, task_type=task_type, device=device,
        verbose=verbose, thread_count=thread_count,
    )
    if score is None:
        raise RuntimeError("missing gate calibration score")
    center = float(np.median(score))
    scale = float(np.std(score))
    y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    p_full = valid["gate_p_full"].to_numpy(np.float64)
    p_recent = valid["gate_p_recent"].to_numpy(np.float64)
    rows = []
    for strength in strengths:
        alpha = dynamic_alpha(
            score, center=center, scale=scale, base_alpha=base_alpha, strength=strength,
            alpha_min=alpha_min, alpha_max=alpha_max,
        )
        pred = (1.0 - alpha) * p_full + alpha * p_recent
        rows.append(
            {
                "strength": float(strength),
                "brier": _brier(y, pred),
                "alpha_mean": float(alpha.mean()),
                "alpha_std": float(alpha.std()),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "score_center": center,
                "score_scale": scale,
            }
        )
    table = pd.DataFrame(rows).sort_values(["brier", "strength"]).reset_index(drop=True)
    best_strength = float(table.iloc[0]["strength"])
    del model, train, valid
    gc.collect()
    return best_strength, table


def fit_final_gate(
    *,
    oof: pd.DataFrame,
    features: list[str],
    seed: int,
    gate_iterations: int,
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
):
    model, _, categorical = _fit_gate_regressor(
        train=oof, valid=None, features=features, target_col="gate_advantage_recent",
        seed=seed, iterations=gate_iterations, task_type=task_type, device=device,
        verbose=verbose, thread_count=thread_count,
    )
    score = _predict_gate(model, oof, features, categorical)
    center = float(np.median(score))
    scale = float(np.std(score))
    return model, categorical, center, scale


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Leakage-safe row-wise dynamic gate for full/recent CatBoost experts. The gate is trained only on "
            "strict forward OOF expert errors from historical seasons and, at inference, uses only the current "
            "row plus the two current-row expert predictions. No test-row interaction is permitted."
        )
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--gate-iterations", type=int, default=300)
    p.add_argument("--gate-oof-years", default="2021,2022,2023")
    p.add_argument("--calibration-year", type=int, default=2023)
    p.add_argument("--strengths", default="0,0.125,0.25,0.5,1,2,4")
    p.add_argument("--alpha-base", type=float, default=0.20)
    p.add_argument("--alpha-min", type=float, default=0.02)
    p.add_argument("--alpha-max", type=float, default=0.60)
    p.add_argument("--beta-r", type=float, default=0.10)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--thread-count", type=int, default=6)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/rowwise_dynamic_gate")
    args = p.parse_args()

    if not (0.0 < args.alpha_base < 1.0):
        raise ValueError("alpha-base must be in (0,1)")
    if not (0.0 <= args.alpha_min <= args.alpha_base <= args.alpha_max <= 1.0):
        raise ValueError("require 0 <= alpha_min <= alpha_base <= alpha_max <= 1")
    if not (0.0 <= args.beta_r <= 1.0):
        raise ValueError("beta-r must be in [0,1]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target_col = config["data"]["target_col"]
    frame, invariant_check = recent_core.prepare_frame(config)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)

    base_features = recent_core.feature_set("recent_raw_game_type")
    # Gate deliberately drops direct season and team identity. It may use month,
    # game_type, count/context, history, and expert disagreement from THIS row.
    row_gate_features = [
        f for f in base_features if f not in {"season", "pitcher_team_id", "batter_team_id"}
    ]
    row_gate_features = list(dict.fromkeys(row_gate_features))
    outputs_only_features = list(SYNTHETIC_GATE_FEATURES)
    context_gate_features = list(dict.fromkeys(SYNTHETIC_GATE_FEATURES + row_gate_features))
    if target_col in context_gate_features or "row_id" in context_gate_features:
        raise RuntimeError("target/row_id leaked into gate feature set")

    gate_years = parse_ints(args.gate_oof_years)
    strengths = parse_floats(args.strengths)
    if args.calibration_year not in gate_years:
        raise ValueError("calibration-year must be included in gate-oof-years")
    if min(gate_years) < 2021:
        raise ValueError("gate OOF starts at 2021 so full/recent experts have distinct histories")

    devices = gated_core.parse_devices(args.devices)
    device = devices[0] if args.task_type == "GPU" else "CPU"
    r_fast_features = gated_core._feature_sets(base_features)["r_fast"]
    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Row-wise Dynamic Gate]")
    print(f"  rows={len(frame):,} iterations={args.iterations} gate_iterations={args.gate_iterations}")
    print(f"  gate_oof_years={gate_years} calibration={args.calibration_year}")
    print(f"  fixed baseline alpha={args.alpha_base:.3f} beta_r={args.beta_r:.3f}")
    print("  HARD CONSTRAINT: each validation/test row is gated independently; no other test row is read")

    print("\n[1/3] Building strict forward OOF expert predictions for gate training")
    oof = build_gate_oof(
        frame=frame,
        years=gate_years,
        row_features=row_gate_features,
        expert_features=base_features,
        target_col=target_col,
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        device=device,
        verbose=args.verbose,
        thread_count=args.thread_count,
    )
    oof.to_csv(outdir / "gate_oof_predictions.csv", index=False)

    gate_specs = {
        "outputs_only": outputs_only_features,
        "row_context": context_gate_features,
    }
    calibrated: dict[str, dict[str, object]] = {}
    calibration_rows = []

    print("\n[2/3] Calibrating gate strength on the latest historical OOF season")
    for i, (name, features) in enumerate(gate_specs.items()):
        best_strength, table = calibrate_gate(
            oof=oof,
            features=features,
            target_col=target_col,
            calibration_year=args.calibration_year,
            strengths=strengths,
            base_alpha=args.alpha_base,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            seed=seed + 100 + i,
            gate_iterations=args.gate_iterations,
            task_type=args.task_type,
            device=device,
            verbose=args.verbose,
            thread_count=args.thread_count,
        )
        table.insert(0, "gate", name)
        calibration_rows.append(table)
        model, categorical, center, scale = fit_final_gate(
            oof=oof,
            features=features,
            seed=seed + 200 + i,
            gate_iterations=args.gate_iterations,
            task_type=args.task_type,
            device=device,
            verbose=args.verbose,
            thread_count=args.thread_count,
        )
        calibrated[name] = {
            "features": features,
            "model": model,
            "categorical": categorical,
            "strength": best_strength,
            "center": center,
            "scale": scale,
        }
        print(
            f"  {name:<12s} strength={best_strength:g} final_score_center={center:+.6g} "
            f"scale={scale:.6g} features={len(features)}"
        )
        if name == "row_context":
            imp = pd.DataFrame({"feature": features, "importance": model.get_feature_importance()})
            imp.sort_values("importance", ascending=False).to_csv(outdir / "gate_feature_importance.csv", index=False)

    calibration = pd.concat(calibration_rows, ignore_index=True)
    calibration.to_csv(outdir / "gate_calibration.csv", index=False)

    print("\n[3/3] Evaluating fixed vs dynamic gates on established 2025 proxy folds")
    fold_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    fold_weights = {spec.name: float(spec.weight) for spec in proxy_core.DEFAULT_FOLDS}

    for spec in proxy_core.DEFAULT_FOLDS:
        recent_mask, full_mask, valid_mask = proxy_core.fold_masks(frame, spec, "season", "game_month")
        valid = frame.loc[valid_mask].copy()
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        is_r = _token(valid["game_type"]).eq("R").to_numpy()
        is_f = _token(valid["game_type"]).eq("F").to_numpy()
        recent_r_mask = recent_mask & _token(frame["game_type"]).eq("R")

        p_full = _fit_expert(
            train=frame.loc[full_mask].copy(), valid=valid, features=base_features,
            target_col=target_col, config=config, iterations=args.iterations, task_type=args.task_type,
            device=device, verbose=args.verbose, thread_count=args.thread_count,
        )
        p_recent = _fit_expert(
            train=frame.loc[recent_mask].copy(), valid=valid, features=base_features,
            target_col=target_col, config=config, iterations=args.iterations, task_type=args.task_type,
            device=device, verbose=args.verbose, thread_count=args.thread_count,
        )
        p_r = _fit_expert(
            train=frame.loc[recent_r_mask].copy(), valid=valid, features=r_fast_features,
            target_col=target_col, config=config, iterations=args.iterations, task_type=args.task_type,
            device=device, verbose=args.verbose, thread_count=args.thread_count,
        )

        predictions: dict[str, np.ndarray] = {
            "fixed": _blend_with_r(
                p_full, p_recent, p_r, is_r, args.alpha_base, args.beta_r
            )
        }
        gate_valid = _gate_frame(valid[row_gate_features], p_full, p_recent)
        for name, info in calibrated.items():
            score = _predict_gate(
                info["model"], gate_valid, info["features"], info["categorical"]
            )
            alpha = dynamic_alpha(
                score,
                center=float(info["center"]),
                scale=float(info["scale"]),
                base_alpha=args.alpha_base,
                strength=float(info["strength"]),
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
            )
            predictions[name] = _blend_with_r(p_full, p_recent, p_r, is_r, alpha, args.beta_r)
            alpha_rows.append(
                {
                    "fold": spec.name,
                    "gate": name,
                    "alpha_mean": float(alpha.mean()),
                    "alpha_std": float(alpha.std()),
                    "alpha_min": float(alpha.min()),
                    "alpha_p10": float(np.quantile(alpha, 0.10)),
                    "alpha_p50": float(np.quantile(alpha, 0.50)),
                    "alpha_p90": float(np.quantile(alpha, 0.90)),
                    "alpha_max": float(alpha.max()),
                    "alpha_mean_R": float(alpha[is_r].mean()) if is_r.any() else np.nan,
                    "alpha_mean_F": float(alpha[is_f].mean()) if is_f.any() else np.nan,
                }
            )

        fixed_brier = _brier(y, predictions["fixed"])
        print(f"\n  [{spec.name}] valid={len(valid):,} rate={y.mean():.6f}")
        for variant, pred in predictions.items():
            metric = probability_metrics(y, pred)
            row = {
                "fold": spec.name,
                "weight": float(spec.weight),
                "variant": variant,
                "brier": float(metric["brier"]),
                "raw_score": float(metric["raw_score"]),
                "delta_brier_vs_fixed": float(metric["brier"] - fixed_brier),
                "r_brier": _subset_brier(y, pred, is_r),
                "f_brier": _subset_brier(y, pred, is_f),
            }
            fold_rows.append(row)
            print(
                f"    {variant:<12s} brier={row['brier']:.8f} raw={row['raw_score']:+.2f} "
                f"dBrier={row['delta_brier_vs_fixed']:+.8f} R={row['r_brier']:.8f} F={row['f_brier']:.8f}"
            )

        del valid, p_full, p_recent, p_r, gate_valid, predictions
        gc.collect()

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(outdir / "fold_metrics.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(outdir / "alpha_diagnostics.csv", index=False)

    summary_rows = []
    for variant, group in folds.groupby("variant", sort=False):
        weights = np.asarray([fold_weights[str(f)] for f in group["fold"]], dtype=np.float64)
        weights /= weights.sum()
        brier = group["brier"].to_numpy(np.float64)
        delta = group["delta_brier_vs_fixed"].to_numpy(np.float64)
        raw = group["raw_score"].to_numpy(np.float64)
        summary_rows.append(
            {
                "variant": variant,
                "weighted_brier": float(np.dot(weights, brier)),
                "weighted_raw_score": float(np.dot(weights, raw)),
                "weighted_delta_brier_vs_fixed": float(np.dot(weights, delta)),
                "worst_delta_brier_vs_fixed": float(delta.max()),
                "best_delta_brier_vs_fixed": float(delta.min()),
                "improved_folds": int(np.count_nonzero(delta < 0.0)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("weighted_brier").reset_index(drop=True)
    summary.to_csv(outdir / "summary.csv", index=False)

    metadata = {
        "experiment": "rowwise_dynamic_gate",
        "row_independent_inference": True,
        "gate_training": "strict season-forward OOF expert advantage",
        "gate_oof_years": gate_years,
        "calibration_year": int(args.calibration_year),
        "expert_iterations": int(args.iterations),
        "gate_iterations": int(args.gate_iterations),
        "alpha_base": float(args.alpha_base),
        "alpha_bounds": [float(args.alpha_min), float(args.alpha_max)],
        "beta_r": float(args.beta_r),
        "gate_variants": {
            name: {
                "features": info["features"],
                "strength": float(info["strength"]),
                "center": float(info["center"]),
                "scale": float(info["scale"]),
            }
            for name, info in calibrated.items()
        },
        "canonical_invariant_check": invariant_check,
        "important_constraint": "At inference the gate uses only x_i, p_full(x_i), and p_recent(x_i). No statistics or state from other hidden-test rows are used.",
    }
    save_json(metadata, outdir / "metadata.json")

    print("\n[Summary]")
    print(summary.to_string(index=False, formatters={
        "weighted_brier": "{:.8f}".format,
        "weighted_raw_score": "{:+.2f}".format,
        "weighted_delta_brier_vs_fixed": "{:+.8f}".format,
        "worst_delta_brier_vs_fixed": "{:+.8f}".format,
        "best_delta_brier_vs_fixed": "{:+.8f}".format,
    }))
    print("\nPromotion rule: dynamic gate should beat fixed alpha in weighted Brier and not rely on a single fold; otherwise keep the fixed gate.")
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
