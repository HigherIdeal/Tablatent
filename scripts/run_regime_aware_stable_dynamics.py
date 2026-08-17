from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_2025_proxy_validation as proxy_core
import run_gated_r_specialist_suite as gated_core
import run_stable_player_dynamics as dyn_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


VARIANTS = ("base", "lag1", "gru", "gru_lag1")


def _token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def _fold_cutoff_time(monthly: pd.DataFrame, spec: proxy_core.FoldSpec) -> int:
    if spec.kind == "season_forward":
        candidates = monthly.loc[monthly["season"].le(2023), "_time"]
    elif spec.kind == "within_2024":
        if spec.cutoff_month is None:
            raise ValueError(f"within-2024 fold missing cutoff_month: {spec}")
        candidates = monthly.loc[
            monthly["season"].lt(2024)
            | (monthly["season"].eq(2024) & monthly["game_month"].le(spec.cutoff_month)),
            "_time",
        ]
    else:
        raise ValueError(f"unknown fold kind: {spec.kind}")
    if candidates.empty:
        raise RuntimeError(f"no GRU training months for {spec.name}")
    return int(candidates.max())


def _fold_validation_max_time(monthly: pd.DataFrame, spec: proxy_core.FoldSpec) -> int:
    if spec.kind == "season_forward":
        candidates = monthly.loc[monthly["season"].eq(2024), "_time"]
    elif spec.kind == "within_2024":
        if spec.valid_month_start is None:
            raise ValueError(f"within-2024 fold missing valid_month_start: {spec}")
        mask = monthly["season"].eq(2024) & monthly["game_month"].ge(spec.valid_month_start)
        if spec.valid_month_end is not None:
            mask &= monthly["game_month"].le(spec.valid_month_end)
        candidates = monthly.loc[mask, "_time"]
    else:
        raise ValueError(f"unknown fold kind: {spec.kind}")
    if candidates.empty:
        raise RuntimeError(f"no GRU validation months for {spec.name}")
    return int(candidates.max())


def make_selfsup_tensors_until(
    monthly: pd.DataFrame,
    scaled: np.ndarray,
    max_target_time: int,
    lookback: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_dim = len(dyn_core.STABLE_FEATURES)
    xs: list[np.ndarray] = []
    lengths: list[int] = []
    ys: list[np.ndarray] = []
    times: list[int] = []

    for _, idx in monthly.groupby("_pitcher", sort=False, observed=True).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        for j in range(1, len(pos)):
            target_pos = pos[j]
            target_time = int(monthly.loc[target_pos, "_time"])
            if target_time > max_target_time:
                continue
            start = max(0, j - lookback)
            history = scaled[pos[start:j]]
            if len(history) == 0:
                continue
            padded = np.zeros((lookback, scaled.shape[1]), dtype=np.float32)
            padded[: len(history)] = history
            xs.append(padded)
            lengths.append(len(history))
            ys.append(scaled[target_pos, :target_dim])
            times.append(target_time)

    if not xs:
        raise RuntimeError("no self-supervised GRU samples before fold cutoff")
    return (
        torch.from_numpy(np.stack(xs)),
        torch.as_tensor(lengths, dtype=torch.long),
        torch.from_numpy(np.stack(ys)),
        torch.as_tensor(times, dtype=torch.long),
    )


def _fit_one(
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
    predictions, _ = gated_core._fit_prefixes(
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
    return predictions[iterations]


def _subset_brier(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean((y[mask] - p[mask]) ** 2))


def _weighted_summary(results: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, group in results.groupby("variant", sort=False):
        w = np.asarray([weights[str(name)] for name in group["fold"]], dtype=np.float64)
        w /= w.sum()
        brier = group["brier"].to_numpy(np.float64)
        raw = group["raw_score"].to_numpy(np.float64)
        delta = group["delta_brier_vs_base"].to_numpy(np.float64)
        rows.append(
            {
                "variant": str(variant),
                "weighted_brier": float(np.dot(w, brier)),
                "weighted_raw_score": float(np.dot(w, raw)),
                "weighted_delta_brier_vs_base": float(np.dot(w, delta)),
                "worst_delta_brier_vs_base": float(delta.max()),
                "best_delta_brier_vs_base": float(delta.min()),
                "improved_folds": int(np.count_nonzero(delta < 0.0)),
                "folds": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["weighted_brier", "worst_delta_brier_vs_base"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add the causal stable-player GRU only to the recent expert of the current "
            "full-history + recent + R-fast gated system, and evaluate on the established "
            "2025 proxy folds. The full expert and R-fast specialist are held fixed."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--alpha-recent", type=float, default=0.20)
    parser.add_argument("--beta-r", type=float, default=0.10)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=6)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/regime_aware_stable_dynamics")
    args = parser.parse_args()

    if args.iterations <= 0 or args.hidden_dim <= 0 or args.lookback <= 0:
        raise ValueError("iterations/hidden-dim/lookback must be positive")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs/batch-size/patience must be positive")
    if not (0.0 <= args.alpha_recent <= 1.0):
        raise ValueError("alpha-recent must be in [0,1]")
    if not (0.0 <= args.beta_r <= 1.0):
        raise ValueError("beta-r must be in [0,1]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    dyn_core.set_seed(seed)
    torch_dev = dyn_core.torch_device(args.torch_device)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    if "pitcher_id" not in frame.columns:
        raise ValueError("pitcher_id is required for temporal state")
    sort_cols = [season_col, "game_month"] + ([row_id_col] if row_id_col in frame.columns else [])
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    monthly = dyn_core.build_monthly(frame)

    base_features = recent_core.feature_set("recent_raw_game_type")
    r_fast_features = gated_core._feature_sets(base_features)["r_fast"]
    input_cols = dyn_core.STABLE_FEATURES + dyn_core.AUX_FEATURES
    devices = gated_core.parse_devices(args.devices)
    cb_device = devices[0] if args.task_type == "GPU" else "CPU"

    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    histories_dir = outdir / "gru_history"
    histories_dir.mkdir(parents=True, exist_ok=True)

    print("[Regime-Aware Stable Dynamics]")
    print(f"  rows={len(frame):,} pitcher_months={len(monthly):,}")
    print(f"  torch={torch.__version__} device={torch_dev} stable={len(dyn_core.STABLE_FEATURES)} hidden={args.hidden_dim}")
    print(f"  CatBoost iterations={args.iterations} alpha_recent={args.alpha_recent:.3f} beta_r={args.beta_r:.3f}")
    print("  temporal state is injected ONLY into recent_raw; full_raw and R-fast remain unchanged")
    print("  GRU never sees control_success; each month embedding uses only strictly earlier months")

    rows: list[dict[str, object]] = []
    fold_meta: list[dict[str, object]] = []
    fold_weights = {spec.name: float(spec.weight) for spec in proxy_core.DEFAULT_FOLDS}

    for fold_index, spec in enumerate(proxy_core.DEFAULT_FOLDS):
        cutoff_time = _fold_cutoff_time(monthly, spec)
        valid_max_time = _fold_validation_max_time(monthly, spec)
        scaler_monthly = monthly.loc[monthly["_time"].le(cutoff_time)].copy()
        scaler = dyn_core.fit_scaler(scaler_monthly, input_cols)
        scaled_all = dyn_core.scale(monthly, input_cols, scaler)
        x, lengths, y_self, times = make_selfsup_tensors_until(
            monthly, scaled_all, cutoff_time, args.lookback
        )
        model, history = dyn_core.train_gru(
            x=x,
            lengths=lengths,
            y=y_self,
            times=times,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=torch_dev,
            seed=seed + 100 + fold_index,
        )
        pd.DataFrame(history).to_csv(histories_dir / f"{spec.name}.csv", index=False)

        use_monthly = monthly.loc[monthly["_time"].le(valid_max_time)].copy().reset_index(drop=True)
        use_scaled = dyn_core.scale(use_monthly, input_cols, scaler)
        state, dyn_cols, lag_cols = dyn_core.build_causal_state(
            monthly=use_monthly,
            scaled=use_scaled,
            model=model,
            hidden_dim=args.hidden_dim,
            lookback=args.lookback,
            batch_size=args.batch_size,
            device=torch_dev,
        )
        fold_frame = dyn_core.attach_state(frame, state)
        recent_mask, full_mask, valid_mask = proxy_core.fold_masks(
            fold_frame, spec, season_col, "game_month"
        )
        valid = fold_frame.loc[valid_mask].copy()
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt = _token(valid["game_type"]).to_numpy()
        is_r = gt == "R"
        is_f = gt == "F"
        recent_r_mask = recent_mask & _token(fold_frame["game_type"]).eq("R")

        print(
            f"\n[Fold {spec.name}] cutoff_time={cutoff_time} valid={len(valid):,} "
            f"rate={y.mean():.6f} GRU_samples={len(x):,} epochs={len(history)} "
            f"best_selfsup={min(h['valid_mse'] for h in history):.6f}"
        )
        print(
            f"  train full={int(full_mask.sum()):,} recent={int(recent_mask.sum()):,} "
            f"recentR={int(recent_r_mask.sum()):,} Rvalid={int(is_r.sum()):,} Fvalid={int(is_f.sum()):,}"
        )

        # Hold the current regime-aware architecture fixed except for the recent expert.
        p_full = _fit_one(
            train=fold_frame.loc[full_mask].copy(),
            valid=valid,
            features=base_features,
            target_col=target_col,
            config=config,
            iterations=args.iterations,
            task_type=args.task_type,
            device=cb_device,
            verbose=args.verbose,
            thread_count=args.thread_count,
        )
        p_r_fast = _fit_one(
            train=fold_frame.loc[recent_r_mask].copy(),
            valid=valid,
            features=r_fast_features,
            target_col=target_col,
            config=config,
            iterations=args.iterations,
            task_type=args.task_type,
            device=cb_device,
            verbose=args.verbose,
            thread_count=args.thread_count,
        )

        recent_feature_sets = {
            "base": list(base_features),
            "lag1": list(base_features) + list(lag_cols) + ["dyn_history_months", "dyn_known"],
            "gru": list(base_features) + list(dyn_cols) + ["dyn_history_months", "dyn_known"],
            "gru_lag1": list(base_features)
            + list(dyn_cols)
            + list(lag_cols)
            + ["dyn_history_months", "dyn_known"],
        }

        variant_predictions: dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            p_recent = _fit_one(
                train=fold_frame.loc[recent_mask].copy(),
                valid=valid,
                features=recent_feature_sets[variant],
                target_col=target_col,
                config=config,
                iterations=args.iterations,
                task_type=args.task_type,
                device=cb_device,
                verbose=args.verbose,
                thread_count=args.thread_count,
            )
            final = gated_core.gated_prediction(
                p_old=p_full,
                p_recent=p_recent,
                p_specialist=p_r_fast,
                is_r=is_r,
                alpha_recent=args.alpha_recent,
                beta_r=args.beta_r,
            )
            variant_predictions[variant] = final

        base_metric = probability_metrics(y, variant_predictions["base"])
        for variant in VARIANTS:
            pred = variant_predictions[variant]
            metric = probability_metrics(y, pred)
            delta = float(metric["brier"] - base_metric["brier"])
            row = {
                "fold": spec.name,
                "weight": float(spec.weight),
                "variant": variant,
                "brier": float(metric["brier"]),
                "raw_score": float(metric["raw_score"]),
                "delta_brier_vs_base": delta,
                "r_brier": _subset_brier(y, pred, is_r),
                "f_brier": _subset_brier(y, pred, is_f),
                "prediction_mean": float(metric["prediction_mean"]),
                "prediction_std": float(metric["prediction_std"]),
            }
            rows.append(row)
            print(
                f"  {variant:<8s} brier={row['brier']:.8f} raw={row['raw_score']:+.2f} "
                f"dBrier={delta:+.8f} R={row['r_brier']:.8f} F={row['f_brier']:.8f}"
            )

        fold_meta.append(
            {
                "fold": spec.name,
                "weight": float(spec.weight),
                "cutoff_time": int(cutoff_time),
                "validation_max_time": int(valid_max_time),
                "valid_rows": int(len(valid)),
                "target_rate": float(y.mean()),
                "gru_samples": int(len(x)),
                "gru_epochs": int(len(history)),
                "gru_best_selfsup_mse": float(min(h["valid_mse"] for h in history)),
                "full_train_rows": int(full_mask.sum()),
                "recent_train_rows": int(recent_mask.sum()),
                "recent_r_train_rows": int(recent_r_mask.sum()),
            }
        )

        pd.DataFrame(rows).to_csv(outdir / "fold_metrics_checkpoint.csv", index=False)
        del model, state, fold_frame, valid, x, lengths, y_self, times
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    summary = _weighted_summary(results, fold_weights)
    results.to_csv(outdir / "fold_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(fold_meta).to_csv(outdir / "fold_meta.csv", index=False)

    metadata = {
        "experiment": "regime_aware_stable_dynamics",
        "architecture": "full_raw + recent_raw(+temporal ablation) + R-fast",
        "temporal_injection": "recent expert only",
        "iterations": int(args.iterations),
        "alpha_recent": float(args.alpha_recent),
        "beta_r": float(args.beta_r),
        "stable_features": list(dyn_core.STABLE_FEATURES),
        "aux_features": list(dyn_core.AUX_FEATURES),
        "hidden_dim": int(args.hidden_dim),
        "lookback": int(args.lookback),
        "fold_weights": fold_weights,
        "invariant_check": invariant_check,
        "guardrails": [
            "GRU never sees control_success",
            "embedding for month t uses only months strictly before t",
            "GRU/scaler fit only through each proxy fold training cutoff",
            "full_raw and R-fast are unchanged across temporal variants",
            "fixed alpha/beta; no ensemble-weight tuning on these folds",
        ],
    }
    save_json(outdir / "metadata.json", metadata)

    print("\n[Summary]")
    print(summary.to_string(index=False, formatters={
        "weighted_brier": "{:.8f}".format,
        "weighted_raw_score": "{:+.2f}".format,
        "weighted_delta_brier_vs_base": "{:+.8f}".format,
        "worst_delta_brier_vs_base": "{:+.8f}".format,
        "best_delta_brier_vs_base": "{:+.8f}".format,
    }))
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
