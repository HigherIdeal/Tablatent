from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_2025_proxy_validation as proxy_core
import run_gated_r_specialist_suite as gated_core
import run_regime_aware_stable_dynamics as regime_core
import run_stable_player_dynamics as dyn_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


VARIANTS = ("base", "lag1", "transformer", "transformer_lag1")


class TinyTemporalTransformer(nn.Module):
    """Small future-query Transformer for causal pitcher-month state encoding."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        lookback: int,
        target_dim: int,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.lookback = int(lookback)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback + 1, d_model))
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, target_dim))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = self.input_proj(x) + self.pos[:, : self.lookback]
        q = self.query.expand(b, -1, -1) + self.pos[:, self.lookback : self.lookback + 1]
        tokens = torch.cat([h, q], dim=1)

        idx = torch.arange(self.lookback, device=x.device).unsqueeze(0)
        pad_hist = idx < (self.lookback - lengths).unsqueeze(1)
        pad = torch.cat(
            [pad_hist, torch.zeros((b, 1), dtype=torch.bool, device=x.device)],
            dim=1,
        )
        z = self.encoder(tokens, src_key_padding_mask=pad)
        return self.norm(z[:, -1])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, lengths))


def add_relative_time_features(content: np.ndarray, hist_times: np.ndarray, query_time: int) -> np.ndarray:
    """Append age-to-query and observation gap; no labels/current state."""
    age = np.asarray(query_time - hist_times, dtype=np.float32) / 12.0
    if len(hist_times) == 0:
        gap = np.empty(0, dtype=np.float32)
    else:
        previous = np.r_[hist_times[0], hist_times[:-1]]
        gap = np.asarray(hist_times - previous, dtype=np.float32) / 12.0
    return np.concatenate(
        [
            content.astype(np.float32, copy=False),
            np.clip(age[:, None], 0.0, 10.0),
            np.clip(gap[:, None], 0.0, 5.0),
        ],
        axis=1,
    )


def make_transformer_selfsup(
    monthly: pd.DataFrame,
    scaled: np.ndarray,
    max_target_time: int,
    lookback: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Past-only history -> multi-horizon future stable state.

    Targets are used only for self-supervised training and are constrained to
    observations at or before the fold cutoff.
    """
    stable_dim = len(dyn_core.STABLE_FEATURES)
    input_dim = scaled.shape[1] + 2
    xs: list[np.ndarray] = []
    lengths: list[int] = []
    ys: list[np.ndarray] = []
    times: list[int] = []

    for _, idx in monthly.groupby("_pitcher", sort=False, observed=True).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        ptimes = monthly.loc[pos, "_time"].to_numpy(np.int64)
        for j in range(1, len(pos)):
            target_pos = pos[j]
            target_time = int(ptimes[j])
            if target_time > max_target_time:
                continue

            start = max(0, j - lookback)
            hist_pos = pos[start:j]
            hist_times = ptimes[start:j]
            hist = add_relative_time_features(scaled[hist_pos], hist_times, target_time)
            if len(hist) == 0:
                continue

            padded = np.zeros((lookback, input_dim), dtype=np.float32)
            padded[-len(hist) :] = hist

            future_j = []
            for k in range(j, min(j + 3, len(pos))):
                if int(ptimes[k]) <= max_target_time:
                    future_j.append(pos[k])
            if not future_j:
                continue
            future = scaled[np.asarray(future_j, dtype=int), :stable_dim]
            next1 = future[0]
            mean3 = future.mean(axis=0)
            delta1 = next1 - scaled[hist_pos[-1], :stable_dim]
            vol3 = future.std(axis=0)
            target = np.concatenate([next1, mean3, delta1, vol3]).astype(np.float32)

            xs.append(padded)
            lengths.append(len(hist))
            ys.append(target)
            times.append(target_time)

    if not xs:
        raise RuntimeError("no Transformer self-supervised samples before fold cutoff")
    return (
        torch.from_numpy(np.stack(xs)),
        torch.as_tensor(lengths, dtype=torch.long),
        torch.from_numpy(np.stack(ys)),
        torch.as_tensor(times, dtype=torch.long),
    )


def train_transformer(
    x: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    times: torch.Tensor,
    *,
    d_model: int,
    nhead: int,
    layers: int,
    ff_dim: int,
    dropout: float,
    lookback: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    seed: int,
) -> tuple[TinyTemporalTransformer, list[dict[str, float]]]:
    dyn_core.set_seed(seed)
    unique_times = np.asarray(sorted(set(times.tolist())), dtype=int)
    split_idx = max(1, int(len(unique_times) * 0.8))
    split = unique_times[split_idx] if split_idx < len(unique_times) else unique_times[-1] + 1
    train_mask = times < split
    val_mask = times >= split
    if not bool(train_mask.any()) or not bool(val_mask.any()):
        train_mask = torch.ones_like(times, dtype=torch.bool)
        val_mask = torch.zeros_like(times, dtype=torch.bool)

    model = TinyTemporalTransformer(
        input_dim=x.shape[2],
        d_model=d_model,
        nhead=nhead,
        layers=layers,
        ff_dim=ff_dim,
        dropout=dropout,
        lookback=lookback,
        target_dim=y.shape[1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    train_loader = DataLoader(
        TensorDataset(x[train_mask], lengths[train_mask], y[train_mask]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = (
        DataLoader(TensorDataset(x[val_mask], lengths[val_mask], y[val_mask]), batch_size=batch_size)
        if bool(val_mask.any())
        else None
    )

    best = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, lb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            count += len(yb)
        train_loss = total / max(count, 1)

        if val_loader is None:
            val_loss = train_loss
        else:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for xb, lb, yb in val_loader:
                    xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
                    loss = loss_fn(model(xb, lb), yb)
                    total += float(loss.item()) * len(yb)
                    count += len(yb)
            val_loss = total / max(count, 1)

        history.append(
            {
                "epoch": epoch,
                "train_multihorizon_loss": train_loss,
                "valid_multihorizon_loss": val_loss,
            }
        )
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best is not None:
        model.load_state_dict(best)
    return model.eval(), history


def build_transformer_state(
    monthly: pd.DataFrame,
    scaled: np.ndarray,
    model: TinyTemporalTransformer,
    *,
    d_model: int,
    lookback: int,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    n = len(monthly)
    input_dim = scaled.shape[1] + 2
    padded = np.zeros((n, lookback, input_dim), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    lag = np.full((n, len(dyn_core.STABLE_FEATURES)), np.nan, dtype=np.float32)

    for _, idx in monthly.groupby("_pitcher", sort=False, observed=True).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        ptimes = monthly.loc[pos, "_time"].to_numpy(np.int64)
        raw = monthly.loc[pos, dyn_core.STABLE_FEATURES].to_numpy(np.float32)
        for j, row_pos in enumerate(pos):
            if j == 0:
                continue
            start = max(0, j - lookback)
            hist_pos = pos[start:j]
            hist_times = ptimes[start:j]
            hist = add_relative_time_features(scaled[hist_pos], hist_times, int(ptimes[j]))
            padded[row_pos, -len(hist) :] = hist
            lengths[row_pos] = len(hist)
            lag[row_pos] = raw[j - 1]

    emb = np.zeros((n, d_model), dtype=np.float32)
    valid = np.flatnonzero(lengths > 0)
    with torch.no_grad():
        for start in range(0, len(valid), batch_size):
            ids = valid[start : start + batch_size]
            xb = torch.from_numpy(padded[ids]).to(device)
            lb = torch.from_numpy(lengths[ids]).to(device)
            emb[ids] = model.encode(xb, lb).cpu().numpy().astype(np.float32)

    tf_cols = [f"dyn_tf_{i:02d}" for i in range(d_model)]
    lag_cols = [f"dyn_lag1_{name}" for name in dyn_core.STABLE_FEATURES]
    state = monthly[["_pitcher", "_time"]].copy()
    state[tf_cols] = emb
    state[lag_cols] = lag
    state["dyn_history_months"] = 0.0
    state["dyn_known"] = (lengths > 0).astype(np.float32)
    for _, idx in monthly.groupby("_pitcher", sort=False, observed=True).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        state.loc[pos, "dyn_history_months"] = np.arange(len(pos), dtype=np.float32)
    return state, tf_cols, lag_cols


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Tiny future-query Transformer ablation inside the current "
            "full_raw + recent_raw + R-fast regime-aware architecture."
        )
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--alpha-recent", type=float, default=0.20)
    p.add_argument("--beta-r", type=float, default=0.10)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--thread-count", type=int, default=6)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--torch-device", default="auto")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lookback", type=int, default=36)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--output-dir", default="outputs/regime_aware_stable_transformer")
    args = p.parse_args()

    if min(
        args.iterations,
        args.d_model,
        args.heads,
        args.layers,
        args.ff_dim,
        args.lookback,
        args.epochs,
        args.batch_size,
        args.patience,
    ) <= 0:
        raise ValueError("positive hyperparameters required")
    if args.d_model % args.heads != 0:
        raise ValueError("d-model must be divisible by heads")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("dropout must be in [0,1)")
    if not (0.0 <= args.alpha_recent <= 1.0 and 0.0 <= args.beta_r <= 1.0):
        raise ValueError("alpha/beta must be in [0,1]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    dyn_core.set_seed(seed)
    torch_dev = dyn_core.torch_device(args.torch_device)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
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
    history_dir = outdir / "transformer_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    print("[Regime-Aware Stable Transformer]")
    print(f"  rows={len(frame):,} pitcher_months={len(monthly):,}")
    print(
        f"  torch={torch.__version__} device={torch_dev} "
        f"d_model={args.d_model} heads={args.heads} layers={args.layers} lookback={args.lookback}"
    )
    print("  future-query Transformer; multi-horizon self-supervision; no control_success")
    print("  temporal state is injected ONLY into recent_raw; full_raw and R-fast stay fixed")

    rows: list[dict[str, object]] = []
    fold_meta: list[dict[str, object]] = []
    fold_weights = {spec.name: float(spec.weight) for spec in proxy_core.DEFAULT_FOLDS}

    for fold_index, spec in enumerate(proxy_core.DEFAULT_FOLDS):
        cutoff_time = regime_core._fold_cutoff_time(monthly, spec)
        valid_max_time = regime_core._fold_validation_max_time(monthly, spec)

        scaler_monthly = monthly.loc[monthly["_time"].le(cutoff_time)].copy()
        scaler = dyn_core.fit_scaler(scaler_monthly, input_cols)
        scaled_all = dyn_core.scale(monthly, input_cols, scaler)
        x, lengths, y_self, times = make_transformer_selfsup(monthly, scaled_all, cutoff_time, args.lookback)
        model, history = train_transformer(
            x,
            lengths,
            y_self,
            times,
            d_model=args.d_model,
            nhead=args.heads,
            layers=args.layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            lookback=args.lookback,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=torch_dev,
            seed=seed + 500 + fold_index,
        )
        pd.DataFrame(history).to_csv(history_dir / f"{spec.name}.csv", index=False)

        use_monthly = monthly.loc[monthly["_time"].le(valid_max_time)].copy().reset_index(drop=True)
        use_scaled = dyn_core.scale(use_monthly, input_cols, scaler)
        state, tf_cols, lag_cols = build_transformer_state(
            use_monthly,
            use_scaled,
            model,
            d_model=args.d_model,
            lookback=args.lookback,
            batch_size=args.batch_size,
            device=torch_dev,
        )
        fold_frame = dyn_core.attach_state(frame, state)
        recent_mask, full_mask, valid_mask = proxy_core.fold_masks(fold_frame, spec, season_col, "game_month")
        valid = fold_frame.loc[valid_mask].copy()
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt = regime_core._token(valid["game_type"]).to_numpy()
        is_r = gt == "R"
        is_f = gt == "F"
        recent_r_mask = recent_mask & regime_core._token(fold_frame["game_type"]).eq("R")

        print(
            f"\n[Fold {spec.name}] valid={len(valid):,} rate={y.mean():.6f} "
            f"TF_samples={len(x):,} epochs={len(history)} "
            f"best_selfsup={min(h['valid_multihorizon_loss'] for h in history):.6f}"
        )

        p_full = regime_core._fit_one(
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
        p_r_fast = regime_core._fit_one(
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
            "transformer": list(base_features) + list(tf_cols) + ["dyn_history_months", "dyn_known"],
            "transformer_lag1": list(base_features)
            + list(tf_cols)
            + list(lag_cols)
            + ["dyn_history_months", "dyn_known"],
        }

        final_predictions: dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            p_recent = regime_core._fit_one(
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
            final_predictions[variant] = gated_core.gated_prediction(
                p_old=p_full,
                p_recent=p_recent,
                p_specialist=p_r_fast,
                is_r=is_r,
                alpha_recent=args.alpha_recent,
                beta_r=args.beta_r,
            )

        base_metric = probability_metrics(y, final_predictions["base"])
        for variant in VARIANTS:
            pred = final_predictions[variant]
            metric = probability_metrics(y, pred)
            delta = float(metric["brier"] - base_metric["brier"])
            row = {
                "fold": spec.name,
                "weight": float(spec.weight),
                "variant": variant,
                "brier": float(metric["brier"]),
                "raw_score": float(metric["raw_score"]),
                "delta_brier_vs_base": delta,
                "r_brier": regime_core._subset_brier(y, pred, is_r),
                "f_brier": regime_core._subset_brier(y, pred, is_f),
            }
            rows.append(row)
            print(
                f"  {variant:<17s} brier={row['brier']:.8f} raw={row['raw_score']:+.2f} "
                f"dBrier={delta:+.8f} R={row['r_brier']:.8f} F={row['f_brier']:.8f}"
            )

        fold_meta.append(
            {
                "fold": spec.name,
                "weight": float(spec.weight),
                "cutoff_time": int(cutoff_time),
                "validation_max_time": int(valid_max_time),
                "valid_rows": int(len(valid)),
                "transformer_samples": int(len(x)),
                "transformer_epochs": int(len(history)),
                "best_selfsup": float(min(h["valid_multihorizon_loss"] for h in history)),
            }
        )
        pd.DataFrame(rows).to_csv(outdir / "fold_metrics_checkpoint.csv", index=False)

        del model, state, fold_frame, valid, x, lengths, y_self, times
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    summary = regime_core._weighted_summary(results, fold_weights)
    results.to_csv(outdir / "fold_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(fold_meta).to_csv(outdir / "fold_meta.csv", index=False)
    save_json(
        {
            "experiment": "regime_aware_stable_transformer",
            "architecture": "full_raw + recent_raw(+tiny transformer) + R-fast",
            "temporal_injection": "recent expert only",
            "self_supervision": "next1 + next3mean + next1delta + next3volatility",
            "d_model": int(args.d_model),
            "heads": int(args.heads),
            "layers": int(args.layers),
            "ff_dim": int(args.ff_dim),
            "dropout": float(args.dropout),
            "lookback": int(args.lookback),
            "iterations": int(args.iterations),
            "alpha_recent": float(args.alpha_recent),
            "beta_r": float(args.beta_r),
            "stable_features": list(dyn_core.STABLE_FEATURES),
            "guardrails": [
                "Transformer never sees control_success",
                "game_type never enters Transformer",
                "month-t state uses only months strictly before t",
                "scaler/Transformer training stops at each proxy-fold cutoff",
                "full_raw and R-fast branches are fixed across variants",
            ],
            "invariant_check": invariant_check,
        },
        outdir / "metadata.json",
    )

    print("\n[Summary]")
    print(
        summary.to_string(
            index=False,
            formatters={
                "weighted_brier": "{:.8f}".format,
                "weighted_raw_score": "{:+.2f}".format,
                "weighted_delta_brier_vs_base": "{:+.8f}".format,
                "worst_delta_brier_vs_base": "{:+.8f}".format,
                "best_delta_brier_vs_base": "{:+.8f}".format,
            },
        )
    )
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
