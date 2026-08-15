from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm.auto import tqdm

from .data import load_frame, split_masks
from .stage2_mlp import _evaluate, _loader
from .utils import save_json, seed_everything


class BilinearProbabilityHead(nn.Module):
    """Frozen context/history latents with a learned cross-branch interaction."""

    def __init__(self, context_dim: int, history_dim: int, interaction_dim: int = 16):
        super().__init__()
        self.context_dim = int(context_dim)
        self.history_dim = int(history_dim)
        self.interaction_dim = int(interaction_dim)
        self.bilinear = nn.Bilinear(self.context_dim, self.history_dim, self.interaction_dim)
        self.output = nn.Linear(
            self.context_dim + self.history_dim + self.interaction_dim,
            1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        c = z[:, : self.context_dim]
        h = z[:, self.context_dim : self.context_dim + self.history_dim]
        g = self.bilinear(c, h)
        features = torch.cat([c, h, g], dim=1)
        return self.output(features).squeeze(-1)


def _load_split_latents(config: dict) -> tuple[np.ndarray, int, int]:
    latent_dir = Path(config["paths"]["output_dir"]) / "latents"
    context_path = latent_dir / "context.npy"
    history_path = latent_dir / "history.npy"
    if not context_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            "Stage 1 latent가 없습니다. 먼저 Stage1을 학습하거나 "
            "python scripts/stage1_cache.py pull 을 실행하세요."
        )

    context = np.load(context_path, mmap_mode="r")
    history = np.load(history_path, mmap_mode="r")
    if len(context) != len(history):
        raise ValueError("context/history latent row 수가 다릅니다.")
    if context.ndim != 2 or history.ndim != 2:
        raise ValueError("context/history latent는 2D array여야 합니다.")

    context_dim = int(context.shape[1])
    history_dim = int(history.shape[1])
    z = np.concatenate(
        [np.asarray(context, dtype=np.float32), np.asarray(history, dtype=np.float32)],
        axis=1,
    )
    return z, context_dim, history_dim


def train_stage2(config: dict) -> dict:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    z, context_dim, history_dim = _load_split_latents(config)
    if len(z) != len(frame):
        raise ValueError(f"latent rows={len(z):,}, frame rows={len(frame):,} 불일치")

    target_col = config["data"]["target_col"]
    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=np.float32)

    train_mask = split["train"]
    val_mask = split["val"]
    train_z_raw = z[train_mask]
    val_z_raw = z[val_mask]
    train_y = y[train_mask]
    val_y = y[val_mask]

    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_z_raw).astype(np.float32, copy=False)
    val_z = scaler.transform(val_z_raw).astype(np.float32, copy=False)

    cfg = config.get("stage2", {})
    interaction_dim = int(cfg.get("interaction_dim", 16))
    if interaction_dim <= 0:
        raise ValueError("stage2.interaction_dim must be positive")
    batch_size = int(cfg.get("batch_size", 8192))
    eval_batch_size = int(cfg.get("eval_batch_size", 16384))
    num_workers = int(cfg.get("num_workers", 0))
    threshold = float(cfg.get("threshold", 0.5))

    train_loader = _loader(train_z, train_y, batch_size, True, num_workers)
    val_loader = _loader(val_z, val_y, eval_batch_size, False, num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BilinearProbabilityHead(context_dim, history_dim, interaction_dim).to(device)

    train_mean = float(train_y.mean())
    base_logit = math.log(train_mean / (1.0 - train_mean))
    with torch.no_grad():
        model.bilinear.bias.zero_()
        model.output.weight.zero_()
        model.output.bias.fill_(base_logit)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.003)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = int(cfg.get("epochs", 30))
    patience = int(cfg.get("patience", 5))
    min_delta = float(cfg.get("min_delta", 1e-6))
    best_bce = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict] = []

    output_dir = Path(config["paths"]["output_dir"]) / "stage2_bilinear"
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture = (
        f"c({context_dim}) + h({history_dim}) + "
        f"Bilinear(c,h->{interaction_dim}) -> Linear({context_dim + history_dim + interaction_dim},1) -> sigmoid"
    )
    print(
        f"[Bilinear Stage2] train={len(train_z):,}, val={len(val_z):,}, "
        f"context_dim={context_dim}, history_dim={history_dim}, "
        f"interaction_dim={interaction_dim}, parameters={parameter_count:,}, device={device}"
    )
    print(f"[Bilinear Stage2] {architecture}; loss=BCE on raw 0/1 labels")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"bilinear e{epoch}", leave=False)
        for xb, yb in bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            bs = len(xb)
            loss_sum += float(loss.item()) * bs
            seen += bs
            bar.set_postfix(bce=f"{loss_sum / max(seen, 1):.6f}")

        train_bce = loss_sum / max(seen, 1)
        val_metrics, _ = _evaluate(model, val_loader, device, threshold)
        row = {
            "epoch": epoch,
            "train_bce": float(train_bce),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: train_bce={train_bce:.8f}, "
            f"val_bce={val_metrics['bce']:.8f}, "
            f"val_brier={val_metrics['brier']:.8f}, "
            f"acc={val_metrics['accuracy']:.5f}, "
            f"auc={val_metrics['auc']:.5f}, "
            f"p_std={val_metrics['prediction_std']:.5f}"
        )

        if val_metrics["bce"] < best_bce - min_delta:
            best_bce = val_metrics["bce"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Bilinear Stage2 best_state가 없습니다.")
    model.load_state_dict(best_state)
    final_metrics, val_pred = _evaluate(model, val_loader, device, threshold)

    baseline_pred = np.full_like(val_y, train_mean, dtype=np.float32)
    baseline_brier = float(np.mean(np.square(baseline_pred - val_y)))
    val_rate = float(val_y.mean())
    official_baseline_brier = val_rate * (1.0 - val_rate)
    official_score = max(
        0.0,
        100000.0 * (1.0 - final_metrics["brier"] / official_baseline_brier),
    )

    result = {
        "model": "BilinearProbabilityHead",
        "architecture": architecture,
        "context_dim": context_dim,
        "history_dim": history_dim,
        "interaction_dim": interaction_dim,
        "parameter_count": int(parameter_count),
        "train_rows": int(len(train_z)),
        "val_rows": int(len(val_z)),
        "best_epoch": int(best_epoch),
        "selection_metric": "validation BCE",
        "training_loss": "BCEWithLogitsLoss on raw 0/1 labels",
        "validation": final_metrics,
        "baseline": {
            "train_mean_probability": train_mean,
            "train_mean_brier": baseline_brier,
            "validation_rate": val_rate,
            "official_baseline_brier": official_baseline_brier,
            "official_style_score": official_score,
        },
        "history": history,
        "standardized_on_train_only": True,
        "device": str(device),
    }

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "context_dim": context_dim,
            "history_dim": history_dim,
            "interaction_dim": interaction_dim,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "threshold": threshold,
        },
        output_dir / "stage2_bilinear_best.pt",
    )
    save_json(result, output_dir / "metrics.json")

    val_global = np.flatnonzero(val_mask)
    pred_frame = pd.DataFrame(
        {
            "global_index": val_global,
            "target": val_y,
            "probability": val_pred,
            "predicted_class": (val_pred >= threshold).astype(np.int8),
        }
    )
    row_id_col = config["data"].get("row_id_col")
    if row_id_col and row_id_col in frame.columns:
        pred_frame.insert(1, row_id_col, frame.loc[val_mask, row_id_col].to_numpy())
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    print("\n[Bilinear Stage2 validation]")
    print(f"best epoch                 : {best_epoch}")
    print(f"validation BCE             : {final_metrics['bce']:.8f}")
    print(f"validation Brier           : {final_metrics['brier']:.8f}")
    print(f"validation accuracy        : {final_metrics['accuracy']:.6f}")
    print(f"validation AUC             : {final_metrics['auc']:.6f}")
    print(
        f"prediction mean / std      : {final_metrics['prediction_mean']:.6f} / "
        f"{final_metrics['prediction_std']:.6f}"
    )
    print(f"official baseline Brier    : {official_baseline_brier:.8f}")
    print(f"official-style score       : {official_score:.2f}")
    return result
