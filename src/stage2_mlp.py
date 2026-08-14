from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .data import load_frame, split_masks
from .knn_probability import _load_latents
from .utils import save_json, seed_everything


class MLPProbabilityHead(nn.Module):
    """Frozen Stage-1 latent -> nonlinear Bernoulli logit."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (64, 32)):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    model.eval()
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    bce_sum = 0.0
    rows = 0

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb_dev = yb.to(device, non_blocking=True)
            logits = model(xb)
            bce_sum += float(loss_fn(logits, yb_dev).item())
            rows += len(yb)
            logits_all.append(logits.cpu().numpy())
            y_all.append(yb.numpy())

    logits_np = np.concatenate(logits_all).astype(np.float32, copy=False)
    y_np = np.concatenate(y_all).astype(np.float32, copy=False)
    p = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -30.0, 30.0)))
    pred_class = (p >= threshold).astype(np.float32)

    metrics = {
        "bce": float(bce_sum / max(rows, 1)),
        "brier": float(np.mean(np.square(p - y_np))),
        "accuracy": float(np.mean(pred_class == y_np)),
        "auc": float(roc_auc_score(y_np, p)),
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
    }
    return metrics, p.astype(np.float32, copy=False)


def train_stage2(config: dict) -> dict:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    z = _load_latents(config)
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
    batch_size = int(cfg.get("batch_size", 8192))
    eval_batch_size = int(cfg.get("eval_batch_size", 16384))
    num_workers = int(cfg.get("num_workers", 0))
    threshold = float(cfg.get("threshold", 0.5))
    hidden_dims = tuple(int(x) for x in cfg.get("hidden_dims", [64, 32]))
    if len(hidden_dims) != 2:
        raise ValueError("stage2.hidden_dims must contain exactly two dimensions")

    train_loader = _loader(train_z, train_y, batch_size, True, num_workers)
    val_loader = _loader(val_z, val_y, eval_batch_size, False, num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPProbabilityHead(train_z.shape[1], hidden_dims).to(device)

    train_mean = float(train_y.mean())
    base_logit = math.log(train_mean / (1.0 - train_mean))
    with torch.no_grad():
        model.net[-1].bias.fill_(base_logit)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 0.001)),
        weight_decay=float(cfg.get("weight_decay", 0.0001)),
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

    output_dir = Path(config["paths"]["output_dir"]) / "stage2_mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[MLP Stage2] train={len(train_z):,}, val={len(val_z):,}, "
        f"latent_dim={train_z.shape[1]}, hidden={hidden_dims}, "
        f"parameters={parameter_count:,}, device={device}"
    )
    print(
        "[MLP Stage2] z -> Linear(32,64) -> GELU -> Linear(64,32) -> GELU "
        "-> Linear(32,1) -> sigmoid; loss=BCE on raw 0/1 labels"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"mlp e{epoch}", leave=False)
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
        raise RuntimeError("MLP Stage2 best_state가 없습니다.")
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
        "model": "MLPProbabilityHead",
        "architecture": "32 -> 64 -> GELU -> 32 -> GELU -> 1 -> sigmoid",
        "latent_dim": int(train_z.shape[1]),
        "hidden_dims": list(hidden_dims),
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
            "latent_dim": int(train_z.shape[1]),
            "hidden_dims": list(hidden_dims),
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "threshold": threshold,
        },
        output_dir / "stage2_mlp_best.pt",
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

    print("\n[MLP Stage2 validation]")
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
