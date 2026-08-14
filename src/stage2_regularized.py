from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .stage2 import ResidualProbabilityMLP, _brier_np, _run_validation, _stage2_dir
from .utils import save_json, seed_everything


def train_stage2(config: dict) -> dict:
    """Train a conservative residual probability model around the local prior.

    The local empirical probability is already a useful estimator. Stage 2 is
    therefore allowed to move away from it only when the latent/local features
    provide repeatable evidence. The training objective is

        data_loss + anchor_lambda * mean((p - p_local)^2)

    and epoch 0 (the untouched local prior) is treated as a valid checkpoint.
    If learning cannot beat the prior on 2023, the final model remains exactly
    the local prior instead of saving a worse epoch-1 model.
    """
    seed_everything(config["seed"])
    root = _stage2_dir(config)
    metadata_path = root / "metadata.json"
    required = [
        root / "train_features.npy",
        root / "train_target.npy",
        root / "val_features.npy",
        root / "val_target.npy",
        root / "val_global_index.npy",
        metadata_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage 2 dataset이 없습니다. 먼저 python scripts/build_stage2_dataset.py --config configs/default.yaml 를 실행하세요.\n"
            + "\n".join(missing)
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_x_np = np.load(root / "train_features.npy")
    train_y_np = np.load(root / "train_target.npy")
    val_x_np = np.load(root / "val_features.npy")
    val_y_np = np.load(root / "val_target.npy")
    val_global = np.load(root / "val_global_index.npy")

    cfg = config.get("stage2", {})
    prior_k = int(cfg.get("prior_k", max(metadata["k_values"])))
    prior_name = f"local_prob_k{prior_k}"
    if prior_name not in metadata["feature_names"]:
        raise ValueError(f"{prior_name}가 Stage2 dataset에 없습니다.")
    prior_index = metadata["feature_names"].index(prior_name)

    hidden_dims = [int(x) for x in cfg.get("hidden_dims", [64, 32])]
    dropout = float(cfg.get("dropout", 0.15))
    anchor_lambda = float(cfg.get("anchor_lambda", 5.0))
    if anchor_lambda < 0:
        raise ValueError("stage2.anchor_lambda는 0 이상이어야 합니다.")

    model = ResidualProbabilityMLP(
        input_dim=train_x_np.shape[1],
        prior_index=prior_index,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    train_ds = TensorDataset(torch.from_numpy(train_x_np), torch.from_numpy(train_y_np))
    val_ds = TensorDataset(torch.from_numpy(val_x_np), torch.from_numpy(val_y_np))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 4096)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("eval_batch_size", 8192)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    learning_rate = float(cfg.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(cfg.get("weight_decay", 1e-3)),
    )
    loss_name = str(cfg.get("loss", "brier")).lower()
    if loss_name not in {"brier", "bce"}:
        raise ValueError("stage2.loss는 brier 또는 bce여야 합니다.")

    epochs = int(cfg.get("epochs", 30))
    patience = int(cfg.get("patience", 5))
    min_delta = float(cfg.get("min_delta", 1e-7))
    checkpoint_path = root / "stage2_best.pt"

    local_prior_val = val_x_np[:, prior_index].astype(np.float32, copy=False)
    prior_brier = _brier_np(val_y_np, local_prior_val)
    mean_brier = _brier_np(
        val_y_np,
        np.full_like(val_y_np, float(metadata["train_target_mean"])),
    )

    # The zero-initialized residual head makes epoch 0 exactly equal to p_local.
    initial_brier, initial_pred = _run_validation(model, val_loader, dev)
    if abs(initial_brier - prior_brier) > 1e-6:
        raise RuntimeError(
            f"epoch-0 prediction이 local prior와 다릅니다: {initial_brier:.8f} vs {prior_brier:.8f}"
        )

    best_brier = initial_brier
    best_epoch = 0
    stale = 0
    history: list[dict] = [
        {
            "epoch": 0,
            "train_objective": None,
            "train_data_loss": None,
            "train_anchor_loss": 0.0,
            "val_brier": float(initial_brier),
            "val_mean_abs_correction": 0.0,
        }
    ]

    def save_checkpoint(epoch: int) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "input_dim": int(train_x_np.shape[1]),
                "prior_index": int(prior_index),
                "prior_name": prior_name,
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "anchor_lambda": anchor_lambda,
                "learning_rate": learning_rate,
                "feature_names": metadata["feature_names"],
                "best_epoch": int(epoch),
            },
            checkpoint_path,
        )

    save_checkpoint(0)
    print(
        f"epoch 00: val_brier={initial_brier:.8f} "
        f"(local prior; anchor_lambda={anchor_lambda:g}, lr={learning_rate:g})"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_objective = 0.0
        total_data = 0.0
        total_anchor = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"stage2 e{epoch}", leave=False)

        for xb, yb in bar:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            p = torch.sigmoid(logits)
            p0 = xb[:, prior_index].clamp(1e-4, 1.0 - 1e-4)

            if loss_name == "brier":
                data_loss = torch.mean(torch.square(p - yb))
            else:
                data_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)

            anchor_loss = torch.mean(torch.square(p - p0))
            loss = data_loss + anchor_lambda * anchor_loss
            loss.backward()
            optimizer.step()

            bs = len(xb)
            total_objective += float(loss.item()) * bs
            total_data += float(data_loss.item()) * bs
            total_anchor += float(anchor_loss.item()) * bs
            seen += bs
            bar.set_postfix(
                obj=f"{total_objective / max(seen, 1):.6f}",
                data=f"{total_data / max(seen, 1):.6f}",
                anchor=f"{total_anchor / max(seen, 1):.6f}",
            )

        train_objective = total_objective / max(seen, 1)
        train_data_loss = total_data / max(seen, 1)
        train_anchor_loss = total_anchor / max(seen, 1)
        val_brier, val_pred = _run_validation(model, val_loader, dev)
        mean_abs_correction = float(np.mean(np.abs(val_pred - local_prior_val)))

        history.append(
            {
                "epoch": epoch,
                "train_objective": train_objective,
                "train_data_loss": train_data_loss,
                "train_anchor_loss": train_anchor_loss,
                "val_brier": val_brier,
                "val_mean_abs_correction": mean_abs_correction,
            }
        )
        print(
            f"epoch {epoch:02d}: objective={train_objective:.8f}, "
            f"data={train_data_loss:.8f}, anchor={train_anchor_loss:.8f}, "
            f"val_brier={val_brier:.8f}, mean|delta_p|={mean_abs_correction:.6f}"
        )

        if val_brier < best_brier - min_delta:
            best_brier = val_brier
            best_epoch = epoch
            stale = 0
            save_checkpoint(epoch)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_brier, val_pred = _run_validation(model, val_loader, dev)
    correction = val_pred - local_prior_val

    result = {
        "best_epoch": int(best_epoch),
        "best_is_local_prior": bool(best_epoch == 0),
        "validation_brier": float(final_brier),
        "local_prior_brier": float(prior_brier),
        "train_mean_brier": float(mean_brier),
        "skill_vs_local_prior": float(1.0 - final_brier / prior_brier),
        "skill_vs_train_mean": float(1.0 - final_brier / mean_brier),
        "prior_feature": prior_name,
        "loss": loss_name,
        "anchor_lambda": anchor_lambda,
        "learning_rate": learning_rate,
        "mean_abs_correction": float(np.mean(np.abs(correction))),
        "std_correction": float(np.std(correction)),
        "max_abs_correction": float(np.max(np.abs(correction))),
        "device": str(dev),
        "history": history,
    }
    save_json(result, root / "stage2_metrics.json")

    pd.DataFrame(
        {
            "global_index": val_global,
            "target": val_y_np,
            "local_prior": local_prior_val,
            "stage2_probability": val_pred,
            "correction": correction,
        }
    ).to_csv(root / "stage2_val_predictions.csv", index=False)

    print("\n[Stage2 validation]")
    print(f"train mean baseline Brier : {mean_brier:.8f}")
    print(f"local prior Brier         : {prior_brier:.8f}")
    print(f"Stage2 Brier              : {final_brier:.8f}")
    print(f"best epoch                : {best_epoch}")
    print(f"mean |prob correction|    : {result['mean_abs_correction']:.8f}")
    print(f"skill vs local prior      : {result['skill_vs_local_prior']:+.6f}")
    print(f"skill vs train mean       : {result['skill_vs_train_mean']:+.6f}")
    return result
