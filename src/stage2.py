from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .data import load_frame, split_masks
from .knn_probability import _load_latents, _make_faiss_index
from .utils import ensure_output_dirs, save_json, seed_everything


def _stage2_dir(config: dict) -> Path:
    path = Path(config["paths"]["output_dir"]) / "stage2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _local_feature_names(ks: list[int]) -> list[str]:
    names: list[str] = []
    for k in ks:
        names.extend(
            [
                f"local_prob_k{k}",
                f"local_effective_n_k{k}",
                f"local_radius_k{k}",
            ]
        )
    return names


def _query_local_features(
    index,
    query_z: np.ndarray,
    pool_y: np.ndarray,
    ks: list[int],
    prior_strength: float,
    prior_prob: float,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """
    Similar latent states define a local empirical Bernoulli distribution.

    For each k we use an adaptive Gaussian kernel whose bandwidth is the
    distance to the k-th neighbor. The local success probability is then
    shrunk toward a pool-only global prior:

        p = (sum(w*y) + alpha*p0) / (sum(w) + alpha)

    effective_n and neighborhood radius are retained so Stage 2 can learn
    how much confidence to place in the local probability.
    """
    max_k = max(ks)
    out = np.empty((len(query_z), 3 * len(ks)), dtype=np.float32)

    for start in tqdm(range(0, len(query_z), batch_size), desc=desc):
        stop = min(start + batch_size, len(query_z))
        d2, nn_idx = index.search(
            np.ascontiguousarray(query_z[start:stop], dtype=np.float32),
            max_k,
        )
        if np.any(nn_idx < 0):
            raise RuntimeError(
                "FAISS가 충분한 이웃을 찾지 못했습니다. nprobe를 키우거나 index_type=flat을 사용하세요."
            )

        nn_y = pool_y[nn_idx].astype(np.float32, copy=False)
        col = 0
        for k in ks:
            dk = np.maximum(d2[:, :k], 0.0).astype(np.float32, copy=False)
            yk = nn_y[:, :k]

            # FAISS L2 returns squared distances. The k-th squared distance
            # is an adaptive bandwidth for each query.
            bandwidth2 = np.maximum(dk[:, -1], 1e-6)
            weights = np.exp(-0.5 * dk / bandwidth2[:, None]).astype(np.float32)
            sum_w = weights.sum(axis=1)
            sum_w2 = np.square(weights).sum(axis=1)
            weighted_success = (weights * yk).sum(axis=1)

            local_prob = (
                weighted_success + prior_strength * float(prior_prob)
            ) / (sum_w + prior_strength)
            effective_n = np.square(sum_w) / np.maximum(sum_w2, 1e-8)
            radius = np.sqrt(np.maximum(dk[:, -1], 0.0))

            out[start:stop, col] = local_prob
            out[start:stop, col + 1] = effective_n
            out[start:stop, col + 2] = radius
            col += 3

    return out


def _save_split(
    root: Path,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    global_index: np.ndarray,
) -> None:
    np.save(root / f"{name}_features.npy", np.asarray(x, dtype=np.float32))
    np.save(root / f"{name}_target.npy", np.asarray(y, dtype=np.float32))
    np.save(root / f"{name}_global_index.npy", np.asarray(global_index, dtype=np.int64))


def build_stage2_dataset(config: dict, include_test: bool = False) -> dict:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    z = _load_latents(config)
    if len(z) != len(frame):
        raise ValueError(f"latent rows={len(z):,}, frame rows={len(frame):,} 불일치")

    cfg = config.get("stage2_dataset", {})
    ks = sorted({int(k) for k in cfg.get("k_values", [100, 500, 1000])})
    if not ks or min(ks) <= 0:
        raise ValueError("stage2_dataset.k_values는 양의 정수여야 합니다.")

    folds = int(cfg.get("crossfit_folds", 5))
    if folds < 2:
        raise ValueError("crossfit_folds는 2 이상이어야 합니다.")
    prior_strength = float(cfg.get("prior_strength", 50.0))
    if prior_strength < 0:
        raise ValueError("prior_strength는 0 이상이어야 합니다.")
    batch_size = int(cfg.get("query_batch_size", 2048))

    target_col = config["data"]["target_col"]
    y_all = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=np.float32)

    train_mask = split["train"]
    val_mask = split["val"]
    test_mask = split["test"]
    train_global = np.flatnonzero(train_mask)
    val_global = np.flatnonzero(val_mask)
    test_global = np.flatnonzero(test_mask)

    # Geometry is defined only by Stage 1 training seasons.
    scaler = StandardScaler()
    train_z = scaler.fit_transform(z[train_mask]).astype(np.float32, copy=False)
    val_z = scaler.transform(z[val_mask]).astype(np.float32, copy=False)
    test_z = scaler.transform(z[test_mask]).astype(np.float32, copy=False)
    train_y = y_all[train_mask]
    val_y = y_all[val_mask]
    test_y = y_all[test_mask]

    # Cross-fitted local statistics: a training row never uses its own fold's
    # target labels when constructing local probability features.
    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(train_z))
    fold_id = np.empty(len(train_z), dtype=np.int16)
    fold_id[order] = np.arange(len(train_z), dtype=np.int64) % folds
    train_local = np.empty((len(train_z), 3 * len(ks)), dtype=np.float32)

    for fold in range(folds):
        query_local_idx = np.flatnonzero(fold_id == fold)
        pool_local_idx = np.flatnonzero(fold_id != fold)
        pool_z = train_z[pool_local_idx]
        pool_y = train_y[pool_local_idx]
        pool_prior = float(pool_y.mean())

        print(
            f"[Stage2 dataset] cross-fit fold {fold + 1}/{folds}: "
            f"pool={len(pool_z):,}, query={len(query_local_idx):,}, prior={pool_prior:.6f}"
        )
        index, _ = _make_faiss_index(pool_z, cfg, config["seed"] + fold)
        train_local[query_local_idx] = _query_local_features(
            index=index,
            query_z=train_z[query_local_idx],
            pool_y=pool_y,
            ks=ks,
            prior_strength=prior_strength,
            prior_prob=pool_prior,
            batch_size=batch_size,
            desc=f"crossfit {fold + 1}/{folds}",
        )
        del index

    # Validation queries use only 2019-2022 labels.
    train_prior = float(train_y.mean())
    full_index, backend = _make_faiss_index(train_z, cfg, config["seed"])
    val_local = _query_local_features(
        index=full_index,
        query_z=val_z,
        pool_y=train_y,
        ks=ks,
        prior_strength=prior_strength,
        prior_prob=train_prior,
        batch_size=batch_size,
        desc="2023 Stage2 local features",
    )

    train_x = np.concatenate([train_z, train_local], axis=1).astype(np.float32, copy=False)
    val_x = np.concatenate([val_z, val_local], axis=1).astype(np.float32, copy=False)

    root = _stage2_dir(config)
    _save_split(root, "train", train_x, train_y, train_global)
    _save_split(root, "val", val_x, val_y, val_global)

    latent_names = [f"z_{i:02d}" for i in range(train_z.shape[1])]
    local_names = _local_feature_names(ks)
    feature_names = latent_names + local_names

    metadata = {
        "train_seasons": config["data"]["train_seasons"],
        "val_seasons": config["data"]["val_seasons"],
        "test_seasons": config["data"]["test_seasons"],
        "train_rows": int(len(train_x)),
        "val_rows": int(len(val_x)),
        "latent_dim": int(train_z.shape[1]),
        "feature_dim": int(train_x.shape[1]),
        "feature_names": feature_names,
        "k_values": ks,
        "crossfit_folds": folds,
        "prior_strength": prior_strength,
        "train_target_mean": train_prior,
        "latent_scaler_mean": scaler.mean_.astype(float).tolist(),
        "latent_scaler_scale": scaler.scale_.astype(float).tolist(),
        "faiss_backend": backend,
        "local_probability_definition": "adaptive Gaussian distance-weighted Bernoulli mean with empirical-Bayes shrinkage",
        "train_local_features_are_cross_fitted": True,
        "validation_neighbor_pool": "2019-2022 only",
        "test_built": bool(include_test),
    }

    if include_test:
        test_local = _query_local_features(
            index=full_index,
            query_z=test_z,
            pool_y=train_y,
            ks=ks,
            prior_strength=prior_strength,
            prior_prob=train_prior,
            batch_size=batch_size,
            desc="2024 Stage2 local features",
        )
        test_x = np.concatenate([test_z, test_local], axis=1).astype(np.float32, copy=False)
        _save_split(root, "test", test_x, test_y, test_global)
        metadata["test_rows"] = int(len(test_x))

    save_json(metadata, root / "metadata.json")
    print(
        f"[Stage2 dataset] saved: train={train_x.shape}, val={val_x.shape}, "
        f"features={train_x.shape[1]}"
    )
    if not include_test:
        print("[Stage2 dataset] 2024 holdout was not materialized. Use --include-test only when ready.")
    return metadata


class ResidualProbabilityMLP(nn.Module):
    """
    Stage 2 predicts a correction to a local empirical probability prior.

    Initial final-layer weights are zero, so the model starts exactly from
    p_local and only learns evidence-supported corrections from z and
    neighborhood reliability features.
    """

    def __init__(
        self,
        input_dim: int,
        prior_index: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        self.prior_index = int(prior_index)
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.SiLU(), nn.Dropout(dropout)])
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p0 = x[:, self.prior_index].clamp(1e-4, 1.0 - 1e-4)
        base_logit = torch.logit(p0)
        delta = self.head(self.backbone(x)).squeeze(-1)
        return base_logit + delta


def _brier_np(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(p) - np.asarray(y))))


def _run_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            p = torch.sigmoid(logits)
            preds.append(p.cpu().numpy())
            ys.append(yb.numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(ys)
    return _brier_np(target, pred), pred


def train_stage2(config: dict) -> dict:
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

    import json

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

    model = ResidualProbabilityMLP(
        input_dim=train_x_np.shape[1],
        prior_index=prior_index,
        hidden_dims=[int(x) for x in cfg.get("hidden_dims", [128, 64])],
        dropout=float(cfg.get("dropout", 0.10)),
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    train_ds = TensorDataset(torch.from_numpy(train_x_np), torch.from_numpy(train_y_np))
    val_ds = TensorDataset(torch.from_numpy(val_x_np), torch.from_numpy(val_y_np))
    batch_size = int(cfg.get("batch_size", 4096))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    loss_name = str(cfg.get("loss", "brier")).lower()
    if loss_name not in {"brier", "bce"}:
        raise ValueError("stage2.loss는 brier 또는 bce여야 합니다.")

    epochs = int(cfg.get("epochs", 30))
    patience = int(cfg.get("patience", 5))
    best_brier = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    checkpoint_path = root / "stage2_best.pt"

    local_prior_val = val_x_np[:, prior_index]
    prior_brier = _brier_np(val_y_np, local_prior_val)
    mean_brier = _brier_np(
        val_y_np,
        np.full_like(val_y_np, float(metadata["train_target_mean"])),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"stage2 e{epoch}", leave=False)
        for xb, yb in bar:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if loss_name == "brier":
                loss = torch.mean(torch.square(torch.sigmoid(logits) - yb))
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

            bs = len(xb)
            total_loss += float(loss.item()) * bs
            seen += bs
            bar.set_postfix(loss=f"{total_loss / max(seen, 1):.6f}")

        train_loss = total_loss / max(seen, 1)
        val_brier, _ = _run_validation(model, val_loader, dev)
        row = {"epoch": epoch, "train_loss": train_loss, "val_brier": val_brier}
        history.append(row)
        print(f"epoch {epoch:02d}: train_loss={train_loss:.8f}, val_brier={val_brier:.8f}")

        if val_brier < best_brier - 1e-7:
            best_brier = val_brier
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": int(train_x_np.shape[1]),
                    "prior_index": int(prior_index),
                    "prior_name": prior_name,
                    "hidden_dims": [int(x) for x in cfg.get("hidden_dims", [128, 64])],
                    "dropout": float(cfg.get("dropout", 0.10)),
                    "feature_names": metadata["feature_names"],
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_brier, val_pred = _run_validation(model, val_loader, dev)

    result = {
        "best_epoch": int(best_epoch),
        "validation_brier": float(final_brier),
        "local_prior_brier": float(prior_brier),
        "train_mean_brier": float(mean_brier),
        "skill_vs_local_prior": float(1.0 - final_brier / prior_brier),
        "skill_vs_train_mean": float(1.0 - final_brier / mean_brier),
        "prior_feature": prior_name,
        "loss": loss_name,
        "device": str(dev),
        "history": history,
    }
    save_json(result, root / "stage2_metrics.json")

    pred_frame = pd.DataFrame(
        {
            "global_index": val_global,
            "target": val_y_np,
            "local_prior": local_prior_val,
            "stage2_probability": val_pred,
        }
    )
    pred_frame.to_csv(root / "stage2_val_predictions.csv", index=False)

    print("\n[Stage2 validation]")
    print(f"train mean baseline Brier : {mean_brier:.8f}")
    print(f"local prior Brier         : {prior_brier:.8f}")
    print(f"Stage2 Brier              : {final_brier:.8f}")
    print(f"skill vs local prior      : {result['skill_vs_local_prior']:+.6f}")
    print(f"skill vs train mean       : {result['skill_vs_train_mean']:+.6f}")
    return result
