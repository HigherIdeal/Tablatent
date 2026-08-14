from __future__ import annotations

import copy

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .data import (
    ContextPreprocessor,
    HistoryPreprocessor,
    load_frame,
    select_history_columns,
    split_masks,
)
from .models import (
    ContextVAE,
    HistoryVAE,
    context_reconstruction_loss,
    kl_divergence_standard_normal,
)
from .utils import device, ensure_output_dirs, save_json, seed_everything


def _loader(*arrays, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    tensors = []
    for array in arrays:
        a = np.asarray(array)
        if np.issubdtype(a.dtype, np.integer):
            tensors.append(torch.from_numpy(a.astype(np.int64, copy=False)))
        else:
            tensors.append(torch.from_numpy(a.astype(np.float32, copy=False)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _beta_for_epoch(cfg: dict, epoch: int) -> float:
    beta = float(cfg.get("kl_beta", 1e-4))
    warmup = int(cfg.get("kl_warmup_epochs", 10))
    if warmup <= 0:
        return beta
    return beta * min(1.0, float(epoch) / float(warmup))


def _train_context(
    model: ContextVAE,
    train_cat: np.ndarray,
    train_num: np.ndarray,
    val_cat: np.ndarray,
    val_num: np.ndarray,
    cfg: dict,
    common: dict,
):
    dev = device()
    model.to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=common["learning_rate"],
        weight_decay=common["weight_decay"],
    )
    train_loader = _loader(
        train_cat,
        train_num,
        batch_size=common["batch_size"],
        shuffle=True,
        workers=common["num_workers"],
    )
    val_loader = _loader(
        val_cat,
        val_num,
        batch_size=common["batch_size"],
        shuffle=False,
        workers=common["num_workers"],
    )

    final_beta = float(cfg.get("kl_beta", 1e-4))
    best = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, common["epochs"] + 1):
        beta = _beta_for_epoch(cfg, epoch)
        model.train()
        total_obj = 0.0
        total_recon = 0.0
        total_kl = 0.0
        seen = 0

        for cat, num in tqdm(train_loader, desc=f"context VAE e{epoch}", leave=False):
            cat = cat.to(dev, non_blocking=True)
            num = num.to(dev, non_blocking=True)

            noisy_cat = cat.clone()
            if cfg["category_mask_prob"] > 0:
                mask = torch.rand(noisy_cat.shape, device=dev) < cfg["category_mask_prob"]
                noisy_cat[mask] = 0

            noisy_num = num
            if cfg["numeric_noise_std"] > 0:
                noisy_num = num + torch.randn_like(num) * cfg["numeric_noise_std"]

            optimizer.zero_grad(set_to_none=True)
            cat_logits, pred_num, mu, logvar = model(noisy_cat, noisy_num)
            recon, _ = context_reconstruction_loss(
                cat_logits,
                cat,
                pred_num,
                num,
                cfg["categorical_loss_weight"],
                cfg["numeric_loss_weight"],
            )
            kl = kl_divergence_standard_normal(mu, logvar)
            loss = recon + beta * kl
            loss.backward()
            optimizer.step()

            bs = len(cat)
            seen += bs
            total_obj += float(loss.item()) * bs
            total_recon += float(recon.item()) * bs
            total_kl += float(kl.item()) * bs

        # Deterministic validation through posterior mean mu. The validation
        # objective always uses final beta so epochs remain comparable during warm-up.
        model.eval()
        val_obj = 0.0
        val_recon_total = 0.0
        val_kl_total = 0.0
        val_seen = 0
        with torch.inference_mode():
            for cat, num in val_loader:
                cat = cat.to(dev, non_blocking=True)
                num = num.to(dev, non_blocking=True)
                mu, logvar = model.encode_distribution(cat, num)
                cat_logits, pred_num = model.decode(mu)
                recon, _ = context_reconstruction_loss(
                    cat_logits,
                    cat,
                    pred_num,
                    num,
                    cfg["categorical_loss_weight"],
                    cfg["numeric_loss_weight"],
                )
                kl = kl_divergence_standard_normal(mu, logvar)
                objective = recon + final_beta * kl
                bs = len(cat)
                val_seen += bs
                val_obj += float(objective.item()) * bs
                val_recon_total += float(recon.item()) * bs
                val_kl_total += float(kl.item()) * bs

        row = {
            "epoch": epoch,
            "beta": beta,
            "train_loss": total_obj / max(seen, 1),
            "train_recon": total_recon / max(seen, 1),
            "train_kl": total_kl / max(seen, 1),
            "val_loss": val_obj / max(val_seen, 1),
            "val_recon": val_recon_total / max(val_seen, 1),
            "val_kl": val_kl_total / max(val_seen, 1),
        }
        history.append(row)
        print("[context VAE]", row)

        if row["val_loss"] < best - 1e-7:
            best = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= common["patience"]:
                break

    if best_state is None:
        raise RuntimeError("context VAE best_state 없음")
    model.load_state_dict(best_state)
    return history


def _train_history(
    model: HistoryVAE,
    train_x: np.ndarray,
    val_x: np.ndarray,
    cfg: dict,
    common: dict,
):
    dev = device()
    model.to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=common["learning_rate"],
        weight_decay=common["weight_decay"],
    )
    recon_fn = nn.SmoothL1Loss()
    train_loader = _loader(
        train_x,
        batch_size=common["batch_size"],
        shuffle=True,
        workers=common["num_workers"],
    )
    val_loader = _loader(
        val_x,
        batch_size=common["batch_size"],
        shuffle=False,
        workers=common["num_workers"],
    )

    final_beta = float(cfg.get("kl_beta", 1e-4))
    best = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, common["epochs"] + 1):
        beta = _beta_for_epoch(cfg, epoch)
        model.train()
        total_obj = 0.0
        total_recon = 0.0
        total_kl = 0.0
        seen = 0

        for (x,) in tqdm(train_loader, desc=f"history VAE e{epoch}", leave=False):
            x = x.to(dev, non_blocking=True)
            noisy = x + torch.randn_like(x) * cfg["noise_std"]

            optimizer.zero_grad(set_to_none=True)
            pred, mu, logvar = model(noisy)
            recon = recon_fn(pred, x)
            kl = kl_divergence_standard_normal(mu, logvar)
            loss = recon + beta * kl
            loss.backward()
            optimizer.step()

            bs = len(x)
            seen += bs
            total_obj += float(loss.item()) * bs
            total_recon += float(recon.item()) * bs
            total_kl += float(kl.item()) * bs

        model.eval()
        val_obj = 0.0
        val_recon_total = 0.0
        val_kl_total = 0.0
        val_seen = 0
        with torch.inference_mode():
            for (x,) in val_loader:
                x = x.to(dev, non_blocking=True)
                mu, logvar = model.encode_distribution(x)
                pred = model.decoder(mu)
                recon = recon_fn(pred, x)
                kl = kl_divergence_standard_normal(mu, logvar)
                objective = recon + final_beta * kl
                bs = len(x)
                val_seen += bs
                val_obj += float(objective.item()) * bs
                val_recon_total += float(recon.item()) * bs
                val_kl_total += float(kl.item()) * bs

        row = {
            "epoch": epoch,
            "beta": beta,
            "train_loss": total_obj / max(seen, 1),
            "train_recon": total_recon / max(seen, 1),
            "train_kl": total_kl / max(seen, 1),
            "val_loss": val_obj / max(val_seen, 1),
            "val_recon": val_recon_total / max(val_seen, 1),
            "val_kl": val_kl_total / max(val_seen, 1),
        }
        history.append(row)
        print("[history VAE]", row)

        if row["val_loss"] < best - 1e-7:
            best = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= common["patience"]:
                break

    if best_state is None:
        raise RuntimeError("history VAE best_state 없음")
    model.load_state_dict(best_state)
    return history


def _encode_context_distribution(
    model: ContextVAE,
    categorical: np.ndarray,
    numeric: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device
    mus = []
    logvars = []
    loader = _loader(categorical, numeric, batch_size=batch_size, shuffle=False, workers=0)
    with torch.inference_mode():
        for cat, num in tqdm(loader, desc="export context posterior", leave=False):
            mu, logvar = model.encode_distribution(
                cat.to(dev, non_blocking=True),
                num.to(dev, non_blocking=True),
            )
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    return np.concatenate(mus).astype(np.float32), np.concatenate(logvars).astype(np.float32)


def _encode_history_distribution(
    model: HistoryVAE,
    x: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device
    mus = []
    logvars = []
    loader = _loader(x, batch_size=batch_size, shuffle=False, workers=0)
    with torch.inference_mode():
        for (batch,) in tqdm(loader, desc="export history posterior", leave=False):
            mu, logvar = model.encode_distribution(batch.to(dev, non_blocking=True))
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    return np.concatenate(mus).astype(np.float32), np.concatenate(logvars).astype(np.float32)


def train_stage1(config: dict) -> None:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    out = ensure_output_dirs(config["paths"]["output_dir"])

    context_prep = ContextPreprocessor().fit(frame.loc[split["train"]])
    context_all = context_prep.transform(frame)
    context_cfg = config["stage1"]["context"]
    context_model = ContextVAE(
        cardinalities=context_prep.cardinalities,
        numeric_dim=context_all.numeric.shape[1],
        hidden_dims=context_cfg["hidden_dims"],
        latent_dim=context_cfg["latent_dim"],
        embedding_dim_max=context_cfg["embedding_dim_max"],
        dropout=context_cfg["dropout"],
    )
    context_training = _train_context(
        context_model,
        context_all.categorical[split["train"]],
        context_all.numeric[split["train"]],
        context_all.categorical[split["val"]],
        context_all.numeric[split["val"]],
        context_cfg,
        config["stage1"],
    )

    history_columns = select_history_columns(frame.loc[split["train"]], config)
    history_prep = HistoryPreprocessor(history_columns).fit(frame.loc[split["train"]])
    history_all = history_prep.transform(frame)
    history_cfg = config["stage1"]["history"]
    history_model = HistoryVAE(
        input_dim=history_all.shape[1],
        hidden_dims=history_cfg["hidden_dims"],
        latent_dim=history_cfg["latent_dim"],
        dropout=history_cfg["dropout"],
    )
    history_training = _train_history(
        history_model,
        history_all[split["train"]],
        history_all[split["val"]],
        history_cfg,
        config["stage1"],
    )

    context_mu, context_logvar = _encode_context_distribution(
        context_model,
        context_all.categorical,
        context_all.numeric,
        config["stage1"]["batch_size"],
    )
    history_mu, history_logvar = _encode_history_distribution(
        history_model,
        history_all,
        config["stage1"]["batch_size"],
    )

    storage_dtype = (
        np.float16
        if config["stage1"].get("latent_storage_dtype", "float16") == "float16"
        else np.float32
    )
    # context.npy/history.npy remain the deterministic Stage2 interface.
    # For a VAE they store posterior means mu, not random samples.
    np.save(out["latents"] / "context.npy", context_mu.astype(storage_dtype))
    np.save(out["latents"] / "history.npy", history_mu.astype(storage_dtype))
    np.save(out["latents"] / "context_logvar.npy", context_logvar.astype(storage_dtype))
    np.save(out["latents"] / "history_logvar.npy", history_logvar.astype(storage_dtype))

    torch.save(
        {
            "model_type": "vae",
            "state_dict": context_model.cpu().state_dict(),
            "cardinalities": context_prep.cardinalities,
            "numeric_dim": context_all.numeric.shape[1],
            "hidden_dims": context_cfg["hidden_dims"],
            "latent_dim": context_cfg["latent_dim"],
            "embedding_dim_max": context_cfg["embedding_dim_max"],
            "dropout": context_cfg["dropout"],
            "kl_beta": context_cfg.get("kl_beta", 1e-4),
        },
        out["checkpoints"] / "stage1_context.pt",
    )
    torch.save(
        {
            "model_type": "vae",
            "state_dict": history_model.cpu().state_dict(),
            "input_dim": history_all.shape[1],
            "hidden_dims": history_cfg["hidden_dims"],
            "latent_dim": history_cfg["latent_dim"],
            "dropout": history_cfg["dropout"],
            "kl_beta": history_cfg.get("kl_beta", 1e-4),
        },
        out["checkpoints"] / "stage1_history.pt",
    )
    joblib.dump(
        {
            "context": context_prep,
            "history": history_prep,
            "history_columns": history_columns,
        },
        out["checkpoints"] / "preprocessors.joblib",
    )
    save_json(
        {
            "model_type": "VAE",
            "latent_export": "posterior_mean_mu",
            "context_training": context_training,
            "history_training": history_training,
            "context_categorical_columns": context_prep.categorical_columns,
            "context_numeric_columns": context_prep.numeric_columns,
            "context_cardinalities": context_prep.cardinalities,
            "history_columns": history_columns,
            "context_latent_dim": int(context_mu.shape[1]),
            "history_latent_dim": int(history_mu.shape[1]),
            "combined_stage2_dim": int(context_mu.shape[1] + history_mu.shape[1]),
            "context_kl_beta": float(context_cfg.get("kl_beta", 1e-4)),
            "history_kl_beta": float(history_cfg.get("kl_beta", 1e-4)),
            "rows": int(len(frame)),
        },
        out["logs"] / "stage1_training.json",
    )


def _load_stage1(config: dict, frame: pd.DataFrame):
    out = ensure_output_dirs(config["paths"]["output_dir"])
    prep_bundle = joblib.load(out["checkpoints"] / "preprocessors.joblib")
    context_prep: ContextPreprocessor = prep_bundle["context"]
    history_prep: HistoryPreprocessor = prep_bundle["history"]

    context_arrays = context_prep.transform(frame)
    history_x = history_prep.transform(frame)

    c = torch.load(out["checkpoints"] / "stage1_context.pt", map_location="cpu", weights_only=True)
    if c.get("model_type") != "vae":
        raise RuntimeError("stage1_context.pt가 VAE checkpoint가 아닙니다. Stage1을 다시 학습하세요.")
    context_model = ContextVAE(
        cardinalities=c["cardinalities"],
        numeric_dim=c["numeric_dim"],
        hidden_dims=c["hidden_dims"],
        latent_dim=c["latent_dim"],
        embedding_dim_max=c["embedding_dim_max"],
        dropout=c["dropout"],
    )
    context_model.load_state_dict(c["state_dict"])
    context_model.to(device()).eval()

    h = torch.load(out["checkpoints"] / "stage1_history.pt", map_location="cpu", weights_only=True)
    if h.get("model_type") != "vae":
        raise RuntimeError("stage1_history.pt가 VAE checkpoint가 아닙니다. Stage1을 다시 학습하세요.")
    history_model = HistoryVAE(
        input_dim=h["input_dim"],
        hidden_dims=h["hidden_dims"],
        latent_dim=h["latent_dim"],
        dropout=h["dropout"],
    )
    history_model.load_state_dict(h["state_dict"])
    history_model.to(device()).eval()
    return out, context_prep, history_prep, context_arrays, history_x, context_model, history_model


def _posterior_metrics(mu: np.ndarray, logvar: np.ndarray) -> dict:
    std = np.exp(0.5 * logvar)
    kl_per_row = -0.5 * np.sum(1.0 + logvar - np.square(mu) - np.exp(logvar), axis=1)
    mu_dim_std = mu.std(axis=0)
    return {
        "mu_std_mean": float(mu_dim_std.mean()),
        "mu_std_min": float(mu_dim_std.min()),
        "mu_std_max": float(mu_dim_std.max()),
        "posterior_std_mean": float(std.mean()),
        "posterior_std_min": float(std.min()),
        "posterior_std_max": float(std.max()),
        "kl_mean_per_row": float(kl_per_row.mean()),
    }


def _context_metrics(
    model: ContextVAE,
    categorical: np.ndarray,
    numeric: np.ndarray,
    categorical_names: list[str],
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dev = next(model.parameters()).device
    loader = _loader(categorical, numeric, batch_size=batch_size, shuffle=False, workers=0)
    correct = np.zeros(len(categorical_names), dtype=np.int64)
    total = 0
    num_abs_sum = None
    num_sq_sum = None
    mus = []
    logvars = []
    pred_categories = []
    pred_numeric = []

    with torch.inference_mode():
        for cat, num in loader:
            cat = cat.to(dev, non_blocking=True)
            num = num.to(dev, non_blocking=True)
            mu, logvar = model.encode_distribution(cat, num)
            logits, pred_num = model.decode(mu)
            pred_cat = torch.stack([x.argmax(dim=1) for x in logits], dim=1)
            correct += (pred_cat == cat).sum(dim=0).cpu().numpy()
            total += len(cat)

            err = (pred_num - num).cpu().numpy()
            abs_sum = np.abs(err).sum(axis=0)
            sq_sum = np.square(err).sum(axis=0)
            num_abs_sum = abs_sum if num_abs_sum is None else num_abs_sum + abs_sum
            num_sq_sum = sq_sum if num_sq_sum is None else num_sq_sum + sq_sum

            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            pred_categories.append(pred_cat.cpu().numpy())
            pred_numeric.append(pred_num.cpu().numpy())

    accuracy = correct / max(total, 1)
    mu_np = np.concatenate(mus).astype(np.float32)
    logvar_np = np.concatenate(logvars).astype(np.float32)
    metrics = {
        "categorical_accuracy": {
            name: float(value) for name, value in zip(categorical_names, accuracy)
        },
        "categorical_macro_accuracy": float(accuracy.mean()) if len(accuracy) else None,
        "numeric_scaled_mae": (num_abs_sum / max(total, 1)).tolist(),
        "numeric_scaled_rmse": np.sqrt(num_sq_sum / max(total, 1)).tolist(),
        "posterior": _posterior_metrics(mu_np, logvar_np),
    }
    return (
        metrics,
        np.concatenate(pred_categories),
        np.concatenate(pred_numeric),
        mu_np,
        logvar_np,
    )


def _history_metrics(
    model: HistoryVAE,
    x: np.ndarray,
    feature_names: list[str],
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    dev = next(model.parameters()).device
    loader = _loader(x, batch_size=batch_size, shuffle=False, workers=0)
    preds = []
    mus = []
    logvars = []
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(dev, non_blocking=True)
            mu, logvar = model.encode_distribution(batch)
            pred = model.decoder(mu)
            preds.append(pred.cpu().numpy())
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

    pred = np.concatenate(preds).astype(np.float32)
    mu_np = np.concatenate(mus).astype(np.float32)
    logvar_np = np.concatenate(logvars).astype(np.float32)
    err = pred - x
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(np.square(err), axis=0))

    labels = list(feature_names)
    if pred.shape[1] > len(labels):
        labels += [f"missing_indicator_{i}" for i in range(pred.shape[1] - len(labels))]

    metrics = {
        "scaled_mae_mean": float(mae.mean()),
        "scaled_rmse_mean": float(rmse.mean()),
        "scaled_mae_by_feature": {name: float(v) for name, v in zip(labels, mae)},
        "scaled_rmse_by_feature": {name: float(v) for name, v in zip(labels, rmse)},
        "posterior": _posterior_metrics(mu_np, logvar_np),
    }
    return metrics, pred, mu_np, logvar_np


def evaluate_stage1(config: dict) -> dict:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    (
        out,
        context_prep,
        history_prep,
        context_arrays,
        history_x,
        context_model,
        history_model,
    ) = _load_stage1(config, frame)

    batch_size = config.get("evaluation", {}).get("batch_size", 8192)
    result = {"model_type": "VAE", "latent_for_stage2": "posterior_mean_mu"}
    sample_rows = config.get("evaluation", {}).get("sample_rows", 2000)
    rng = np.random.default_rng(config["seed"])

    for split_name in ("train", "val", "test"):
        mask = split[split_name]
        cat = context_arrays.categorical[mask]
        num = context_arrays.numeric[mask]
        hist = history_x[mask]

        c_metrics, c_pred_cat, c_pred_num, _, _ = _context_metrics(
            context_model,
            cat,
            num,
            context_prep.categorical_columns,
            batch_size,
        )
        h_metrics, h_pred, _, _ = _history_metrics(
            history_model,
            hist,
            history_prep.columns,
            batch_size,
        )
        result[split_name] = {
            "rows": int(mask.sum()),
            "context": c_metrics,
            "history": h_metrics,
        }

        local_n = len(cat)
        take = min(sample_rows, local_n)
        chosen = rng.choice(local_n, size=take, replace=False)
        sample = pd.DataFrame({"local_index": chosen})
        for i, name in enumerate(context_prep.categorical_columns):
            sample[f"ctx_true_{name}"] = cat[chosen, i]
            sample[f"ctx_pred_{name}"] = c_pred_cat[chosen, i]
        for i in range(num.shape[1]):
            sample[f"ctx_num_true_{i}"] = num[chosen, i]
            sample[f"ctx_num_pred_{i}"] = c_pred_num[chosen, i]
        for i, name in enumerate(history_prep.columns):
            sample[f"hist_true_{name}"] = hist[chosen, i]
            sample[f"hist_pred_{name}"] = h_pred[chosen, i]
        sample.to_csv(
            out["logs"] / f"reconstruction_sample_{split_name}.csv",
            index=False,
            encoding="utf-8",
        )

    save_json(result, out["logs"] / "stage1_reconstruction_metrics.json")
    print(result)
    return result
