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
from .models import ContextAutoencoder, HistoryAutoencoder, context_reconstruction_loss
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


def _train_context(
    model: ContextAutoencoder,
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

    best = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, common["epochs"] + 1):
        model.train()
        train_total = 0.0
        for cat, num in tqdm(train_loader, desc=f"context e{epoch}", leave=False):
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
            cat_logits, pred_num = model(noisy_cat, noisy_num)
            loss, _ = context_reconstruction_loss(
                cat_logits,
                cat,
                pred_num,
                num,
                cfg["categorical_loss_weight"],
                cfg["numeric_loss_weight"],
            )
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(cat)

        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for cat, num in val_loader:
                cat = cat.to(dev, non_blocking=True)
                num = num.to(dev, non_blocking=True)
                cat_logits, pred_num = model(cat, num)
                loss, _ = context_reconstruction_loss(
                    cat_logits,
                    cat,
                    pred_num,
                    num,
                    cfg["categorical_loss_weight"],
                    cfg["numeric_loss_weight"],
                )
                val_total += loss.item() * len(cat)

        row = {
            "epoch": epoch,
            "train_loss": train_total / len(train_cat),
            "val_loss": val_total / len(val_cat),
        }
        history.append(row)
        print("[context]", row)

        if row["val_loss"] < best - 1e-7:
            best = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= common["patience"]:
                break

    if best_state is None:
        raise RuntimeError("context best_state 없음")
    model.load_state_dict(best_state)
    return history


def _train_history(
    model: HistoryAutoencoder,
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
    loss_fn = nn.SmoothL1Loss()
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

    best = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, common["epochs"] + 1):
        model.train()
        train_total = 0.0
        for (x,) in tqdm(train_loader, desc=f"history e{epoch}", leave=False):
            x = x.to(dev, non_blocking=True)
            noisy = x + torch.randn_like(x) * cfg["noise_std"]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(noisy), x)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(x)

        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for (x,) in val_loader:
                x = x.to(dev, non_blocking=True)
                val_total += loss_fn(model(x), x).item() * len(x)

        row = {
            "epoch": epoch,
            "train_loss": train_total / len(train_x),
            "val_loss": val_total / len(val_x),
        }
        history.append(row)
        print("[history]", row)

        if row["val_loss"] < best - 1e-7:
            best = row["val_loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= common["patience"]:
                break

    if best_state is None:
        raise RuntimeError("history best_state 없음")
    model.load_state_dict(best_state)
    return history


def _encode_context(
    model: ContextAutoencoder,
    categorical: np.ndarray,
    numeric: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    result = []
    loader = _loader(categorical, numeric, batch_size=batch_size, shuffle=False, workers=0)
    with torch.inference_mode():
        for cat, num in tqdm(loader, desc="export context latent", leave=False):
            z = model.encode(cat.to(dev, non_blocking=True), num.to(dev, non_blocking=True))
            result.append(z.cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def _encode_history(model: HistoryAutoencoder, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    result = []
    loader = _loader(x, batch_size=batch_size, shuffle=False, workers=0)
    with torch.inference_mode():
        for (batch,) in tqdm(loader, desc="export history latent", leave=False):
            result.append(model.encoder(batch.to(dev, non_blocking=True)).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def train_stage1(config: dict) -> None:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    out = ensure_output_dirs(config["paths"]["output_dir"])

    context_prep = ContextPreprocessor().fit(frame.loc[split["train"]])
    context_all = context_prep.transform(frame)
    context_cfg = config["stage1"]["context"]
    context_model = ContextAutoencoder(
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
    history_model = HistoryAutoencoder(
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

    context_latent = _encode_context(
        context_model,
        context_all.categorical,
        context_all.numeric,
        config["stage1"]["batch_size"],
    )
    history_latent = _encode_history(
        history_model,
        history_all,
        config["stage1"]["batch_size"],
    )

    storage_dtype = (
        np.float16 if config["stage1"].get("latent_storage_dtype", "float16") == "float16" else np.float32
    )
    np.save(out["latents"] / "context.npy", context_latent.astype(storage_dtype))
    np.save(out["latents"] / "history.npy", history_latent.astype(storage_dtype))

    torch.save(
        {
            "state_dict": context_model.cpu().state_dict(),
            "cardinalities": context_prep.cardinalities,
            "numeric_dim": context_all.numeric.shape[1],
            "hidden_dims": context_cfg["hidden_dims"],
            "latent_dim": context_cfg["latent_dim"],
            "embedding_dim_max": context_cfg["embedding_dim_max"],
            "dropout": context_cfg["dropout"],
        },
        out["checkpoints"] / "stage1_context.pt",
    )
    torch.save(
        {
            "state_dict": history_model.cpu().state_dict(),
            "input_dim": history_all.shape[1],
            "hidden_dims": history_cfg["hidden_dims"],
            "latent_dim": history_cfg["latent_dim"],
            "dropout": history_cfg["dropout"],
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
            "context_training": context_training,
            "history_training": history_training,
            "context_categorical_columns": context_prep.categorical_columns,
            "context_numeric_columns": context_prep.numeric_columns,
            "context_cardinalities": context_prep.cardinalities,
            "history_columns": history_columns,
            "context_latent_dim": int(context_latent.shape[1]),
            "history_latent_dim": int(history_latent.shape[1]),
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
    context_model = ContextAutoencoder(
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
    history_model = HistoryAutoencoder(
        input_dim=h["input_dim"],
        hidden_dims=h["hidden_dims"],
        latent_dim=h["latent_dim"],
        dropout=h["dropout"],
    )
    history_model.load_state_dict(h["state_dict"])
    history_model.to(device()).eval()
    return out, context_prep, history_prep, context_arrays, history_x, context_model, history_model


def _context_metrics(
    model: ContextAutoencoder,
    categorical: np.ndarray,
    numeric: np.ndarray,
    categorical_names: list[str],
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    dev = next(model.parameters()).device
    loader = _loader(categorical, numeric, batch_size=batch_size, shuffle=False, workers=0)
    correct = np.zeros(len(categorical_names), dtype=np.int64)
    total = 0
    num_abs_sum = None
    num_sq_sum = None
    latents = []
    pred_categories = []
    pred_numeric = []

    with torch.inference_mode():
        for cat, num in loader:
            cat = cat.to(dev, non_blocking=True)
            num = num.to(dev, non_blocking=True)
            z = model.encode(cat, num)
            logits, pred_num = model.decode(z)
            pred_cat = torch.stack([x.argmax(dim=1) for x in logits], dim=1)
            correct += (pred_cat == cat).sum(dim=0).cpu().numpy()
            total += len(cat)

            err = (pred_num - num).cpu().numpy()
            abs_sum = np.abs(err).sum(axis=0)
            sq_sum = np.square(err).sum(axis=0)
            num_abs_sum = abs_sum if num_abs_sum is None else num_abs_sum + abs_sum
            num_sq_sum = sq_sum if num_sq_sum is None else num_sq_sum + sq_sum

            latents.append(z.cpu().numpy())
            pred_categories.append(pred_cat.cpu().numpy())
            pred_numeric.append(pred_num.cpu().numpy())

    accuracy = correct / max(total, 1)
    metrics = {
        "categorical_accuracy": {
            name: float(value) for name, value in zip(categorical_names, accuracy)
        },
        "categorical_macro_accuracy": float(accuracy.mean()) if len(accuracy) else None,
        "numeric_scaled_mae": (num_abs_sum / max(total, 1)).tolist(),
        "numeric_scaled_rmse": np.sqrt(num_sq_sum / max(total, 1)).tolist(),
    }
    z = np.concatenate(latents).astype(np.float32)
    metrics["latent_std_mean"] = float(z.std(axis=0).mean())
    metrics["latent_std_min"] = float(z.std(axis=0).min())
    metrics["latent_std_max"] = float(z.std(axis=0).max())
    return metrics, np.concatenate(pred_categories), np.concatenate(pred_numeric), z


def _history_metrics(
    model: HistoryAutoencoder,
    x: np.ndarray,
    feature_names: list[str],
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    dev = next(model.parameters()).device
    loader = _loader(x, batch_size=batch_size, shuffle=False, workers=0)
    preds = []
    latents = []
    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(dev, non_blocking=True)
            z = model.encoder(batch)
            pred = model.decoder(z)
            preds.append(pred.cpu().numpy())
            latents.append(z.cpu().numpy())

    pred = np.concatenate(preds).astype(np.float32)
    z = np.concatenate(latents).astype(np.float32)
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
        "latent_std_mean": float(z.std(axis=0).mean()),
        "latent_std_min": float(z.std(axis=0).min()),
        "latent_std_max": float(z.std(axis=0).max()),
    }
    return metrics, pred, z


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
    result = {}
    sample_rows = config.get("evaluation", {}).get("sample_rows", 2000)
    rng = np.random.default_rng(config["seed"])

    for split_name in ("train", "val", "test"):
        mask = split[split_name]
        cat = context_arrays.categorical[mask]
        num = context_arrays.numeric[mask]
        hist = history_x[mask]

        c_metrics, c_pred_cat, c_pred_num, _ = _context_metrics(
            context_model,
            cat,
            num,
            context_prep.categorical_columns,
            batch_size,
        )
        h_metrics, h_pred, _ = _history_metrics(
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
