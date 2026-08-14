from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import TabularPreprocessor, split_feature_groups
from .models import DenoisingAutoencoder, ProbabilityHead
from .utils import device, ensure_output_dirs, save_json, seed_everything


def load_frame(config: dict) -> pd.DataFrame:
    path = Path(config["paths"]["processed_dir"]) / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. python scripts/prepare_data.py 를 먼저 실행하세요.")
    return pd.read_csv(path, low_memory=False)


def masks(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    season = pd.to_numeric(frame[config["data"]["season_col"]], errors="raise").astype(int)
    result = {name: season.isin(config["data"][f"{name}_seasons"]).to_numpy() for name in ("train", "val", "test")}
    if any(not value.any() for value in result.values()):
        raise ValueError({name: int(value.sum()) for name, value in result.items()})
    if np.any(result["train"] & result["val"]) or np.any(result["train"] & result["test"]) or np.any(result["val"] & result["test"]):
        raise ValueError("시즌 split이 서로 겹칩니다.")
    return result


def _loader(*arrays: np.ndarray, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(a, dtype="float32")) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


def _train_autoencoder(model, train_x, val_x, cfg, seed):
    dev = device()
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.SmoothL1Loss()
    train_loader = _loader(train_x, batch_size=cfg["batch_size"], shuffle=True, workers=cfg["num_workers"])
    val_loader = _loader(val_x, batch_size=cfg["batch_size"], shuffle=False, workers=cfg["num_workers"])
    generator = torch.Generator(device=dev).manual_seed(seed)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_total = 0.0
        for (x,) in train_loader:
            x = x.to(dev, non_blocking=True)
            noisy = x + torch.randn(x.shape, generator=generator, device=dev) * cfg["noise_std"]
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
        row = {"epoch": epoch, "train_loss": train_total / len(train_x), "val_loss": val_total / len(val_x)}
        history.append(row)
        if row["val_loss"] < best - 1e-7:
            best, best_state, stale = row["val_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


def _encode(model: DenoisingAutoencoder, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    result = []
    with torch.inference_mode():
        for (batch,) in _loader(x, batch_size=batch_size, shuffle=False, workers=0):
            result.append(model.encoder(batch.to(dev)).cpu().numpy())
    return np.concatenate(result).astype("float32")


def train_stage1(config: dict) -> None:
    seed_everything(config["seed"])
    frame, split = load_frame(config), None
    split = masks(frame, config)
    groups = split_feature_groups(frame.loc[split["train"]], config)
    preprocessors = {"current": TabularPreprocessor(groups.current), "history": TabularPreprocessor(groups.history)}
    arrays, models, histories = {}, {}, {}
    cfg = config["stage1"]
    for name, columns in (("current", groups.current), ("history", groups.history)):
        prep = preprocessors[name]
        train_x = prep.fit_transform(frame.loc[split["train"], columns])
        all_x = prep.transform(frame[columns])
        val_x = all_x[split["val"]]
        latent_dim = cfg[f"latent_dim_{name}"]
        model = DenoisingAutoencoder(train_x.shape[1], cfg["hidden_dims"], latent_dim, cfg["dropout"])
        histories[name] = _train_autoencoder(model, train_x, val_x, cfg, config["seed"] + len(models))
        arrays[name] = _encode(model, all_x, cfg["batch_size"])
        models[name] = model
    out = ensure_output_dirs(config["paths"]["output_dir"])
    for name, model in models.items():
        torch.save({"state_dict": model.cpu().state_dict(), "input_dim": preprocessors[name].transform(frame.iloc[:1][getattr(groups, name)]).shape[1], "hidden_dims": cfg["hidden_dims"], "latent_dim": cfg[f"latent_dim_{name}"], "dropout": cfg["dropout"]}, out["checkpoints"] / f"stage1_{name}.pt")
        np.save(out["latents"] / f"{name}.npy", arrays[name])
    np.save(out["latents"] / "row_id.npy", np.arange(len(frame), dtype="int64"))
    joblib.dump({"current": preprocessors["current"], "history": preprocessors["history"], "groups": groups}, out["checkpoints"] / "preprocessors.joblib")
    save_json(histories, out["logs"] / "stage1_history.json")
    save_json({"current": groups.current, "history": groups.history, "excluded": groups.excluded}, out["logs"] / "feature_groups.json")


def train_stage2(config: dict) -> None:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = masks(frame, config)
    out = ensure_output_dirs(config["paths"]["output_dir"])
    latent = np.concatenate([np.load(out["latents"] / "current.npy"), np.load(out["latents"] / "history.npy")], axis=1).astype("float32")
    y = pd.to_numeric(frame[config["data"]["target_col"]], errors="raise").to_numpy(dtype="float32")
    cfg, dev = config["stage2"], device()
    model = ProbabilityHead(latent.shape[1], cfg["hidden_dims"], cfg["dropout"]).to(dev)
    # Unweighted log-loss preserves probability calibration required by Brier.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train_loader = _loader(latent[split["train"]], y[split["train"]], batch_size=cfg["batch_size"], shuffle=True, workers=cfg["num_workers"])
    val_loader = _loader(latent[split["val"]], y[split["val"]], batch_size=cfg["batch_size"], shuffle=False, workers=cfg["num_workers"])
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for x, target in train_loader:
            x, target = x.to(dev), target.to(dev)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        model.eval()
        probs, truth = [], []
        with torch.inference_mode():
            for x, target in val_loader:
                probs.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
                truth.append(target.numpy())
        score = brier_score_loss(np.concatenate(truth), np.concatenate(probs))
        history.append({"epoch": epoch, "train_loss": total / split["train"].sum(), "val_brier": score})
        if score < best - 1e-7:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(state)
    torch.save({"state_dict": model.cpu().state_dict(), "input_dim": latent.shape[1], "hidden_dims": cfg["hidden_dims"], "dropout": cfg["dropout"], "train_climatology": float(y[split["train"]].mean())}, out["checkpoints"] / "stage2_predictor.pt")
    save_json(history, out["logs"] / "stage2_history.json")


def evaluate(config: dict) -> dict:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = masks(frame, config)
    out = ensure_output_dirs(config["paths"]["output_dir"])
    latent = np.concatenate([np.load(out["latents"] / "current.npy"), np.load(out["latents"] / "history.npy")], axis=1).astype("float32")
    checkpoint = torch.load(out["checkpoints"] / "stage2_predictor.pt", map_location="cpu", weights_only=True)
    model = ProbabilityHead(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device()).eval()
    probabilities = []
    with torch.inference_mode():
        for (x,) in _loader(latent[split["test"]], batch_size=config["stage2"]["batch_size"], shuffle=False, workers=0):
            probabilities.append(torch.sigmoid(model(x.to(device()))).cpu().numpy())
    probability = np.concatenate(probabilities)
    y = pd.to_numeric(frame.loc[split["test"], config["data"]["target_col"]], errors="raise").to_numpy()
    model_brier = float(brier_score_loss(y, probability))
    baseline = np.full(len(y), checkpoint["train_climatology"])
    baseline_brier = float(brier_score_loss(y, baseline))
    metrics = {"split": "test", "seasons": config["data"]["test_seasons"], "rows": len(y), "brier": model_brier, "baseline_brier": baseline_brier, "brier_skill_score": 1.0 - model_brier / baseline_brier, "train_climatology": checkpoint["train_climatology"]}
    save_json(metrics, out["logs"] / "metrics.json")
    pd.DataFrame({"row_id": np.flatnonzero(split["test"]), "control_success_probability": probability, "target": y}).to_csv(out["logs"] / "test_predictions.csv", index=False, encoding="utf-8")
    return metrics
