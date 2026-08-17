#!/usr/bin/env python3
"""Train and temporally evaluate the Physics-Arsenal Mixture-of-Experts.

Current-pitch Trackman columns are loaded only as auxiliary labels.  Model
inputs consist of one independent main-table row plus frozen, prior-season
arsenal tokens, preserving the 2025 row-independence requirement.

This is a heavy GPU script and is intended to be run explicitly by the user.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pitch_arsenal_moe import (  # noqa: E402
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    PHYSICAL_TARGETS,
    RAW_CONTEXT_REQUIRED,
    ModelConfig,
    PhysicsArsenalMoE,
    add_context_features,
    apply_arsenal_scaler,
    apply_scaler,
    category_cardinalities,
    compute_prior_logits,
    encode_categories,
    fit_arsenal_scaler,
    fit_category_maps,
    fit_regime_priors,
    fit_scaler,
    load_arsenal_tables,
    log_transform_arsenal,
    parameter_count,
    raw_context_numeric,
    recency_weights,
    regime_indices,
    resolve_arsenal,
)


DEFAULT_TRAIN = ROOT / "data" / "raw" / "train.csv"
DEFAULT_ARSENAL_DIR = ROOT / "data" / "processed" / "pitch_arsenal_moe"
DEFAULT_OUTPUT = ROOT / "outputs" / "pitch_arsenal_moe_v2"
TARGET = "control_success"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    weight: float
    kind: str
    cutoff_month: int | None = None
    valid_start: int | None = None
    valid_end: int | None = None


FOLDS = (
    FoldSpec("season_forward_2024", 0.50, "season_forward"),
    FoldSpec("mid_2024", 0.20, "within", 5, 6, 7),
    FoldSpec("late_2024", 0.30, "within", 7, 8, None),
)


@dataclass
class RawArrays:
    context_numeric: np.ndarray
    arsenal: np.ndarray
    arsenal_source: np.ndarray
    regime: np.ndarray
    target: np.ndarray
    seasons: np.ndarray
    aux_group: np.ndarray
    aux_physical: np.ndarray
    aux_physical_mask: np.ndarray


@dataclass
class FoldArrays:
    categorical: np.ndarray
    context_numeric: np.ndarray
    arsenal: np.ndarray
    arsenal_source: np.ndarray
    regime: np.ndarray
    prior_logit: np.ndarray
    target: np.ndarray
    aux_group: np.ndarray
    aux_physical: np.ndarray
    aux_physical_mask: np.ndarray
    sample_weight: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cv", "final"], default="cv")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--arsenal-dir", type=Path, default=DEFAULT_ARSENAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--brier-weight", type=float, default=0.35)
    parser.add_argument("--pitch-loss-weight", type=float, default=0.05)
    parser.add_argument("--physics-loss-weight", type=float, default=0.02)
    parser.add_argument("--arsenal-residual-scale", type=float, default=0.25)
    parser.add_argument("--prior-strength", type=float, default=200.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument(
        "--use-cv-epoch",
        action="store_true",
        help="In final mode, use rounded weighted best epoch from cv_summary.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if not 0 <= args.brier_weight <= 1:
        raise ValueError("--brier-weight must be in [0, 1]")
    if args.pitch_loss_weight < 0 or args.physics_loss_weight < 0:
        raise ValueError("auxiliary loss weights must be non-negative")
    if not 0 <= args.arsenal_residual_scale <= 1:
        raise ValueError("--arsenal-residual-scale must be in [0, 1]")
    if args.prior_strength <= 0:
        raise ValueError("--prior-strength must be positive")
    if args.d_model % args.attention_heads:
        raise ValueError("--d-model must be divisible by --attention-heads")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def load_main_frame(path: Path) -> pd.DataFrame:
    required = sorted({*RAW_CONTEXT_REQUIRED, TARGET})
    frame = pd.read_csv(path, usecols=required, low_memory=False, dtype={"row_id": "string"})
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")
    frame = add_context_features(frame)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["game_month"] = pd.to_numeric(
        frame["game_month"], errors="raise"
    ).astype(int)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(np.float32)
    if frame["row_id"].duplicated().any():
        raise ValueError("train row_id is not unique")
    return frame


def load_auxiliary(
    frame: pd.DataFrame, path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [
        "row_id",
        "aux_pitch_group_id",
        *[f"aux_{column}" for column in PHYSICAL_TARGETS],
    ]
    auxiliary = pd.read_parquet(path, columns=columns)
    auxiliary["row_id"] = auxiliary["row_id"].astype("string")
    if auxiliary["row_id"].duplicated().any():
        raise ValueError("Auxiliary table contains duplicate row_id")
    auxiliary = auxiliary.set_index("row_id").reindex(frame["row_id"])
    group = pd.to_numeric(
        auxiliary["aux_pitch_group_id"], errors="coerce"
    ).fillna(-1).astype(np.int64).to_numpy()
    physical = np.column_stack(
        [
            pd.to_numeric(auxiliary[f"aux_{column}"], errors="coerce").to_numpy(
                np.float32
            )
            for column in PHYSICAL_TARGETS
        ]
    ).astype(np.float32, copy=False)
    mask = np.isfinite(physical)
    return group, physical, mask


def prepare_raw_arrays(
    frame: pd.DataFrame, arsenal_directory: Path
) -> tuple[RawArrays, list[str]]:
    print("resolving player -> team+hand -> league+hand -> league arsenal fallback")
    tables = load_arsenal_tables(arsenal_directory)
    arsenal, source = resolve_arsenal(frame, tables)
    arsenal = log_transform_arsenal(arsenal, tables.feature_columns)
    aux_group, aux_physical, aux_mask = load_auxiliary(
        frame, arsenal_directory / "aligned_pitch_auxiliary.parquet"
    )
    raw = RawArrays(
        context_numeric=raw_context_numeric(frame),
        arsenal=arsenal,
        arsenal_source=source,
        regime=regime_indices(frame),
        target=frame[TARGET].to_numpy(np.float32),
        seasons=frame["season"].to_numpy(np.int64),
        aux_group=aux_group,
        aux_physical=aux_physical,
        aux_physical_mask=aux_mask,
    )
    return raw, tables.feature_columns


def fold_indices(frame: pd.DataFrame, spec: FoldSpec) -> tuple[np.ndarray, np.ndarray]:
    season = frame["season"].to_numpy(np.int64)
    month = frame["game_month"].to_numpy(np.int64)
    if spec.kind == "season_forward":
        train = season <= 2023
        valid = season == 2024
    else:
        if spec.cutoff_month is None or spec.valid_start is None:
            raise ValueError(f"Invalid fold: {spec}")
        train = (season < 2024) | ((season == 2024) & (month <= spec.cutoff_month))
        valid = (season == 2024) & (month >= spec.valid_start)
        if spec.valid_end is not None:
            valid &= month <= spec.valid_end
    if np.any(train & valid):
        raise RuntimeError(f"Temporal leakage in {spec.name}")
    return np.flatnonzero(train), np.flatnonzero(valid)


def prepare_fold_arrays(
    frame: pd.DataFrame,
    raw: RawArrays,
    train_indices: np.ndarray,
    prior_strength: float,
) -> tuple[FoldArrays, dict[str, Any]]:
    category_maps = fit_category_maps(frame, train_indices)
    categorical = encode_categories(frame, category_maps)
    context_scaler = fit_scaler(raw.context_numeric, train_indices)
    context_numeric = apply_scaler(raw.context_numeric, context_scaler)
    arsenal_scaler = fit_arsenal_scaler(raw.arsenal, train_indices)
    arsenal = apply_arsenal_scaler(raw.arsenal, arsenal_scaler)
    sample_weight = recency_weights(raw.seasons, train_indices)
    regime_priors = fit_regime_priors(
        frame, raw.target, train_indices, sample_weight
    )
    prior_logit = compute_prior_logits(frame, regime_priors, prior_strength)

    aux_train = train_indices[raw.aux_group[train_indices] >= 0]
    if not len(aux_train):
        raise RuntimeError("Training split contains no exact auxiliary rows")
    physical_scaler = fit_scaler(raw.aux_physical, aux_train)
    aux_physical = apply_scaler(raw.aux_physical, physical_scaler)
    arrays = FoldArrays(
        categorical=categorical,
        context_numeric=context_numeric,
        arsenal=arsenal,
        arsenal_source=raw.arsenal_source,
        regime=raw.regime,
        prior_logit=prior_logit,
        target=raw.target,
        aux_group=raw.aux_group,
        aux_physical=aux_physical,
        aux_physical_mask=raw.aux_physical_mask,
        sample_weight=sample_weight,
    )
    preprocessing = {
        "category_maps": category_maps,
        "context_scaler": context_scaler,
        "arsenal_scaler": arsenal_scaler,
        "physical_scaler": physical_scaler,
        "regime_priors": regime_priors,
        "prior_strength": prior_strength,
    }
    return arrays, preprocessing


def make_model(
    args: argparse.Namespace,
    preprocessing: dict[str, Any],
    arsenal_feature_dim: int,
) -> PhysicsArsenalMoE:
    config = ModelConfig(
        category_cardinalities=category_cardinalities(
            preprocessing["category_maps"]
        ),
        context_numeric_dim=len(CONTEXT_NUMERIC),
        arsenal_feature_dim=arsenal_feature_dim,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        dropout=args.dropout,
        arsenal_residual_scale=args.arsenal_residual_scale,
    )
    return PhysicsArsenalMoE(config)


def tensor_batch(
    arrays: FoldArrays, indices: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "categorical": torch.as_tensor(
            arrays.categorical[indices], dtype=torch.long, device=device
        ),
        "context_numeric": torch.as_tensor(
            arrays.context_numeric[indices], dtype=torch.float32, device=device
        ),
        "arsenal": torch.as_tensor(
            arrays.arsenal[indices], dtype=torch.float32, device=device
        ),
        "arsenal_source": torch.as_tensor(
            arrays.arsenal_source[indices], dtype=torch.long, device=device
        ),
        "regime": torch.as_tensor(
            arrays.regime[indices], dtype=torch.long, device=device
        ),
        "prior_logit": torch.as_tensor(
            arrays.prior_logit[indices], dtype=torch.float32, device=device
        ),
        "target": torch.as_tensor(
            arrays.target[indices], dtype=torch.float32, device=device
        ),
        "aux_group": torch.as_tensor(
            arrays.aux_group[indices], dtype=torch.long, device=device
        ),
        "aux_physical": torch.as_tensor(
            arrays.aux_physical[indices], dtype=torch.float32, device=device
        ),
        "aux_physical_mask": torch.as_tensor(
            arrays.aux_physical_mask[indices], dtype=torch.bool, device=device
        ),
        "sample_weight": torch.as_tensor(
            arrays.sample_weight[indices], dtype=torch.float32, device=device
        ),
    }


def forward_model(
    model: PhysicsArsenalMoE, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return model(
        batch["categorical"],
        batch["context_numeric"],
        batch["arsenal"],
        batch["arsenal_source"],
        batch["regime"],
        batch["prior_logit"],
    )


def loss_components(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(
        output["logit"], batch["target"], reduction="none"
    )
    brier = torch.square(output["probability"] - batch["target"])
    success_per_row = (1.0 - args.brier_weight) * bce + args.brier_weight * brier
    weights = batch["sample_weight"]
    success_loss = torch.sum(success_per_row * weights) / torch.sum(weights)

    auxiliary_rows = batch["aux_group"] >= 0
    if auxiliary_rows.any():
        pitch_loss = F.cross_entropy(
            output["selection_logits"][auxiliary_rows],
            batch["aux_group"][auxiliary_rows],
        )
        row_indices = torch.nonzero(auxiliary_rows, as_tuple=False).squeeze(1)
        group_indices = batch["aux_group"][auxiliary_rows]
        predicted_physics = output["physics_prediction"][
            row_indices, group_indices
        ]
        target_physics = batch["aux_physical"][auxiliary_rows]
        physics_mask = batch["aux_physical_mask"][auxiliary_rows]
        if physics_mask.any():
            physics_elements = F.smooth_l1_loss(
                predicted_physics,
                target_physics,
                reduction="none",
                beta=0.5,
            )
            physics_loss = physics_elements[physics_mask].mean()
        else:
            physics_loss = success_loss.new_zeros(())
    else:
        pitch_loss = success_loss.new_zeros(())
        physics_loss = success_loss.new_zeros(())

    total = (
        success_loss
        + args.pitch_loss_weight * pitch_loss
        + args.physics_loss_weight * physics_loss
    )
    metrics = {
        "total": float(total.detach().item()),
        "success": float(success_loss.detach().item()),
        "pitch": float(pitch_loss.detach().item()),
        "physics": float(physics_loss.detach().item()),
    }
    return total, metrics


def train_epoch(
    model: PhysicsArsenalMoE,
    arrays: FoldArrays,
    train_indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    order = train_indices.copy()
    rng.shuffle(order)
    totals = {name: 0.0 for name in ("total", "success", "pitch", "physics")}
    brier_sum = 0.0
    target_sum = 0.0
    metric_rows = 0
    batches = math.ceil(len(order) / args.batch_size)
    progress = tqdm(
        range(0, len(order), args.batch_size),
        total=batches,
        desc=f"epoch {epoch:02d} train",
        unit="batch",
        leave=False,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    for batch_number, start in enumerate(progress, start=1):
        indices = order[start : start + args.batch_size]
        batch = tensor_batch(arrays, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            output = forward_model(model, batch)
            loss, metrics = loss_components(output, batch, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for name, value in metrics.items():
            totals[name] += value
        with torch.no_grad():
            batch_rows = int(batch["target"].numel())
            brier_sum += float(
                torch.sum(
                    torch.square(output["probability"] - batch["target"])
                ).item()
            )
            target_sum += float(torch.sum(batch["target"]).item())
            metric_rows += batch_rows
        running_brier = brier_sum / metric_rows
        running_rate = target_sum / metric_rows
        running_reference = running_rate * (1.0 - running_rate)
        running_score = (
            max(0.0, 100_000.0 * (1.0 - running_brier / running_reference))
            if running_reference > 0
            else 0.0
        )
        progress.set_postfix(
            loss=f"{totals['total'] / batch_number:.2e}",
            train_brier=f"{running_brier:.6f}",
            train_score=f"{running_score:.1f}",
        )
    train_brier = brier_sum / metric_rows
    train_rate = target_sum / metric_rows
    train_reference = train_rate * (1.0 - train_rate)
    train_score = (
        max(0.0, 100_000.0 * (1.0 - train_brier / train_reference))
        if train_reference > 0
        else 0.0
    )
    return {
        **{name: value / batches for name, value in totals.items()},
        "brier": train_brier,
        "raw_score": train_score,
    }


def predict_indices(
    model: PhysicsArsenalMoE,
    arrays: FoldArrays,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_parts: list[np.ndarray] = []
    mixture_parts: list[np.ndarray] = []
    progress = tqdm(
        range(0, len(indices), batch_size),
        total=math.ceil(len(indices) / batch_size),
        desc=description,
        unit="batch",
        leave=False,
    )
    with torch.inference_mode():
        for start in progress:
            selected = indices[start : start + batch_size]
            batch = tensor_batch(arrays, selected, device)
            output = forward_model(model, batch)
            probability_parts.append(
                output["probability"].float().cpu().numpy()
            )
            mixture_parts.append(
                output["mixture_weights"].float().cpu().numpy()
            )
    return np.concatenate(probability_parts), np.concatenate(mixture_parts)


def brier_metrics(
    frame: pd.DataFrame,
    raw: RawArrays,
    valid_indices: np.ndarray,
    probability: np.ndarray,
    mixture: np.ndarray,
) -> dict[str, float | int]:
    target = raw.target[valid_indices]
    brier = float(np.mean(np.square(probability - target)))
    target_rate = float(np.mean(target))
    reference = target_rate * (1.0 - target_rate)
    raw_score = float(max(0.0, 100_000.0 * (1.0 - brier / reference)))
    result: dict[str, float | int] = {
        "rows": len(valid_indices),
        "target_rate": target_rate,
        "brier": brier,
        "raw_score": raw_score,
    }
    game_type = frame["game_type"].astype("string").to_numpy()[valid_indices]
    for regime in ("R", "F"):
        mask = game_type == regime
        result[f"{regime.lower()}_rows"] = int(mask.sum())
        result[f"{regime.lower()}_brier"] = (
            float(np.mean(np.square(probability[mask] - target[mask])))
            if mask.any()
            else math.nan
        )
    aux_group = raw.aux_group[valid_indices]
    aux_mask = aux_group >= 0
    result["aux_pitch_rows"] = int(aux_mask.sum())
    result["aux_pitch_accuracy"] = (
        float(np.mean(np.argmax(mixture[aux_mask], axis=1) == aux_group[aux_mask]))
        if aux_mask.any()
        else math.nan
    )
    return result


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def save_checkpoint_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: PhysicsArsenalMoE,
    preprocessing: dict[str, Any],
    arsenal_features: list[str],
    epoch: int,
    fold: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state": cpu_state_dict(model),
        "model_config": model.config.to_dict(),
        "preprocessing": preprocessing,
        "context_categorical": CONTEXT_CATEGORICAL,
        "context_numeric": CONTEXT_NUMERIC,
        "arsenal_features": arsenal_features,
        "physical_targets": PHYSICAL_TARGETS,
        "epoch": epoch,
        "fold": fold,
        "seed": args.seed,
        "training_args": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "inference_contract": (
            "one main-table row plus Trackman season < feature season arsenal; "
            "current-pitch auxiliary columns are forbidden inputs; v2 uses a "
            "direct success head plus bounded arsenal residual"
        ),
    }


def make_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    train_rows: int,
    epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(train_rows / args.batch_size)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * 0.05))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def train_cv(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    raw: RawArrays,
    arsenal_features: list[str],
    device: torch.device,
) -> None:
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    outer = tqdm(FOLDS, desc="temporal folds", unit="fold")
    for fold_number, spec in enumerate(outer, start=1):
        outer.set_postfix(fold=spec.name)
        train_indices, valid_indices = fold_indices(frame, spec)
        tqdm.write(
            f"[{fold_number}/{len(FOLDS)}] {spec.name}: "
            f"train={len(train_indices):,}, valid={len(valid_indices):,}"
        )
        arrays, preprocessing = prepare_fold_arrays(
            frame, raw, train_indices, args.prior_strength
        )
        model = make_model(args, preprocessing, len(arsenal_features)).to(device)
        tqdm.write(f"model_parameters: {parameter_count(model):,}")
        optimizer, scheduler = make_optimizer_and_scheduler(
            model, args, len(train_indices), args.epochs
        )
        grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
        rng = np.random.default_rng(args.seed + fold_number)
        best_brier = math.inf
        best_epoch = 0
        stale_epochs = 0
        best_probability: np.ndarray | None = None
        best_mixture: np.ndarray | None = None
        checkpoint_path = checkpoint_dir / f"{spec.name}.pt"

        epoch_progress = tqdm(
            range(1, args.epochs + 1),
            desc=f"{spec.name} epochs",
            unit="epoch",
            leave=False,
        )
        for epoch in epoch_progress:
            train_losses = train_epoch(
                model,
                arrays,
                train_indices,
                optimizer,
                scheduler,
                grad_scaler,
                device,
                args,
                epoch,
                rng,
            )
            probability, mixture = predict_indices(
                model,
                arrays,
                valid_indices,
                device,
                args.eval_batch_size,
                f"epoch {epoch:02d} valid",
            )
            metrics = brier_metrics(
                frame, raw, valid_indices, probability, mixture
            )
            current_brier = float(metrics["brier"])
            improved = current_brier < best_brier - 1e-7
            if improved:
                best_brier = current_brier
                best_epoch = epoch
                stale_epochs = 0
                best_probability = probability.copy()
                best_mixture = mixture.copy()
                save_checkpoint_atomic(
                    checkpoint_payload(
                        model,
                        preprocessing,
                        arsenal_features,
                        epoch,
                        spec.name,
                        args,
                    ),
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
            epoch_progress.set_postfix(
                loss=f"{train_losses['total']:.2e}",
                valid_brier=f"{current_brier:.6f}",
                valid_score=f"{float(metrics['raw_score']):.1f}",
            )
            tqdm.write(
                f"{spec.name} epoch={epoch:02d} "
                f"loss={train_losses['total']:.2e} "
                f"train_brier={train_losses['brier']:.6f} "
                f"train_score={train_losses['raw_score']:.1f} "
                f"valid_brier={current_brier:.6f} "
                f"valid_score={float(metrics['raw_score']):.1f}"
            )
            if stale_epochs >= args.patience:
                tqdm.write(
                    f"{spec.name}: early stopping after {stale_epochs} stale epochs"
                )
                break

        if best_probability is None or best_mixture is None:
            raise RuntimeError(f"No checkpoint was selected for {spec.name}")
        final_metrics = brier_metrics(
            frame, raw, valid_indices, best_probability, best_mixture
        )
        fold_rows.append(
            {
                "fold": spec.name,
                "fold_weight": spec.weight,
                "best_epoch": best_epoch,
                **final_metrics,
            }
        )
        prediction = pd.DataFrame(
            {
                "fold": spec.name,
                "row_id": frame["row_id"].iloc[valid_indices].to_numpy(),
                "game_type": frame["game_type"].iloc[valid_indices].to_numpy(),
                "target": raw.target[valid_indices],
                "probability": best_probability,
            }
        )
        for group_id, group in enumerate(("fastball", "breaking", "offspeed", "other")):
            prediction[f"p_select_{group}"] = best_mixture[:, group_id]
        prediction_frames.append(prediction)
        del arrays, model, optimizer, scheduler, grad_scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_frame = pd.DataFrame(fold_rows)
    metrics_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        args.output_dir / "validation_predictions.parquet",
        index=False,
        compression="zstd",
    )
    weights = metrics_frame["fold_weight"].to_numpy(np.float64)
    summary = {
        "weighted_brier": float(
            np.average(metrics_frame["brier"], weights=weights)
        ),
        "weighted_raw_score": float(
            np.average(metrics_frame["raw_score"], weights=weights)
        ),
        "weighted_r_brier": float(
            np.average(metrics_frame["r_brier"], weights=weights)
        ),
        "weighted_f_brier": float(
            np.average(metrics_frame["f_brier"], weights=weights)
        ),
        "weighted_best_epoch": float(
            np.average(metrics_frame["best_epoch"], weights=weights)
        ),
        "folds": fold_rows,
        "auxiliary_input_guard": (
            "current-pitch type and physics were used only in auxiliary losses"
        ),
    }
    (args.output_dir / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics_frame.to_string(index=False))
    print(json.dumps({key: value for key, value in summary.items() if key != "folds"}, indent=2))


def final_epoch_count(args: argparse.Namespace) -> int:
    if not args.use_cv_epoch:
        return args.epochs
    summary_path = args.output_dir / "cv_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"--use-cv-epoch requires an existing {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return max(1, int(round(float(summary["weighted_best_epoch"]))))


def train_final(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    raw: RawArrays,
    arsenal_features: list[str],
    device: torch.device,
) -> None:
    epochs = final_epoch_count(args)
    train_indices = np.arange(len(frame), dtype=np.int64)
    arrays, preprocessing = prepare_fold_arrays(
        frame, raw, train_indices, args.prior_strength
    )
    model = make_model(args, preprocessing, len(arsenal_features)).to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, args, len(train_indices), epochs
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
    rng = np.random.default_rng(args.seed)
    print(
        f"final training: rows={len(train_indices):,}, epochs={epochs}, "
        f"parameters={parameter_count(model):,}"
    )
    history: list[dict[str, Any]] = []
    progress = tqdm(range(1, epochs + 1), desc="final epochs", unit="epoch")
    for epoch in progress:
        losses = train_epoch(
            model,
            arrays,
            train_indices,
            optimizer,
            scheduler,
            grad_scaler,
            device,
            args,
            epoch,
            rng,
        )
        history.append({"epoch": epoch, **losses})
        progress.set_postfix(
            loss=f"{losses['total']:.2e}",
            train_brier=f"{losses['brier']:.6f}",
            train_score=f"{losses['raw_score']:.1f}",
        )
        tqdm.write(
            f"final epoch={epoch:02d} loss={losses['total']:.2e} "
            f"train_brier={losses['brier']:.6f} "
            f"train_score={losses['raw_score']:.1f}"
        )

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint_atomic(
        checkpoint_payload(
            model,
            preprocessing,
            arsenal_features,
            epochs,
            "final_2019_2024",
            args,
        ),
        final_dir / "model.pt",
    )
    summary = {
        "rows": len(train_indices),
        "epochs": epochs,
        "parameters": parameter_count(model),
        "history": history,
        "checkpoint": str(final_dir / "model.pt"),
    }
    (final_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint: {final_dir / 'model.pt'}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.train = args.train.expanduser().resolve()
    args.arsenal_dir = args.arsenal_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    required_paths = [
        args.train,
        args.arsenal_dir / "arsenal_feature_manifest.json",
        args.arsenal_dir / "pitcher_arsenal_by_season.parquet",
        args.arsenal_dir / "team_hand_arsenal_by_season.parquet",
        args.arsenal_dir / "league_hand_arsenal_by_season.parquet",
        args.arsenal_dir / "league_arsenal_by_season.parquet",
        args.arsenal_dir / "aligned_pitch_auxiliary.parquet",
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_marker = (
        args.output_dir / "cv_summary.json"
        if args.mode == "cv"
        else args.output_dir / "final" / "model.pt"
    )
    if mode_marker.exists() and not args.force:
        raise FileExistsError(f"Output already exists; use --force: {mode_marker}")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    if (
        device.type == "cuda"
        and not args.no_amp
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("Selected CUDA device does not support bfloat16")
    print(f"device: {device}")
    print(f"precision: {'bfloat16' if device.type == 'cuda' and not args.no_amp else 'float32'}")
    print("stage 1/3: loading train context and auxiliary labels")
    frame = load_main_frame(args.train)
    print(f"train_rows: {len(frame):,}")
    print("stage 2/3: resolving temporal arsenal tokens")
    raw, arsenal_features = prepare_raw_arrays(frame, args.arsenal_dir)
    print(
        f"arsenal_shape: {raw.arsenal.shape}; "
        f"aux_rows: {(raw.aux_group >= 0).sum():,}"
    )
    print(f"stage 3/3: {'temporal CV' if args.mode == 'cv' else 'final training'}")
    if args.mode == "cv":
        train_cv(args, frame, raw, arsenal_features, device)
    else:
        train_final(args, frame, raw, arsenal_features, device)
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
