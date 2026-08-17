#!/usr/bin/env python3
"""Run row-independent 2025 inference with trained Physics-Arsenal MoE models."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pitch_arsenal_moe import (  # noqa: E402
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    RAW_CONTEXT_REQUIRED,
    ModelConfig,
    PhysicsArsenalMoE,
    add_context_features,
    apply_arsenal_scaler,
    apply_scaler,
    compute_prior_logits,
    encode_categories,
    load_arsenal_tables,
    log_transform_arsenal,
    raw_context_numeric,
    regime_indices,
    resolve_arsenal,
)


DEFAULT_TEST = ROOT / "data" / "raw" / "test.csv"
DEFAULT_ARSENAL_DIR = ROOT / "data" / "processed" / "pitch_arsenal_moe"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "pitch_arsenal_moe_v2" / "final" / "model.pt"
DEFAULT_OUTPUT = ROOT / "outputs" / "pitch_arsenal_moe_v2" / "submission.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--arsenal-dir", type=Path, default=DEFAULT_ARSENAL_DIR)
    parser.add_argument(
        "--checkpoints",
        default=str(DEFAULT_CHECKPOINT),
        help="Comma-separated checkpoint paths",
    )
    parser.add_argument(
        "--blend-weights",
        default="",
        help="Optional comma-separated non-negative checkpoint weights",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def parse_checkpoint_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()).expanduser().resolve() for item in value.split(",") if item.strip()]
    if not paths:
        raise ValueError("At least one checkpoint is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def parse_weights(value: str, count: int) -> np.ndarray:
    if value.strip():
        weights = np.asarray(
            [float(item.strip()) for item in value.split(",") if item.strip()],
            dtype=np.float64,
        )
    else:
        weights = np.ones(count, dtype=np.float64)
    if len(weights) != count:
        raise ValueError("--blend-weights count must equal --checkpoints count")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Blend weights must be non-negative with at least one positive")
    return weights / weights.sum()


def load_test(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=RAW_CONTEXT_REQUIRED,
        low_memory=False,
        dtype={"row_id": "string"},
    )
    missing = sorted(set(RAW_CONTEXT_REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"test.csv missing columns: {missing}")
    if frame["row_id"].duplicated().any():
        raise ValueError("test row_id is not unique")
    return add_context_features(frame)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model_state",
        "model_config",
        "preprocessing",
        "context_categorical",
        "context_numeric",
        "arsenal_features",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint {path} missing fields: {missing}")
    if checkpoint["context_categorical"] != CONTEXT_CATEGORICAL:
        raise ValueError(f"Categorical feature contract changed for {path}")
    if checkpoint["context_numeric"] != CONTEXT_NUMERIC:
        raise ValueError(f"Numeric feature contract changed for {path}")
    return checkpoint


def predict_checkpoint(
    checkpoint: dict[str, Any],
    frame: pd.DataFrame,
    raw_context: np.ndarray,
    raw_arsenal: np.ndarray,
    arsenal_source: np.ndarray,
    regime: np.ndarray,
    device: torch.device,
    batch_size: int,
    description: str,
) -> np.ndarray:
    preprocessing = checkpoint["preprocessing"]
    categorical = encode_categories(frame, preprocessing["category_maps"])
    context = apply_scaler(raw_context, preprocessing["context_scaler"])
    arsenal = apply_arsenal_scaler(
        raw_arsenal, preprocessing["arsenal_scaler"]
    )
    prior_logit = compute_prior_logits(
        frame,
        preprocessing["regime_priors"],
        float(preprocessing["prior_strength"]),
    )
    config = ModelConfig(**checkpoint["model_config"])
    model = PhysicsArsenalMoE(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    parts: list[np.ndarray] = []
    progress = tqdm(
        range(0, len(frame), batch_size),
        total=math.ceil(len(frame) / batch_size),
        desc=description,
        unit="batch",
    )
    with torch.inference_mode():
        for start in progress:
            end = min(start + batch_size, len(frame))
            output = model(
                torch.as_tensor(categorical[start:end], dtype=torch.long, device=device),
                torch.as_tensor(context[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(arsenal[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(
                    arsenal_source[start:end], dtype=torch.long, device=device
                ),
                torch.as_tensor(regime[start:end], dtype=torch.long, device=device),
                torch.as_tensor(
                    prior_logit[start:end], dtype=torch.float32, device=device
                ),
            )
            parts.append(output["probability"].float().cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(parts).astype(np.float64, copy=False)


def main() -> None:
    args = parse_args()
    args.test = args.test.expanduser().resolve()
    args.arsenal_dir = args.arsenal_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.test.is_file():
        raise FileNotFoundError(args.test)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists; use --force: {args.output}")
    checkpoint_paths = parse_checkpoint_paths(args.checkpoints)
    weights = parse_weights(args.blend_weights, len(checkpoint_paths))
    checkpoints = [load_checkpoint(path) for path in checkpoint_paths]
    device = resolve_device(args.device)
    print(f"device: {device}")

    print("stage 1/3: loading independent test rows")
    frame = load_test(args.test)
    print(f"test_rows: {len(frame):,}")
    print("stage 2/3: resolving frozen prior-season arsenal")
    tables = load_arsenal_tables(args.arsenal_dir)
    expected_features = checkpoints[0]["arsenal_features"]
    if tables.feature_columns != expected_features:
        raise ValueError("Arsenal profile feature contract differs from checkpoint")
    for checkpoint in checkpoints[1:]:
        if checkpoint["arsenal_features"] != expected_features:
            raise ValueError("Checkpoint ensemble has inconsistent arsenal features")
    raw_arsenal, arsenal_source = resolve_arsenal(frame, tables)
    raw_arsenal = log_transform_arsenal(raw_arsenal, tables.feature_columns)
    raw_context = raw_context_numeric(frame)
    regime = regime_indices(frame)
    source_counts = Counter(int(value) for value in arsenal_source[:, 0])
    print(f"profile_source_counts: {dict(source_counts)}")

    print("stage 3/3: checkpoint inference and fixed-weight blending")
    blended = np.zeros(len(frame), dtype=np.float64)
    for index, (path, checkpoint, weight) in enumerate(
        zip(checkpoint_paths, checkpoints, weights), start=1
    ):
        prediction = predict_checkpoint(
            checkpoint,
            frame,
            raw_context,
            raw_arsenal,
            arsenal_source,
            regime,
            device,
            args.batch_size,
            f"model {index}/{len(checkpoints)}",
        )
        blended += float(weight) * prediction
        print(f"checkpoint={path} weight={weight:.6f}")
    blended = np.clip(blended, 0.0, 1.0)
    submission = pd.DataFrame(
        {"row_id": frame["row_id"].to_numpy(), "control_success": blended}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    submission.to_csv(temporary, index=False)
    os.replace(temporary, args.output)
    manifest = {
        "test_path": str(args.test),
        "output_path": str(args.output),
        "rows": len(submission),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "weights": weights.tolist(),
        "prediction_min": float(blended.min()),
        "prediction_max": float(blended.max()),
        "prediction_mean": float(blended.mean()),
        "row_independence": (
            "each prediction used only its row and frozen pre-2025 arsenal lookups"
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"submission: {args.output}")


if __name__ == "__main__":
    main()
