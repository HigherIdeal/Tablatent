from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

REFERENCE_BRIER = 0.247355098
REFERENCE_SCORE = 981.5
REFERENCE_LB = 1098.86143


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, pred={p.shape}")
    brier = float(np.mean((y - p) ** 2))
    ref = float(y.mean() * (1.0 - y.mean()))
    raw_score = float(100000.0 * (1.0 - brier / ref)) if ref > 0 else float("nan")
    return {"brier": brier, "score": raw_score, "target_mean": float(y.mean())}


def eval_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("model/metadata.json").decode("utf-8"))
    b = float(meta["validation_2024_brier"])
    s = float(meta["validation_2024_score"])
    print(f"zip               : {path}")
    print(f"validation Brier  : {b:.9f}")
    print(f"validation score  : {s:.2f}")
    print(f"vs SAFE Brier     : {b - REFERENCE_BRIER:+.9f}")
    print(f"SAFE reference    : Brier={REFERENCE_BRIER:.9f}, score={REFERENCE_SCORE:.1f}, LB={REFERENCE_LB:.5f}")
    print(f"row independent   : {meta.get('independent_test_rows')}")


def eval_npz(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    if "y" not in data or "pred" not in data:
        raise ValueError("NPZ must contain y and pred")
    m = metrics(data["y"], data["pred"])
    print(f"npz               : {path}")
    print(f"Brier             : {m['brier']:.9f}")
    print(f"raw score         : {m['score']:.2f}")
    print(f"target mean       : {m['target_mean']:.6f}")
    print(f"vs SAFE Brier     : {m['brier'] - REFERENCE_BRIER:+.9f}")
    if "game_type" in data:
        gt = data["game_type"].astype(str)
        for dom in ("R", "F"):
            mask = gt == dom
            if mask.any():
                dm = metrics(data["y"][mask], data["pred"][mask])
                print(f"{dom} Brier           : {dm['brier']:.9f}  rows={int(mask.sum()):,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Bitaboost package or saved validation NPZ.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", dest="zip_path")
    group.add_argument("--npz", dest="npz_path")
    args = ap.parse_args()
    if args.zip_path:
        eval_zip(Path(args.zip_path))
    else:
        eval_npz(Path(args.npz_path))


if __name__ == "__main__":
    main()
