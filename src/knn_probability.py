from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .data import load_frame, split_masks
from .utils import ensure_output_dirs, save_json, seed_everything


def _load_latents(config: dict) -> np.ndarray:
    latent_dir = Path(config["paths"]["output_dir"]) / "latents"
    context_path = latent_dir / "context.npy"
    history_path = latent_dir / "history.npy"
    if not context_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            "Stage 1 latent가 없습니다. 먼저 python scripts/train_stage1.py --config configs/default.yaml 를 실행하세요."
        )

    z_context = np.load(context_path, mmap_mode="r")
    z_history = np.load(history_path, mmap_mode="r")
    if len(z_context) != len(z_history):
        raise ValueError("context/history latent row 수가 다릅니다.")

    return np.concatenate(
        [np.asarray(z_context, dtype=np.float32), np.asarray(z_history, dtype=np.float32)],
        axis=1,
    )


def _make_faiss_index(train_z: np.ndarray, cfg: dict, seed: int):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss가 없습니다. pip install -r configs/requirements.txt 를 실행하세요."
        ) from exc

    dim = train_z.shape[1]
    index_type = str(cfg.get("index_type", "ivf_flat")).lower()

    if index_type == "flat":
        cpu_index = faiss.IndexFlatL2(dim)
    elif index_type == "ivf_flat":
        nlist = int(cfg.get("nlist", 4096))
        nlist = min(nlist, max(64, len(train_z) // 100))
        quantizer = faiss.IndexFlatL2(dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    else:
        raise ValueError(f"지원하지 않는 index_type: {index_type}")

    index = cpu_index
    backend = "cpu"
    if bool(cfg.get("use_gpu_if_available", True)) and hasattr(faiss, "StandardGpuResources"):
        try:
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
            backend = "gpu"
        except Exception as exc:
            print(f"[kNN] FAISS GPU 전환 실패 -> CPU 사용: {exc}")

    if not index.is_trained:
        rng = np.random.default_rng(seed)
        sample_size = min(int(cfg.get("index_train_rows", 200_000)), len(train_z))
        sample_idx = rng.choice(len(train_z), size=sample_size, replace=False)
        print(f"[kNN] training FAISS index on {sample_size:,} rows...")
        index.train(np.ascontiguousarray(train_z[sample_idx], dtype=np.float32))

    print(f"[kNN] adding {len(train_z):,} train latents to {index_type} index ({backend})...")
    index.add(np.ascontiguousarray(train_z, dtype=np.float32))

    if hasattr(index, "nprobe"):
        index.nprobe = int(cfg.get("nprobe", 32))

    return index, backend


def _search_predictions(
    index,
    query_z: np.ndarray,
    train_y: np.ndarray,
    ks: list[int],
    batch_size: int,
    desc: str,
):
    max_k = max(ks)
    preds = {k: np.empty(len(query_z), dtype=np.float32) for k in ks}
    first_neighbors = None
    first_distances = None

    for start in tqdm(range(0, len(query_z), batch_size), desc=desc):
        stop = min(start + batch_size, len(query_z))
        distances, indices = index.search(
            np.ascontiguousarray(query_z[start:stop], dtype=np.float32),
            max_k,
        )
        if np.any(indices < 0):
            raise RuntimeError(
                "FAISS가 충분한 이웃을 찾지 못했습니다. nprobe를 키우거나 index_type=flat을 사용하세요."
            )

        neighbor_y = train_y[indices]
        cumulative = np.cumsum(neighbor_y, axis=1, dtype=np.float32)
        for k in ks:
            preds[k][start:stop] = cumulative[:, k - 1] / float(k)

        if first_neighbors is None:
            first_neighbors = indices.copy()
            first_distances = distances.copy()

    return preds, first_neighbors, first_distances


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(p - y)))


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    result = []
    for i in range(bins):
        mask = bucket == i
        result.append(
            {
                "bin": i,
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "rows": int(mask.sum()),
                "mean_prediction": float(p[mask].mean()) if mask.any() else None,
                "actual_rate": float(y[mask].mean()) if mask.any() else None,
            }
        )
    return result


def evaluate_latent_knn(config: dict, evaluate_test: bool = False) -> dict:
    seed_everything(config["seed"])
    frame = load_frame(config)
    split = split_masks(frame, config)
    out = ensure_output_dirs(config["paths"]["output_dir"])

    cfg = config.get("knn_probability", {})
    ks = sorted({int(k) for k in cfg.get("k_values", [20, 50, 100, 200, 500, 1000])})
    if not ks or min(ks) <= 0:
        raise ValueError("k_values는 양의 정수여야 합니다.")

    z = _load_latents(config)
    if len(z) != len(frame):
        raise ValueError(f"latent rows={len(z):,}, frame rows={len(frame):,} 불일치")

    target_col = config["data"]["target_col"]
    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=np.float32)

    train_mask = split["train"]
    val_mask = split["val"]
    test_mask = split["test"]
    train_global_idx = np.flatnonzero(train_mask)

    # Distance geometry must be learned from the neighbor pool only.
    scaler = StandardScaler()
    train_z = scaler.fit_transform(z[train_mask]).astype(np.float32, copy=False)
    val_z = scaler.transform(z[val_mask]).astype(np.float32, copy=False)
    train_y = y[train_mask]
    val_y = y[val_mask]

    index, backend = _make_faiss_index(train_z, cfg, config["seed"])
    batch_size = int(cfg.get("query_batch_size", 4096))

    val_preds, _, _ = _search_predictions(
        index,
        val_z,
        train_y,
        ks,
        batch_size,
        "2023 latent kNN",
    )

    train_mean = float(train_y.mean())
    baseline_pred = np.full(len(val_y), train_mean, dtype=np.float32)
    baseline_brier = _brier(val_y, baseline_pred)

    validation = {}
    for k in ks:
        score = _brier(val_y, val_preds[k])
        validation[str(k)] = {
            "brier": score,
            "skill_vs_train_mean": float(1.0 - score / baseline_brier),
            "prediction_mean": float(val_preds[k].mean()),
            "prediction_std": float(val_preds[k].std()),
        }

    best_k = min(ks, key=lambda k: validation[str(k)]["brier"])
    best_val_pred = val_preds[best_k]

    result = {
        "neighbor_pool": {
            "seasons": config["data"]["train_seasons"],
            "rows": int(train_mask.sum()),
            "target_mean": train_mean,
        },
        "validation": {
            "seasons": config["data"]["val_seasons"],
            "rows": int(val_mask.sum()),
            "baseline_train_mean_brier": baseline_brier,
            "k_results": validation,
            "best_k": int(best_k),
            "best_calibration": _calibration(val_y, best_val_pred),
        },
        "faiss": {
            "index_type": str(cfg.get("index_type", "ivf_flat")),
            "backend": backend,
            "nlist": int(cfg.get("nlist", 4096)),
            "nprobe": int(cfg.get("nprobe", 32)),
        },
        "latent_dim": int(z.shape[1]),
        "standardized_on_train_only": True,
    }

    val_indices = np.flatnonzero(val_mask)
    val_output = pd.DataFrame(
        {
            "global_index": val_indices,
            "target": val_y,
            f"knn_p_k{best_k}": best_val_pred,
        }
    )
    row_id_col = config["data"].get("row_id_col")
    if row_id_col and row_id_col in frame.columns:
        val_output.insert(1, row_id_col, frame.loc[val_mask, row_id_col].to_numpy())
    val_output.to_csv(out["logs"] / "knn_validation_predictions.csv", index=False)

    # Small neighbor sanity check: inspect whether nearest latent points are semantically plausible.
    rng = np.random.default_rng(config["seed"])
    example_count = min(int(cfg.get("neighbor_example_queries", 5)), len(val_z))
    example_rank = min(int(cfg.get("neighbor_example_k", 10)), max(ks))
    query_local = rng.choice(len(val_z), size=example_count, replace=False)
    d, nn = index.search(np.ascontiguousarray(val_z[query_local]), example_rank)
    rows = []
    for q_pos, q_local in enumerate(query_local):
        query_global = val_indices[q_local]
        for rank in range(example_rank):
            neighbor_local = int(nn[q_pos, rank])
            neighbor_global = int(train_global_idx[neighbor_local])
            rows.append(
                {
                    "query_global_index": int(query_global),
                    "query_target": float(y[query_global]),
                    "rank": rank + 1,
                    "neighbor_global_index": neighbor_global,
                    "neighbor_target": float(y[neighbor_global]),
                    "squared_l2_distance": float(d[q_pos, rank]),
                }
            )
    pd.DataFrame(rows).to_csv(out["logs"] / "knn_neighbor_examples.csv", index=False)

    if evaluate_test:
        test_z = scaler.transform(z[test_mask]).astype(np.float32, copy=False)
        test_y = y[test_mask]
        test_preds, _, _ = _search_predictions(
            index,
            test_z,
            train_y,
            [best_k],
            batch_size,
            "2024 holdout latent kNN",
        )
        test_p = test_preds[best_k]
        test_baseline_brier = _brier(
            test_y, np.full(len(test_y), train_mean, dtype=np.float32)
        )
        test_brier = _brier(test_y, test_p)
        result["test"] = {
            "seasons": config["data"]["test_seasons"],
            "rows": int(test_mask.sum()),
            "k": int(best_k),
            "brier": test_brier,
            "baseline_train_mean_brier": test_baseline_brier,
            "skill_vs_train_mean": float(1.0 - test_brier / test_baseline_brier),
            "prediction_mean": float(test_p.mean()),
            "prediction_std": float(test_p.std()),
            "calibration": _calibration(test_y, test_p),
        }

        test_indices = np.flatnonzero(test_mask)
        test_output = pd.DataFrame(
            {
                "global_index": test_indices,
                "target": test_y,
                f"knn_p_k{best_k}": test_p,
            }
        )
        if row_id_col and row_id_col in frame.columns:
            test_output.insert(1, row_id_col, frame.loc[test_mask, row_id_col].to_numpy())
        test_output.to_csv(out["logs"] / "knn_test_predictions.csv", index=False)

    save_json(result, out["logs"] / "knn_probability_metrics.json")

    print("\n[kNN validation]")
    print(f"train mean baseline Brier: {baseline_brier:.8f}")
    for k in ks:
        row = validation[str(k)]
        print(
            f"k={k:4d} | Brier={row['brier']:.8f} | "
            f"skill_vs_mean={row['skill_vs_train_mean']:+.6f} | "
            f"pred_std={row['prediction_std']:.6f}"
        )
    print(f"BEST k={best_k} | Brier={validation[str(best_k)]['brier']:.8f}")
    if evaluate_test:
        print(
            f"[2024 holdout] k={best_k} | Brier={result['test']['brier']:.8f} | "
            f"skill_vs_mean={result['test']['skill_vs_train_mean']:+.6f}"
        )

    return result
