from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_context_interaction_screen as context_core
from src.canonical_features import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, seed_everything


VARIANTS = ("raw", "quantile_pl", "gbdt_global_pl", "gbdt_dual_pl")


def parse_ints(value: str) -> list[int]:
    out = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("at least one fold is required")
    return out


def parse_variants(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(VARIANTS)
    out = [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(out) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return out


def torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    seed_everything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric_features() -> list[str]:
    categorical = set(CANONICAL_CATEGORICAL)
    return [f for f in CANONICAL_FEATURES if f not in categorical]


def fit_scaler(frame: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x = np.where(np.isfinite(x), x, median[None, :])
    q25 = np.quantile(x, 0.25, axis=0)
    q75 = np.quantile(x, 0.75, axis=0)
    scale = q75 - q25
    std = x.std(axis=0)
    fallback = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    return {"median": median.astype(np.float32), "scale": scale.astype(np.float32)}


def transform_numeric(
    frame: pd.DataFrame,
    features: list[str],
    scaler: dict[str, np.ndarray],
    clip_z: float,
) -> np.ndarray:
    x = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    x = np.where(np.isfinite(x), x, scaler["median"][None, :])
    z = (x - scaler["median"][None, :]) / scaler["scale"][None, :]
    return np.clip(z, -clip_z, clip_z).astype(np.float32)


def fit_category_maps(frame: pd.DataFrame, features: list[str]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for feature in features:
        values = frame[feature].astype("string").fillna("<MISSING>").astype(str)
        maps[feature] = {
            value: i + 1 for i, value in enumerate(sorted(values.unique().tolist()))
        }
    return maps


def transform_categories(
    frame: pd.DataFrame,
    features: list[str],
    maps: dict[str, dict[str, int]],
) -> np.ndarray:
    if not features:
        return np.zeros((len(frame), 0), dtype=np.int64)
    columns = []
    for feature in features:
        values = frame[feature].astype("string").fillna("<MISSING>").astype(str)
        columns.append(values.map(maps[feature]).fillna(0).to_numpy(np.int64))
    return np.column_stack(columns)


def cap_knots(values: list[float] | np.ndarray, max_borders: int) -> np.ndarray:
    knots = np.asarray(values, dtype=np.float64)
    knots = np.unique(np.sort(knots[np.isfinite(knots)]))
    if len(knots) > max_borders:
        ids = np.rint(np.linspace(0, len(knots) - 1, max_borders)).astype(int)
        knots = knots[np.unique(ids)]
    return knots.astype(np.float32)


def quantile_knots(
    frame: pd.DataFrame,
    features: list[str],
    max_borders: int,
) -> dict[str, np.ndarray]:
    qs = np.linspace(0.0, 1.0, max_borders + 2)[1:-1]
    result: dict[str, np.ndarray] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(np.float64)
        values = values[np.isfinite(values)]
        result[feature] = (
            cap_knots(np.quantile(values, qs), max_borders)
            if len(values)
            else np.empty(0, dtype=np.float32)
        )
    return result


def fit_catboost_borders(
    frame: pd.DataFrame,
    features: list[str],
    numerical: list[str],
    target: str,
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
    max_borders: int,
) -> dict[str, np.ndarray]:
    from catboost import CatBoostClassifier, Pool

    x, categorical = context_core.prepare_x(frame, features)
    y = pd.to_numeric(frame[target], errors="raise").to_numpy(np.float32)
    params = context_core.catboost_params(config, iterations, task_type, devices, verbose)
    params["border_count"] = max(64, max_borders)
    model = CatBoostClassifier(**params)
    pool = Pool(x, label=y, cat_features=categorical, feature_names=features)
    model.fit(pool)

    result = {feature: np.empty(0, dtype=np.float32) for feature in numerical}
    for feature_index, values in model.get_borders().items():
        feature = features[int(feature_index)]
        if feature in result:
            result[feature] = cap_knots(values, max_borders)
    del model, pool, x
    gc.collect()
    return result


def normalize_knots(
    knots: dict[str, np.ndarray],
    features: list[str],
    scaler: dict[str, np.ndarray],
    clip_z: float,
) -> dict[str, np.ndarray]:
    result = {}
    for i, feature in enumerate(features):
        values = knots.get(feature, np.empty(0, dtype=np.float32))
        z = (values - scaler["median"][i]) / scaler["scale"][i]
        result[feature] = np.clip(z, -clip_z, clip_z).astype(np.float32)
    return result


def pad_knots(
    knots: dict[str, np.ndarray],
    features: list[str],
    max_borders: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(features), max_borders), dtype=np.float32)
    mask = np.zeros_like(values)
    for i, feature in enumerate(features):
        current = cap_knots(knots.get(feature, []), max_borders)
        n = min(len(current), max_borders)
        if n:
            values[i, :n] = current[:n]
            mask[i, :n] = 1.0
    return values, mask


class HingeEncoder(nn.Module):
    """Fixed-knot continuous piecewise-linear basis.

    For each standardized feature z, expose z and ReLU(z-b_k). A downstream
    linear layer can change slope exactly at each supplied knot b_k.
    """

    def __init__(
        self,
        num_features: int,
        knot_sets: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        super().__init__()
        self.names: list[tuple[str, str]] = []
        self.output_dim = num_features
        for i, (knots, mask) in enumerate(knot_sets):
            knot_name, mask_name = f"knots_{i}", f"mask_{i}"
            self.register_buffer(knot_name, torch.from_numpy(knots))
            self.register_buffer(mask_name, torch.from_numpy(mask))
            self.names.append((knot_name, mask_name))
            self.output_dim += int(knots.size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        parts = [z]
        for knot_name, mask_name in self.names:
            knots = getattr(self, knot_name)
            mask = getattr(self, mask_name)
            hinge = torch.relu(z.unsqueeze(-1) - knots.unsqueeze(0))
            parts.append((hinge * mask.unsqueeze(0)).flatten(1))
        return torch.cat(parts, dim=1)


class PiecewiseMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        category_sizes: list[int],
        knot_sets: list[tuple[np.ndarray, np.ndarray]],
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = HingeEncoder(num_features, knot_sets)
        self.embeddings = nn.ModuleList()
        cat_dim = 0
        for size in category_sizes:
            dim = min(16, max(2, int(math.ceil(math.sqrt(size)))))
            self.embeddings.append(nn.Embedding(size, dim))
            cat_dim += dim
        half = max(32, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + cat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(half, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [self.encoder(x_num)]
        parts.extend(embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


def predict(model: PiecewiseMLP, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for x_num, x_cat, _ in loader:
            logits = model(x_num.to(device), x_cat.to(device))
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def train_model(
    xtr_num: np.ndarray,
    xtr_cat: np.ndarray,
    ytr: np.ndarray,
    xva_num: np.ndarray,
    xva_cat: np.ndarray,
    yva: np.ndarray,
    category_sizes: list[int],
    knot_sets: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]], int]:
    set_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(xtr_num),
            torch.from_numpy(xtr_cat),
            torch.from_numpy(ytr.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(xva_num),
            torch.from_numpy(xva_cat),
            torch.from_numpy(yva.astype(np.float32)),
        ),
        batch_size=args.batch_size * 2,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = PiecewiseMLP(
        xtr_num.shape[1], category_sizes, knot_sets, args.hidden_dim, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    best_state, best_brier, best_epoch, stale = None, float("inf"), 0, 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_num, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            count += len(y)

        p = predict(model, valid_loader, device)
        metrics = probability_metrics(yva, p)
        history.append(
            {
                "epoch": epoch,
                "train_bce": total_loss / max(count, 1),
                "valid_brier": metrics["brier"],
                "valid_auc": metrics["auc"],
            }
        )
        print(
            f"      epoch={epoch:02d} bce={history[-1]['train_bce']:.8f} "
            f"brier={metrics['brier']:.8f} auc={metrics['auc']:.5f}"
        )
        if metrics["brier"] < best_brier - 1e-7:
            best_brier = metrics["brier"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    return predict(model, valid_loader, device), history, best_epoch


def knot_meta(knots: dict[str, np.ndarray] | None) -> dict | None:
    if knots is None:
        return None
    return {
        feature: {
            "count": int(len(values)),
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
        for feature, values in knots.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GGPL-inspired ablation. CatBoost discovers numerical split borders; "
            "the neural model receives them only as fixed PWL hinge knots."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--border-iterations", type=int, default=200)
    parser.add_argument("--max-borders", type=int, default=24)
    parser.add_argument("--recent-seasons", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--clip-z", type=float, default=12.0)
    parser.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if args.max_borders < 1 or args.recent_seasons < 1:
        raise ValueError("max-borders and recent-seasons must be >= 1")

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    set_seed(seed)
    target = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    folds = parse_ints(args.folds)
    variants = parse_variants(args.variants)
    device = torch_device(args.torch_device)

    frame = load_frame(config).copy()
    validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame = frame.sort_values([season_col, "game_month", row_id], kind="stable").reset_index(drop=True)

    features = list(CANONICAL_FEATURES)
    numerical = numeric_features()
    categorical = [f for f in features if f in set(CANONICAL_CATEGORICAL)]
    if PITCHER_TEAM_WIN_EXPECTANCY not in numerical:
        raise RuntimeError("canonical win expectancy must be numerical")

    output_dir = Path(config["paths"]["output_dir"]) / "gbdt_guided_piecewise"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = [], {}

    print(
        f"[GBDT-guided PWL] folds={folds} variants={variants} numeric={len(numerical)} "
        f"categorical={len(categorical)} max_borders={args.max_borders} "
        f"torch={torch.__version__} device={device} catboost={catboost.__version__}"
    )
    print("[GBDT-guided PWL] CatBoost predictions/leaves/SHAP are NOT neural inputs.")
    print("[GBDT-guided PWL] Borders are fitted only on each temporal training fold.")

    for val_year in folds:
        train = frame.loc[frame[season_col] < val_year].copy()
        valid = frame.loc[frame[season_col] == val_year].copy()
        recent = train.loc[train[season_col] >= val_year - args.recent_seasons].copy()
        if train.empty or valid.empty or recent.empty:
            raise ValueError(f"Fold {val_year}: empty train/valid/recent split")

        ytr = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
        yva = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
        scaler = fit_scaler(train, numerical)
        xtr_num = transform_numeric(train, numerical, scaler, args.clip_z)
        xva_num = transform_numeric(valid, numerical, scaler, args.clip_z)
        cat_maps = fit_category_maps(train, categorical)
        xtr_cat = transform_categories(train, categorical, cat_maps)
        xva_cat = transform_categories(valid, categorical, cat_maps)
        cat_sizes = [len(cat_maps[f]) + 1 for f in categorical]

        print(
            f"\n[Fold {val_year}] train={len(train):,} recent={len(recent):,} "
            f"valid={len(valid):,} train_rate={ytr.mean():.6f} valid_rate={yva.mean():.6f}"
        )

        q_raw = quantile_knots(train, numerical, args.max_borders)
        q_norm = normalize_knots(q_raw, numerical, scaler, args.clip_z)
        global_raw = recent_raw = None
        if any(v in variants for v in ("gbdt_global_pl", "gbdt_dual_pl")):
            print("  discovering global CatBoost borders...")
            global_raw = fit_catboost_borders(
                train, features, numerical, target, config, args.border_iterations,
                args.task_type, args.devices, args.verbose, args.max_borders,
            )
        if "gbdt_dual_pl" in variants:
            print("  discovering recent CatBoost borders...")
            recent_raw = fit_catboost_borders(
                recent, features, numerical, target, config, args.border_iterations,
                args.task_type, args.devices, args.verbose, args.max_borders,
            )

        global_norm = (
            normalize_knots(global_raw, numerical, scaler, args.clip_z)
            if global_raw is not None else None
        )
        recent_norm = (
            normalize_knots(recent_raw, numerical, scaler, args.clip_z)
            if recent_raw is not None else None
        )
        metadata[str(val_year)] = {
            "recent_seasons": sorted(recent[season_col].unique().tolist()),
            "quantile": knot_meta(q_raw),
            "gbdt_global": knot_meta(global_raw),
            "gbdt_recent": knot_meta(recent_raw),
        }

        for i, variant in enumerate(variants, start=1):
            if variant == "raw":
                knot_sets = []
            elif variant == "quantile_pl":
                knot_sets = [pad_knots(q_norm, numerical, args.max_borders)]
            elif variant == "gbdt_global_pl":
                if global_norm is None:
                    raise RuntimeError("global borders missing")
                knot_sets = [pad_knots(global_norm, numerical, args.max_borders)]
            elif variant == "gbdt_dual_pl":
                if global_norm is None or recent_norm is None:
                    raise RuntimeError("dual borders missing")
                knot_sets = [
                    pad_knots(global_norm, numerical, args.max_borders),
                    pad_knots(recent_norm, numerical, args.max_borders),
                ]
            else:
                raise AssertionError(variant)

            print(f"  [{i:02d}/{len(variants):02d}] {variant}")
            p, history, best_epoch = train_model(
                xtr_num, xtr_cat, ytr, xva_num, xva_cat, yva,
                cat_sizes, knot_sets, args, device, seed,
            )
            metrics = probability_metrics(yva, p)
            rows.append({
                "fold": val_year,
                "variant": variant,
                "train_rows": len(train),
                "recent_rows": len(recent),
                "valid_rows": len(valid),
                "best_epoch": best_epoch,
                **metrics,
            })
            print(
                f"       best={best_epoch} brier={metrics['brier']:.8f} "
                f"score={metrics['raw_score']:.2f} auc={metrics['auc']:.5f} "
                f"p_std={metrics['prediction_std']:.5f}"
            )
            (output_dir / f"history_fold{val_year}_{variant}.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            del p
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "border_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary_rows = []
    raw_lookup = (
        result.loc[result["variant"] == "raw"].set_index("fold")["brier"]
        if "raw" in set(result["variant"]) else None
    )
    for variant, group in result.groupby("variant", sort=False):
        weights = group["valid_rows"].to_numpy(np.float64)
        improved = 0
        if raw_lookup is not None:
            improved = sum(
                float(row.brier) < float(raw_lookup.loc[int(row.fold)])
                for row in group.itertuples() if int(row.fold) in raw_lookup.index
            )
        summary_rows.append({
            "variant": variant,
            "folds": len(group),
            "weighted_brier": float(np.average(group["brier"], weights=weights)),
            "mean_brier": float(group["brier"].mean()),
            "mean_auc": float(group["auc"].mean()),
            "improved_vs_raw_folds": int(improved),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\n[Summary]")
    print(summary.to_string(index=False))
    print(f"saved: {output_dir / 'fold_metrics.csv'}")
    print(f"saved: {output_dir / 'summary.csv'}")
    print(f"saved: {output_dir / 'border_metadata.json'}")


if __name__ == "__main__":
    main()
