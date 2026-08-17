from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, seed_everything


STABLE_FEATURES = [
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "eng_ps_prev1_minus_long",
    "eng_ps_prev3_minus_long",
    "eng_ps_prev5_minus_long",
    "eng_ps_prev1_minus_prev3",
    "eng_ps_prev3_minus_prev5",
    "eng_ps_prev1_minus_prev5",
    "eng_ps_recent_mean_minus_long",
    "eng_ps_recent_range_135",
]
AUX_FEATURES = ["dyn_log_experience", "dyn_log_month_rows"]
VARIANTS = ("base", "lag1", "gru", "gru_lag1")


class StableGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, target_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, target_dim)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, lengths))


def parse_csv_ints(value: str) -> list[int]:
    out = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not out:
        raise ValueError("empty integer list")
    return out


def parse_variants(value: str) -> list[str]:
    out = list(VARIANTS) if value.strip().lower() == "all" else [x.strip() for x in value.split(",") if x.strip()]
    unknown = sorted(set(out) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    if "base" not in out:
        raise ValueError("base must be included so deltas are well-defined")
    return out


def torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pitcher_token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def time_index(frame: pd.DataFrame) -> pd.Series:
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    month = pd.to_numeric(frame["game_month"], errors="raise").astype(int)
    if bool((~month.between(1, 12)).any()):
        raise ValueError("game_month must be in [1, 12]")
    return season * 12 + month - 1


def build_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"pitcher_id", "season", "game_month", "asof_pitcher_n", *STABLE_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing stable-player columns: {missing}")

    work = frame[list(required)].copy()
    work["_pitcher"] = pitcher_token(work["pitcher_id"])
    work["season"] = pd.to_numeric(work["season"], errors="raise").astype(int)
    work["game_month"] = pd.to_numeric(work["game_month"], errors="raise").astype(int)
    for col in STABLE_FEATURES + ["asof_pitcher_n"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    keys = ["_pitcher", "season", "game_month"]
    monthly = work.groupby(keys, observed=True, sort=True)[STABLE_FEATURES].median().reset_index()
    exp = work.groupby(keys, observed=True, sort=True)["asof_pitcher_n"].max().rename("_exp").reset_index()
    n = work.groupby(keys, observed=True, sort=True).size().rename("_rows").reset_index()
    monthly = monthly.merge(exp, on=keys, validate="one_to_one").merge(n, on=keys, validate="one_to_one")
    monthly["dyn_log_experience"] = np.log1p(monthly["_exp"].clip(lower=0))
    monthly["dyn_log_month_rows"] = np.log1p(monthly["_rows"].clip(lower=0))
    monthly["_time"] = time_index(monthly)
    monthly = monthly.sort_values(["_pitcher", "_time"], kind="stable").reset_index(drop=True)
    if monthly.duplicated(["_pitcher", "_time"]).any():
        raise RuntimeError("duplicate pitcher-month")
    return monthly


def fit_scaler(frame: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    x = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x = np.where(np.isfinite(x), x, median[None, :])
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return {"median": median, "mean": mean, "std": std}


def scale(frame: pd.DataFrame, columns: list[str], scaler: dict[str, np.ndarray]) -> np.ndarray:
    x = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    x = np.where(np.isfinite(x), x, scaler["median"][None, :])
    return ((x - scaler["mean"][None, :]) / scaler["std"][None, :]).astype(np.float32)


def make_selfsup_tensors(
    monthly: pd.DataFrame,
    scaled: np.ndarray,
    train_max_season: int,
    lookback: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    use = monthly["season"].to_numpy(int) <= train_max_season
    target_dim = len(STABLE_FEATURES)
    xs: list[np.ndarray] = []
    lengths: list[int] = []
    ys: list[np.ndarray] = []
    times: list[int] = []

    for _, idx in monthly.loc[use].groupby("_pitcher", sort=False, observed=True).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        for j in range(1, len(pos)):
            start = max(0, j - lookback)
            history = scaled[pos[start:j]]
            padded = np.zeros((lookback, scaled.shape[1]), dtype=np.float32)
            padded[: len(history)] = history
            xs.append(padded)
            lengths.append(len(history))
            ys.append(scaled[pos[j], :target_dim])
            times.append(int(monthly.loc[pos[j], "_time"]))

    if not xs:
        raise RuntimeError("no self-supervised GRU samples")
    return (
        torch.from_numpy(np.stack(xs)),
        torch.as_tensor(lengths, dtype=torch.long),
        torch.from_numpy(np.stack(ys)),
        torch.as_tensor(times, dtype=torch.long),
    )


def train_gru(
    x: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
    times: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    seed: int,
) -> tuple[StableGRU, list[dict[str, float]]]:
    set_seed(seed)
    unique_times = np.asarray(sorted(set(times.tolist())), dtype=int)
    split = unique_times[max(1, int(len(unique_times) * 0.8))] if len(unique_times) >= 3 else unique_times[-1] + 1
    train_mask = times < split
    val_mask = times >= split
    if not bool(train_mask.any()) or not bool(val_mask.any()):
        train_mask = torch.ones_like(times, dtype=torch.bool)
        val_mask = torch.zeros_like(times, dtype=torch.bool)

    model = StableGRU(x.shape[2], hidden_dim, y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float]] = []

    train_loader = DataLoader(
        TensorDataset(x[train_mask], lengths[train_mask], y[train_mask]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = (
        DataLoader(
            TensorDataset(x[val_mask], lengths[val_mask], y[val_mask]),
            batch_size=batch_size,
        )
        if bool(val_mask.any())
        else None
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total = count = 0
        for xb, lb, yb in train_loader:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, lb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            count += len(yb)
        train_loss = total / max(count, 1)

        if val_loader is None:
            val_loss = train_loss
        else:
            model.eval()
            se = elems = 0.0
            with torch.no_grad():
                for xb, lb, yb in val_loader:
                    xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
                    diff = model(xb, lb) - yb
                    se += float((diff * diff).sum().item())
                    elems += float(yb.numel())
            val_loss = se / max(elems, 1.0)
        history.append({"epoch": epoch, "train_mse": train_loss, "valid_mse": val_loss})

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best is not None:
        model.load_state_dict(best)
    return model.eval(), history


def build_causal_state(
    monthly: pd.DataFrame,
    scaled: np.ndarray,
    model: StableGRU,
    hidden_dim: int,
    lookback: int,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    n = len(monthly)
    padded = np.zeros((n, lookback, scaled.shape[1]), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    lag = np.full((n, len(STABLE_FEATURES)), np.nan, dtype=np.float32)

    for _, idx in monthly.groupby("_pitcher", sort=False, observed=True).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        raw = monthly.loc[pos, STABLE_FEATURES].to_numpy(np.float32)
        for j, row_pos in enumerate(pos):
            if j == 0:
                continue
            start = max(0, j - lookback)
            hist = scaled[pos[start:j]]
            padded[row_pos, : len(hist)] = hist
            lengths[row_pos] = len(hist)
            lag[row_pos] = raw[j - 1]

    emb = np.zeros((n, hidden_dim), dtype=np.float32)
    valid = np.flatnonzero(lengths > 0)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(valid), batch_size):
            ids = valid[start : start + batch_size]
            xb = torch.from_numpy(padded[ids]).to(device)
            lb = torch.from_numpy(lengths[ids]).to(device)
            emb[ids] = model.encode(xb, lb).cpu().numpy().astype(np.float32)

    dyn_cols = [f"dyn_gru_{i:02d}" for i in range(hidden_dim)]
    lag_cols = [f"dyn_lag1_{name}" for name in STABLE_FEATURES]
    state = monthly[["_pitcher", "_time"]].copy()
    state[dyn_cols] = emb
    state[lag_cols] = lag
    state["dyn_history_months"] = 0.0
    state["dyn_known"] = (lengths > 0).astype(np.float32)
    for _, idx in monthly.groupby("_pitcher", sort=False, observed=True).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        state.loc[pos, "dyn_history_months"] = np.arange(len(pos), dtype=np.float32)
    return state, dyn_cols, lag_cols


def attach_state(frame: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_pitcher"] = pitcher_token(work["pitcher_id"])
    work["_time"] = time_index(work)
    work["_row_order"] = np.arange(len(work), dtype=np.int64)
    feature_cols = [c for c in state.columns if c.startswith("dyn_")]
    out = work.merge(
        state[["_pitcher", "_time", *feature_cols]],
        on=["_pitcher", "_time"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    out = (
        out.sort_values("_row_order", kind="stable")
        .drop(columns=["_pitcher", "_time", "_row_order"])
        .reset_index(drop=True)
    )
    for col in [c for c in feature_cols if c.startswith("dyn_gru_")] + [
        "dyn_history_months",
        "dyn_known",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.float32)
    return out


def fit_catboost(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    features: list[str],
    target: str,
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> tuple[np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier, Pool

    train, valid = frame.loc[train_mask], frame.loc[valid_mask]
    xtr, cat = context_core.prepare_x(train, features)
    xva, cat2 = context_core.prepare_x(valid, features)
    if cat != cat2:
        raise RuntimeError("categorical mismatch")
    ytr = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
    yva = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
    params = context_core.catboost_params(config, iterations, task_type, devices, verbose)
    model = CatBoostClassifier(**params)
    tr_pool = Pool(xtr, label=ytr, cat_features=cat, feature_names=features)
    va_pool = Pool(xva, cat_features=cat, feature_names=features)
    model.fit(tr_pool, verbose=verbose)
    pred = np.asarray(model.predict_proba(va_pool)[:, 1], dtype=np.float64)
    del model, tr_pool, va_pool, xtr, xva, ytr
    gc.collect()
    return pred, yva


def main() -> None:
    p = argparse.ArgumentParser(
        description="Causal self-supervised pitcher-month GRU ablation on low-regime-drift signals."
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--folds", default="2022,2023,2024")
    p.add_argument("--variants", default="all")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--torch-device", default="auto")
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--lookback", type=int, default=24)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--output-dir", default="outputs/stable_player_dynamics")
    args = p.parse_args()

    if min(
        args.iterations,
        args.hidden_dim,
        args.lookback,
        args.epochs,
        args.batch_size,
        args.patience,
    ) <= 0:
        raise ValueError("positive hyperparameters required")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    set_seed(seed)
    device = torch_device(args.torch_device)
    folds = parse_csv_ints(args.folds)
    variants = parse_variants(args.variants)
    target = config["data"]["target_col"]

    frame, invariant_check = recent_core.prepare_frame(config)
    if "pitcher_id" not in frame.columns:
        raise ValueError("pitcher_id is required")
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["game_month"] = pd.to_numeric(frame["game_month"], errors="raise").astype(int)
    sort_cols = ["season", "game_month"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    monthly = build_monthly(frame)
    base_features = recent_core.feature_set("recent_raw_game_type")
    input_cols = STABLE_FEATURES + AUX_FEATURES

    outdir = (ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fold_meta: list[dict[str, object]] = []

    print("[Stable Player Dynamics]")
    print(f"  rows={len(frame):,} pitcher_months={len(monthly):,} folds={folds}")
    print(
        f"  torch={torch.__version__} device={device} "
        f"stable={len(STABLE_FEATURES)} hidden={args.hidden_dim}"
    )
    print("  GRU is self-supervised: next-month stable state; it never sees control_success")

    for fold in folds:
        train_max = fold - 1
        print(f"\n[Fold {fold}] train<= {train_max}, valid={fold}")
        train_monthly = monthly[monthly["season"] <= train_max]
        scaler = fit_scaler(train_monthly, input_cols)
        scaled = scale(monthly, input_cols, scaler)
        x, lengths, y, times = make_selfsup_tensors(
            monthly, scaled, train_max, args.lookback
        )
        model, history = train_gru(
            x,
            lengths,
            y,
            times,
            args.hidden_dim,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.patience,
            device,
            seed + fold,
        )
        print(
            f"  GRU samples={len(x):,} epochs={len(history)} "
            f"best_selfsup={min(h['valid_mse'] for h in history):.6f}"
        )

        use_monthly = monthly[monthly["season"] <= fold].copy().reset_index(drop=True)
        use_scaled = scale(use_monthly, input_cols, scaler)
        state, dyn_cols, lag_cols = build_causal_state(
            use_monthly,
            use_scaled,
            model,
            args.hidden_dim,
            args.lookback,
            args.batch_size,
            device,
        )
        fold_frame = frame[frame["season"] <= fold].copy().reset_index(drop=True)
        fold_frame = attach_state(fold_frame, state)
        train_mask = fold_frame["season"] <= train_max
        valid_mask = fold_frame["season"] == fold

        feature_sets = {
            "base": base_features,
            "lag1": base_features + lag_cols + ["dyn_history_months", "dyn_known"],
            "gru": base_features + dyn_cols + ["dyn_history_months", "dyn_known"],
            "gru_lag1": base_features
            + dyn_cols
            + lag_cols
            + ["dyn_history_months", "dyn_known"],
        }
        base_brier = base_raw = None
        for variant in variants:
            pred, yva = fit_catboost(
                fold_frame,
                train_mask,
                valid_mask,
                feature_sets[variant],
                target,
                config,
                args.iterations,
                args.task_type,
                args.devices,
                args.verbose,
            )
            m = probability_metrics(yva, pred)
            if variant == "base":
                base_brier, base_raw = m["brier"], m["raw_score"]
            row = {
                "fold": fold,
                "variant": variant,
                **m,
                "delta_brier_vs_base": float(m["brier"] - base_brier),
                "delta_raw_vs_base": float(m["raw_score"] - base_raw),
            }
            rows.append(row)
            print(
                f"  {variant:<9} brier={m['brier']:.8f} raw={m['raw_score']:.2f} "
                f"dBrier={row['delta_brier_vs_base']:+.8f}"
            )

        pd.DataFrame(history).to_csv(outdir / f"gru_history_{fold}.csv", index=False)
        fold_meta.append(
            {
                "fold": fold,
                "train_max": train_max,
                "selfsup_samples": int(len(x)),
                "selfsup_best_mse": float(min(h["valid_mse"] for h in history)),
            }
        )
        del model, fold_frame, state, x, lengths, y, times
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "fold_metrics.csv", index=False)
    summary = (
        metrics.groupby("variant", sort=False)
        .agg(
            mean_brier=("brier", "mean"),
            mean_raw_score=("raw_score", "mean"),
            mean_delta_brier_vs_base=("delta_brier_vs_base", "mean"),
            worst_delta_brier_vs_base=("delta_brier_vs_base", "max"),
            improved_folds=(
                "delta_brier_vs_base",
                lambda x: int((x < 0).sum()),
            ),
        )
        .reset_index()
        .sort_values(["mean_brier", "worst_delta_brier_vs_base"])
    )
    summary.to_csv(outdir / "summary.csv", index=False)

    meta = {
        "experiment": "stable_player_dynamics_gru",
        "seed": seed,
        "stable_features": STABLE_FEATURES,
        "aux_features": AUX_FEATURES,
        "folds": folds,
        "variants": variants,
        "gru": {
            "hidden_dim": args.hidden_dim,
            "lookback": args.lookback,
            "epochs": args.epochs,
        },
        "guardrail": (
            "GRU never sees control_success; month-t state uses months strictly before t."
        ),
        "invariant_check": invariant_check,
        "fold_meta": fold_meta,
    }
    (outdir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n[Summary]")
    print(summary.to_string(index=False))
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
