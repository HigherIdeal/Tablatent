from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_2025_proxy_validation as proxy_core
import run_context_interaction_screen as context_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


RECENT_FLAG = "regime_recent"
FAST_CONT = [
    "rr_fastball_hand1",
    "rr_fastball_hand2",
    "ro_fastball_hand1",
    "ro_fastball_hand2",
]
RANGE_CONT = ["rr_recent_range", "ro_recent_range"]
FAST_CAT = "regime_r_fastball_hand_q6"
RANGE_CAT = "regime_r_recent_range_q6"
EXTRA_CATEGORICAL = {FAST_CAT, RANGE_CAT}
EXTRA_CATEGORICAL.update(context_core.INTERACTION_COLUMNS)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    train_policy: str  # recent | full
    extras: tuple[str, ...]
    description: str


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("recent_base", "recent", (), "post-2023 training, existing raw game_type baseline"),
    VariantSpec(
        "recent_both_cont",
        "recent",
        tuple(FAST_CONT + RANGE_CONT),
        "recent-only baseline plus explicit R fastball-hand and recent-range masked continuous features",
    ),
    VariantSpec("full_base", "full", (), "all available history with raw game_type, no regime engineering"),
    VariantSpec("full_regime_flag", "full", (RECENT_FLAG,), "all history plus old/recent regime indicator"),
    VariantSpec(
        "full_fast_cont",
        "full",
        tuple([RECENT_FLAG] + FAST_CONT),
        "all history plus regime-aware R fastball x batter-hand continuous masks",
    ),
    VariantSpec(
        "full_range_cont",
        "full",
        tuple([RECENT_FLAG] + RANGE_CONT),
        "all history plus regime-aware R recent-form range continuous masks",
    ),
    VariantSpec(
        "full_both_cont",
        "full",
        tuple([RECENT_FLAG] + FAST_CONT + RANGE_CONT),
        "all history plus both continuous regime mechanisms",
    ),
    VariantSpec(
        "full_fast_cat6",
        "full",
        (RECENT_FLAG, FAST_CAT),
        "all history plus 6-bin categorical regime x R x fastball x batter-hand cross",
    ),
    VariantSpec(
        "full_range_cat6",
        "full",
        (RECENT_FLAG, RANGE_CAT),
        "all history plus 6-bin categorical regime x R x recent-range cross",
    ),
    VariantSpec(
        "full_both_cat6",
        "full",
        (RECENT_FLAG, FAST_CAT, RANGE_CAT),
        "all history plus both 6-bin categorical regime mechanisms",
    ),
    VariantSpec(
        "full_both_hybrid",
        "full",
        tuple([RECENT_FLAG] + FAST_CONT + RANGE_CONT + [FAST_CAT, RANGE_CAT]),
        "all history plus continuous and categorical representations of both mechanisms",
    ),
    VariantSpec(
        "full_both_cont_count_hand",
        "full",
        tuple([RECENT_FLAG] + FAST_CONT + RANGE_CONT + ["ctx_count_hand"]),
        "best continuous regime candidate plus count x hand context",
    ),
    VariantSpec(
        "full_both_cont_count_base",
        "full",
        tuple([RECENT_FLAG] + FAST_CONT + RANGE_CONT + ["ctx_count_base"]),
        "best continuous regime candidate plus count x base-state context",
    ),
    VariantSpec(
        "full_both_cont_context",
        "full",
        tuple([RECENT_FLAG] + FAST_CONT + RANGE_CONT + context_core.INTERACTION_COLUMNS),
        "best continuous regime candidate plus the compact target-free context crosses",
    ),
)
VARIANT_LOOKUP = {spec.name: spec for spec in VARIANTS}


def parse_ints(value: str) -> list[int]:
    result = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not result or any(x <= 0 for x in result):
        raise ValueError("iterations-grid must contain positive integers")
    return result


def parse_variants(value: str) -> list[VariantSpec]:
    if value.strip().lower() == "all":
        return list(VARIANTS)
    names = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [name for name in names if name not in VARIANT_LOOKUP]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choices={list(VARIANT_LOOKUP)}")
    return [VARIANT_LOOKUP[name] for name in names]


def parse_devices(value: str) -> list[str]:
    tokens = [x.strip() for x in value.replace(":", ",").split(",") if x.strip()]
    return tokens or ["0"]


def _token(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def fit_quantile_edges(series: pd.Series, bins: int = 6) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"No finite values for quantile binning: {series.name}")
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        raise ValueError(f"Insufficient unique values for quantile binning: {series.name}")
    return edges.astype(np.float64)


def apply_quantile_codes(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
    codes = np.full(len(values), -1, dtype=np.int16)
    finite = np.isfinite(values)
    codes[finite] = np.searchsorted(edges[1:-1], values[finite], side="right").astype(np.int16)
    return codes


def _hand_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hand = _token(frame["batter_hand"])
    # The observed dataset uses two hand codes. Keeping string comparison makes the
    # engineered features inference-safe if the source column is categorical.
    levels = sorted([x for x in hand.unique().tolist() if x != "<MISSING>"])
    if len(levels) < 2:
        raise ValueError(f"Expected at least two batter_hand levels, got {levels}")
    return hand.eq(levels[0]).to_numpy(), hand.eq(levels[1]).to_numpy()


def add_regime_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    season_col: str,
    recent_start: int = 2023,
    bins: int = 6,
) -> dict[str, object]:
    """Add target-free regime features to train/valid in place.

    Quantile edges are fit on training rows only. Continuous masks deliberately
    separate old-R and recent-R relationships so the model is not forced to use
    one slope/threshold function across the 2023 structural break.
    """
    required = {
        season_col,
        "game_type",
        "batter_hand",
        "asof_pitcher_fastball_rate",
        "eng_ps_recent_range_135",
    }
    missing = sorted(required - set(train.columns) | required - set(valid.columns))
    if missing:
        raise ValueError(f"Missing regime feature source columns: {missing}")

    for frame in (train, valid):
        season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
        frame[RECENT_FLAG] = season.ge(recent_start).astype(np.float32)

        is_r = _token(frame["game_type"]).eq("R").to_numpy()
        recent = season.ge(recent_start).to_numpy()
        old = ~recent
        h1, h2 = _hand_masks(frame)
        fast = pd.to_numeric(frame["asof_pitcher_fastball_rate"], errors="coerce").to_numpy(np.float32)
        recent_range = pd.to_numeric(frame["eng_ps_recent_range_135"], errors="coerce").to_numpy(np.float32)

        masks = {
            "rr_fastball_hand1": is_r & recent & h1,
            "rr_fastball_hand2": is_r & recent & h2,
            "ro_fastball_hand1": is_r & old & h1,
            "ro_fastball_hand2": is_r & old & h2,
        }
        for name, mask in masks.items():
            out = np.full(len(frame), np.nan, dtype=np.float32)
            out[mask] = fast[mask]
            frame[name] = out

        rr = np.full(len(frame), np.nan, dtype=np.float32)
        ro = np.full(len(frame), np.nan, dtype=np.float32)
        rr[is_r & recent] = recent_range[is_r & recent]
        ro[is_r & old] = recent_range[is_r & old]
        frame["rr_recent_range"] = rr
        frame["ro_recent_range"] = ro

    fast_edges = fit_quantile_edges(train["asof_pitcher_fastball_rate"], bins=bins)
    range_edges = fit_quantile_edges(train["eng_ps_recent_range_135"], bins=bins)

    for frame in (train, valid):
        season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
        is_r = _token(frame["game_type"]).eq("R").to_numpy()
        recent = season.ge(recent_start).to_numpy()
        fast_code = apply_quantile_codes(frame["asof_pitcher_fastball_rate"], fast_edges)
        range_code = apply_quantile_codes(frame["eng_ps_recent_range_135"], range_edges)
        hand = _token(frame["batter_hand"]).to_numpy()

        fast_cat = np.full(len(frame), "NON_R", dtype=object)
        range_cat = np.full(len(frame), "NON_R", dtype=object)
        for i in np.flatnonzero(is_r):
            regime = "RECENT_R" if recent[i] else "OLD_R"
            fq = "MISS" if fast_code[i] < 0 else f"Q{int(fast_code[i]) + 1}"
            rq = "MISS" if range_code[i] < 0 else f"Q{int(range_code[i]) + 1}"
            fast_cat[i] = f"{regime}|H={hand[i]}|FAST={fq}"
            range_cat[i] = f"{regime}|RANGE={rq}"
        frame[FAST_CAT] = pd.Series(fast_cat, index=frame.index, dtype="string").astype(str)
        frame[RANGE_CAT] = pd.Series(range_cat, index=frame.index, dtype="string").astype(str)

    return {
        "fastball_edges": fast_edges.tolist(),
        "recent_range_edges": range_edges.tolist(),
        "bins": bins,
        "recent_start": recent_start,
    }


def prepare_x(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = frame.loc[:, features].copy()
    categorical_set = set(CANONICAL_CATEGORICAL) | EXTRA_CATEGORICAL
    categorical = [feature for feature in features if feature in categorical_set]
    cat_lookup = set(categorical)
    for column in features:
        if column in cat_lookup:
            x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype(np.float32)
            x[column] = x[column].replace([np.inf, -np.inf], np.nan)
    return x, categorical


def make_diagnostic_groups(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    bins: int = 6,
) -> dict[str, np.ndarray]:
    """Build target-free R-only diagnostic groups using edges fit on training only."""
    fast_edges = fit_quantile_edges(train["asof_pitcher_fastball_rate"], bins=bins)
    range_edges = fit_quantile_edges(train["eng_ps_recent_range_135"], bins=bins)
    fast_codes = apply_quantile_codes(valid["asof_pitcher_fastball_rate"], fast_edges)
    range_codes = apply_quantile_codes(valid["eng_ps_recent_range_135"], range_edges)
    is_r = _token(valid["game_type"]).eq("R").to_numpy()
    hands, hand_levels = pd.factorize(_token(valid["batter_hand"]), sort=True)
    n_hands = max(1, len(hand_levels))
    fast_hand = fast_codes.astype(np.int32) * n_hands + hands.astype(np.int32)
    fast_hand[(~is_r) | (fast_codes < 0) | (hands < 0)] = -1
    range_codes = range_codes.astype(np.int32)
    range_codes[(~is_r) | (range_codes < 0)] = -1
    return {"fastball_hand": fast_hand, "recent_range": range_codes}


def conditional_bias_rmse(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    *,
    min_count: int = 200,
) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    valid = (groups >= 0) & np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return float("nan")
    g = groups[valid]
    residual = y[valid] - p[valid]
    n_groups = int(g.max()) + 1
    count = np.bincount(g, minlength=n_groups).astype(np.float64)
    sums = np.bincount(g, weights=residual, minlength=n_groups).astype(np.float64)
    support = count >= min_count
    if not support.any():
        return float("nan")
    means = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    global_bias = float(np.average(residual))
    return float(np.sqrt(np.average((means[support] - global_bias) ** 2, weights=count[support])))


def _subset_brier(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    d = y[mask] - p[mask]
    return float(np.mean(d * d))


def _feature_list(spec: VariantSpec, base_features: list[str]) -> list[str]:
    features = list(base_features) + list(spec.extras)
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate features in {spec.name}")
    return features


def _fit_variant_prefixes(
    *,
    spec: VariantSpec,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    base_features: list[str],
    target_col: str,
    config: dict,
    iterations_grid: list[int],
    task_type: str,
    device: str,
    verbose: int,
    thread_count: int,
    diagnostics: dict[str, np.ndarray],
) -> list[dict]:
    from catboost import CatBoostClassifier, Pool

    features = _feature_list(spec, base_features)
    x_train, categorical = prepare_x(train, features)
    x_valid, valid_categorical = prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")
    y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)

    params = context_core.catboost_params(
        config=config,
        iterations=max(iterations_grid),
        task_type=task_type,
        devices=device,
        verbose=verbose,
    )
    params["thread_count"] = int(thread_count)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)

    gt = _token(valid["game_type"]).to_numpy()
    r_mask = gt == "R"
    f_mask = gt == "F"
    rows: list[dict] = []
    for iterations in iterations_grid:
        pred = np.asarray(
            model.predict_proba(valid_pool, ntree_start=0, ntree_end=int(iterations))[:, 1],
            dtype=np.float64,
        )
        metric = probability_metrics(y_valid, pred)
        rows.append(
            {
                "variant": spec.name,
                "train_policy": spec.train_policy,
                "iterations": int(iterations),
                "train_rows": int(len(train)),
                "valid_rows": int(len(valid)),
                "feature_count": int(len(features)),
                "categorical_count": int(len(categorical)),
                "device": device if task_type == "GPU" else "CPU",
                "r_brier": _subset_brier(y_valid, pred, r_mask),
                "f_brier": _subset_brier(y_valid, pred, f_mask),
                "r_fastball_hand_bias_rmse": conditional_bias_rmse(
                    y_valid, pred, diagnostics["fastball_hand"]
                ),
                "r_recent_range_bias_rmse": conditional_bias_rmse(
                    y_valid, pred, diagnostics["recent_range"]
                ),
                **metric,
            }
        )

    del model, train_pool, valid_pool, x_train, x_valid, y_train, y_valid
    gc.collect()
    return rows


def _weighted(values: pd.Series, names: pd.Series, fold_weights: dict[str, float]) -> float:
    w = np.asarray([fold_weights[str(name)] for name in names], dtype=np.float64)
    v = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    mask = np.isfinite(v) & np.isfinite(w)
    if not mask.any():
        return float("nan")
    w = w[mask]
    w /= w.sum()
    return float(np.dot(w, v[mask]))


def build_summary(results: pd.DataFrame, fold_weights: dict[str, float]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    key = ["fold", "iterations"]
    recent_ref = results.loc[results["variant"].eq("recent_base"), key + ["brier"]].rename(
        columns={"brier": "recent_base_brier"}
    )
    full_ref = results.loc[results["variant"].eq("full_base"), key + ["brier"]].rename(
        columns={"brier": "full_base_brier"}
    )
    merged = results.merge(recent_ref, on=key, how="left").merge(full_ref, on=key, how="left")
    merged["delta_vs_recent_base"] = merged["brier"] - merged["recent_base_brier"]
    merged["delta_vs_full_base"] = merged["brier"] - merged["full_base_brier"]

    rows: list[dict] = []
    metrics = [
        "brier",
        "raw_score",
        "r_brier",
        "f_brier",
        "r_fastball_hand_bias_rmse",
        "r_recent_range_bias_rmse",
        "delta_vs_recent_base",
        "delta_vs_full_base",
    ]
    for (variant, iterations), group in merged.groupby(["variant", "iterations"], sort=False):
        row = {
            "variant": variant,
            "train_policy": str(group["train_policy"].iloc[0]),
            "iterations": int(iterations),
            "folds": int(group["fold"].nunique()),
            "improved_folds_vs_recent": int(np.count_nonzero(group["delta_vs_recent_base"].to_numpy() < 0)),
            "improved_folds_vs_full": int(np.count_nonzero(group["delta_vs_full_base"].to_numpy() < 0)),
            "worst_delta_vs_recent": float(group["delta_vs_recent_base"].max()),
            "worst_delta_vs_full": float(group["delta_vs_full_base"].max()),
        }
        for metric in metrics:
            row[f"weighted_{metric}"] = _weighted(group[metric], group["fold"], fold_weights)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["weighted_brier", "worst_delta_vs_recent"], ascending=[True, True]
    ).reset_index(drop=True)


def best_by_variant(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    ordered = summary.sort_values(["variant", "weighted_brier", "iterations"])
    return ordered.groupby("variant", as_index=False, sort=False).first().sort_values("weighted_brier")


def _save_checkpoint(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["fold", "variant", "iterations"], keep="last")
    df.sort_values(["fold", "train_policy", "variant", "iterations"]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Long-running regime-to-prediction experiment suite. Tests whether explicit 2023 regime features "
            "let all-history CatBoost recover value without forcing one old/recent conditional relationship. "
            "Each variant is fit once at max(iterations-grid) and all smaller tree counts use exact prefixes."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--iterations-grid", default="200,300,400,500")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0", help="GPU ids separated by comma or colon; one concurrent fit per GPU")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--catboost-threads", type=int, default=0, help="0=auto based on CPU cores and worker count")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/regime_feature_prediction_suite")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    variants = parse_variants(args.variants)
    iterations_grid = parse_ints(args.iterations_grid)
    devices = parse_devices(args.devices)
    season_col = config["data"]["season_col"]
    target_col = config["data"]["target_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    context_core.add_context_interactions(frame)
    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    base_features = recent_core.feature_set("recent_raw_game_type")

    fold_weights = {spec.name: spec.weight for spec in proxy_core.DEFAULT_FOLDS}
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "fold_results.partial.csv"
    final_path = output_dir / "fold_results.csv"

    rows: list[dict] = []
    if args.resume and partial_path.is_file():
        old = pd.read_csv(partial_path)
        rows = old.to_dict("records")
        print(f"[Resume] loaded {len(old):,} completed prefix rows from {partial_path}")

    completed = {
        (str(row["fold"]), str(row["variant"]), int(row["iterations"]))
        for row in rows
        if "fold" in row and "variant" in row and "iterations" in row
    }

    if args.task_type == "GPU":
        parallel_workers = len(devices)
    else:
        parallel_workers = max(1, int(args.cpu_workers))
    cpu_count = os.cpu_count() or 1
    if args.catboost_threads > 0:
        thread_count = int(args.catboost_threads)
    else:
        thread_count = max(1, cpu_count // max(1, parallel_workers))

    print("[Regime Feature Prediction Suite]")
    print(f"  catboost            : {catboost.__version__}")
    print(f"  task_type           : {args.task_type}")
    print(f"  devices             : {devices if args.task_type == 'GPU' else 'CPU'}")
    print(f"  concurrent fits     : {parallel_workers} ({'one per GPU' if args.task_type == 'GPU' else 'CPU workers'})")
    print(f"  catboost threads    : {thread_count} per fit")
    print(f"  tree prefixes       : {iterations_grid} (ONE fit at max={max(iterations_grid)} per variant/fold)")
    print(f"  variants            : {[v.name for v in variants]}")
    print("  folds               : season-forward 2024 + mid-2024 + late-2024")
    print("  checkpoint/resume   : enabled" if args.resume else "  checkpoint/resume   : disabled")
    if args.task_type == "GPU" and len(devices) == 1:
        print("  scheduler note      : one GPU => GPU fits stay sequential; prefixes/preprocessing reuse avoid waste")

    fold_metadata: list[dict] = []
    for fold_spec in proxy_core.DEFAULT_FOLDS:
        recent_mask, full_mask, valid_mask = proxy_core.fold_masks(
            frame, fold_spec, season_col, "game_month"
        )
        valid_base = frame.loc[valid_mask].copy()
        full_train_for_diag = frame.loc[full_mask].copy()
        diagnostics = make_diagnostic_groups(full_train_for_diag, valid_base, bins=6)
        y_valid = pd.to_numeric(valid_base[target_col], errors="raise").to_numpy(np.float64)

        print(
            f"\n[Fold {fold_spec.name}] valid={len(valid_base):,} rate={y_valid.mean():.6f} "
            f"recent_train={int(recent_mask.sum()):,} full_train={int(full_mask.sum()):,}"
        )
        fold_metadata.append(
            {
                "fold": fold_spec.name,
                "weight": fold_spec.weight,
                "recent_train_rows": int(recent_mask.sum()),
                "full_train_rows": int(full_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        )
        del full_train_for_diag
        gc.collect()

        for policy, mask in (("recent", recent_mask), ("full", full_mask)):
            policy_variants = [v for v in variants if v.train_policy == policy]
            if not policy_variants:
                continue
            pending = [
                v
                for v in policy_variants
                if not all((fold_spec.name, v.name, it) in completed for it in iterations_grid)
            ]
            if not pending:
                print(f"  [{policy}] all requested variants already completed; skipping")
                continue

            train = frame.loc[mask].copy()
            valid = valid_base.copy()
            feature_meta = add_regime_features(
                train,
                valid,
                season_col=season_col,
                recent_start=2023,
                bins=6,
            )
            print(
                f"  [{policy}] pending={len(pending)}/{len(policy_variants)} train={len(train):,}; "
                f"fast_edges={np.round(feature_meta['fastball_edges'], 4).tolist()}"
            )

            if args.task_type == "GPU":
                executors = {
                    device: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu{device}")
                    for device in devices
                }
                futures = []
                for idx, spec in enumerate(pending):
                    device = devices[idx % len(devices)]
                    future = executors[device].submit(
                        _fit_variant_prefixes,
                        spec=spec,
                        train=train,
                        valid=valid,
                        base_features=base_features,
                        target_col=target_col,
                        config=config,
                        iterations_grid=iterations_grid,
                        task_type=args.task_type,
                        device=device,
                        verbose=args.verbose,
                        thread_count=thread_count,
                        diagnostics=diagnostics,
                    )
                    futures.append((spec, device, future))
            else:
                executor = ThreadPoolExecutor(max_workers=parallel_workers, thread_name_prefix="cpu_fit")
                futures = []
                for spec in pending:
                    future = executor.submit(
                        _fit_variant_prefixes,
                        spec=spec,
                        train=train,
                        valid=valid,
                        base_features=base_features,
                        target_col=target_col,
                        config=config,
                        iterations_grid=iterations_grid,
                        task_type=args.task_type,
                        device="0",
                        verbose=args.verbose,
                        thread_count=thread_count,
                        diagnostics=diagnostics,
                    )
                    futures.append((spec, "CPU", future))

            future_lookup = {future: (spec, device) for spec, device, future in futures}
            try:
                for future in as_completed(future_lookup):
                    spec, device = future_lookup[future]
                    model_rows = future.result()
                    for row in model_rows:
                        row["fold"] = fold_spec.name
                        row["fold_weight"] = float(fold_spec.weight)
                        rows.append(row)
                        completed.add((fold_spec.name, spec.name, int(row["iterations"])))
                    _save_checkpoint(rows, partial_path)
                    best = min(model_rows, key=lambda row: row["brier"])
                    print(
                        f"    done {spec.name:<22} device={device:<3} "
                        f"bestTrees={best['iterations']:>3} brier={best['brier']:.8f} "
                        f"R={best['r_brier']:.8f} F={best['f_brier']:.8f} "
                        f"fastBias={best['r_fastball_hand_bias_rmse']:.5f} "
                        f"rangeBias={best['r_recent_range_bias_rmse']:.5f}",
                        flush=True,
                    )
            finally:
                if args.task_type == "GPU":
                    for executor in executors.values():
                        executor.shutdown(wait=True)
                else:
                    executor.shutdown(wait=True)

            del train, valid
            gc.collect()

        del valid_base, y_valid, diagnostics
        gc.collect()

    results = pd.DataFrame(rows).drop_duplicates(["fold", "variant", "iterations"], keep="last")
    results = results.sort_values(["fold", "train_policy", "variant", "iterations"])
    results.to_csv(final_path, index=False)
    results.to_csv(partial_path, index=False)

    summary = build_summary(results, fold_weights)
    best = best_by_variant(summary)
    summary.to_csv(output_dir / "summary_all_prefixes.csv", index=False)
    best.to_csv(output_dir / "best_by_variant.csv", index=False)

    save_json(
        {
            "seed": int(config["seed"]),
            "variants": [
                {
                    "name": v.name,
                    "train_policy": v.train_policy,
                    "extras": list(v.extras),
                    "description": v.description,
                }
                for v in variants
            ],
            "iterations_grid": iterations_grid,
            "task_type": args.task_type,
            "devices": devices if args.task_type == "GPU" else None,
            "concurrent_fits": parallel_workers,
            "catboost_threads_per_fit": thread_count,
            "fold_weights": fold_weights,
            "folds": fold_metadata,
            "canonical_invariants": invariant_check,
            "regime_definition": "season >= 2023",
            "feature_policy": "target-free; categorical quantile edges fit on each fold training rows only",
            "diagnostics": {
                "r_fastball_hand_bias_rmse": "R-only group residual RMSE after removing overall R residual bias",
                "r_recent_range_bias_rmse": "R-only recent-range group residual RMSE after removing overall R residual bias",
            },
            "scheduler": "one concurrent CatBoost fit per GPU; tree-count sweep uses exact prefixes of one max-tree model",
        },
        output_dir / "run_config.json",
    )

    print("\n[Best Prefix per Variant: lower Brier is better]")
    display_cols = [
        "variant",
        "train_policy",
        "iterations",
        "weighted_brier",
        "weighted_delta_vs_recent_base",
        "weighted_delta_vs_full_base",
        "improved_folds_vs_recent",
        "worst_delta_vs_recent",
        "weighted_r_brier",
        "weighted_f_brier",
        "weighted_r_fastball_hand_bias_rmse",
        "weighted_r_recent_range_bias_rmse",
    ]
    print(
        best[display_cols].to_string(
            index=False,
            formatters={
                "weighted_brier": "{:.8f}".format,
                "weighted_delta_vs_recent_base": "{:+.8f}".format,
                "weighted_delta_vs_full_base": "{:+.8f}".format,
                "worst_delta_vs_recent": "{:+.8f}".format,
                "weighted_r_brier": "{:.8f}".format,
                "weighted_f_brier": "{:.8f}".format,
                "weighted_r_fastball_hand_bias_rmse": "{:.6f}".format,
                "weighted_r_recent_range_bias_rmse": "{:.6f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
