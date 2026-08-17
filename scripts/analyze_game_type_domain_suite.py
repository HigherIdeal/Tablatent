from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


RESPONSE_FEATURES = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "li",
    "pitcher_team_win_expectancy",
]


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    metric = probability_metrics(y, p)
    loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    return {
        "score": float(metric["raw_score"]),
        "brier": float(metric["brier"]),
        "loss": loss,
        "target_rate": float(y.mean()),
    }


def metric_line(name: str, metric: dict[str, float]) -> str:
    return (
        f"{name:<24s} "
        f"score={metric['score']:+9.2f}  "
        f"brier={metric['brier']:.8f}  "
        f"loss={metric['loss']:.8f}"
    )


def build_params(
    *,
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    gpu_ram_part: float,
    pinned_memory_size: str,
) -> dict:
    params = context_core.catboost_params(
        config=config,
        iterations=iterations,
        task_type=task_type,
        devices=devices,
        verbose=0,
    )
    params["thread_count"] = -1
    params["metric_period"] = max(50, int(iterations))
    if task_type == "GPU":
        params["gpu_ram_part"] = float(gpu_ram_part)
        params["pinned_memory_size"] = str(pinned_memory_size)
        params["gpu_cat_features_storage"] = "GpuRam"
    return params


def fit_predict(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    label_col: str,
    features: list[str],
    params: dict,
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    train_x, categorical = context_core.prepare_x(train, features)
    valid_x, valid_categorical = context_core.prepare_x(valid, features)
    if categorical != valid_categorical:
        raise RuntimeError("categorical feature mismatch")
    train_y = pd.to_numeric(train[label_col], errors="raise").to_numpy(np.float32)

    train_pool = Pool(train_x, label=train_y, cat_features=categorical, feature_names=features)
    valid_pool = Pool(valid_x, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=False)
    pred = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)

    del model, train_pool, valid_pool, train_x, valid_x, train_y
    gc.collect()
    return pred


def domain_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "target_rate": float(y.mean()),
    }


def inning_bucket(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series("10+", index=series.index, dtype="string")
    out.loc[x <= 3] = "1-3"
    out.loc[(x >= 4) & (x <= 6)] = "4-6"
    out.loc[(x >= 7) & (x <= 9)] = "7-9"
    out.loc[x.isna()] = "NA"
    return out


def matched_context_gap(frame: pd.DataFrame, target_col: str, season_col: str) -> pd.DataFrame:
    work = frame.copy()
    work["inning_bucket"] = inning_bucket(work["inning"])
    specs = {
        "pitcher_count_hand": [
            season_col,
            "pitcher_id",
            "balls_before",
            "strikes_before",
            "outs_before",
            "pitcher_hand",
            "batter_hand",
        ],
        "pitcher_full_context": [
            season_col,
            "pitcher_id",
            "balls_before",
            "strikes_before",
            "outs_before",
            "pitcher_hand",
            "batter_hand",
            "inning_bucket",
            "top_bottom",
            "base_state",
        ],
    }

    rows: list[dict] = []
    for spec_name, keys in specs.items():
        agg = (
            work.groupby(keys + ["game_type"], observed=True)
            .agg(rows=(target_col, "size"), rate=(target_col, "mean"))
            .reset_index()
        )
        r = agg.loc[agg["game_type"].eq("R")].drop(columns="game_type").rename(
            columns={"rows": "r_rows", "rate": "r_rate"}
        )
        f = agg.loc[agg["game_type"].eq("F")].drop(columns="game_type").rename(
            columns={"rows": "f_rows", "rate": "f_rate"}
        )
        paired = r.merge(f, on=keys, how="inner")
        if paired.empty:
            continue
        paired["weight"] = np.minimum(paired["r_rows"], paired["f_rows"]).astype(float)
        paired["gap_f_minus_r"] = paired["f_rate"] - paired["r_rate"]

        for year, group in paired.groupby(season_col, sort=True):
            w = group["weight"].to_numpy(np.float64)
            gap = group["gap_f_minus_r"].to_numpy(np.float64)
            if w.sum() <= 0:
                continue
            rows.append(
                {
                    "matching": spec_name,
                    "season": int(year),
                    "matched_cells": int(len(group)),
                    "matched_weight": int(w.sum()),
                    "weighted_gap_f_minus_r": float(np.average(gap, weights=w)),
                    "median_cell_gap": float(np.median(gap)),
                    "positive_gap_share": float(np.average(gap > 0, weights=w)),
                }
            )
    return pd.DataFrame(rows).sort_values(["matching", "season"]).reset_index(drop=True)


def quantile_edges(values: np.ndarray, bins: int = 10) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if finite.size < 100:
        return None
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def response_divergence(frame: pd.DataFrame, target_col: str, season_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for year in [2022, 2023, 2024]:
        sub = frame.loc[frame[season_col].eq(year)].copy()
        if sub.empty:
            continue
        for feature in RESPONSE_FEATURES:
            if feature not in sub.columns:
                continue
            values = pd.to_numeric(sub[feature], errors="coerce").to_numpy(np.float64)
            edges = quantile_edges(values, bins=10)
            if edges is None:
                continue
            bins = pd.cut(values, bins=edges, labels=False, include_lowest=True)
            tmp = pd.DataFrame(
                {
                    "bin": bins,
                    "game_type": sub["game_type"].to_numpy(),
                    "target": pd.to_numeric(sub[target_col], errors="coerce").to_numpy(np.float64),
                }
            ).dropna(subset=["bin", "target"])
            agg = (
                tmp.groupby(["bin", "game_type"], observed=True)
                .agg(rows=("target", "size"), rate=("target", "mean"))
                .reset_index()
            )
            r = agg.loc[agg["game_type"].eq("R")].drop(columns="game_type").rename(
                columns={"rows": "r_rows", "rate": "r_rate"}
            )
            f = agg.loc[agg["game_type"].eq("F")].drop(columns="game_type").rename(
                columns={"rows": "f_rows", "rate": "f_rate"}
            )
            paired = r.merge(f, on="bin", how="inner")
            paired = paired.loc[(paired["r_rows"] >= 100) & (paired["f_rows"] >= 100)].copy()
            if paired.empty:
                continue
            paired["weight"] = np.minimum(paired["r_rows"], paired["f_rows"]).astype(float)
            paired["delta_f_minus_r"] = paired["f_rate"] - paired["r_rate"]
            w = paired["weight"].to_numpy(np.float64)
            d = paired["delta_f_minus_r"].to_numpy(np.float64)
            corr = float(np.corrcoef(paired["r_rate"], paired["f_rate"])[0, 1]) if len(paired) >= 3 else np.nan
            summary_rows.append(
                {
                    "season": int(year),
                    "feature": feature,
                    "shared_bins": int(len(paired)),
                    "support_weight": int(w.sum()),
                    "weighted_mean_delta_f_minus_r": float(np.average(d, weights=w)),
                    "weighted_mae_delta": float(np.average(np.abs(d), weights=w)),
                    "weighted_rmse_delta": float(np.sqrt(np.average(d * d, weights=w))),
                    "curve_corr": corr,
                }
            )
            for _, row in paired.iterrows():
                detail_rows.append(
                    {
                        "season": int(year),
                        "feature": feature,
                        "bin": int(row["bin"]),
                        "r_rows": int(row["r_rows"]),
                        "f_rows": int(row["f_rows"]),
                        "r_rate": float(row["r_rate"]),
                        "f_rate": float(row["f_rate"]),
                        "delta_f_minus_r": float(row["delta_f_minus_r"]),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["season", "weighted_rmse_delta"], ascending=[True, False]).reset_index(drop=True)
    return summary, pd.DataFrame(detail_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-view R/F domain audit: domain separability, cross-domain target transfer, "
            "same-pitcher context-matched target gaps, and R/F response-curve divergence."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/game_type_domain_suite")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()
    frame = frame.loc[frame["game_type"].isin(["R", "F"])].copy()
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    sort_cols = [season_col, "game_month"] + ([row_id_col] if row_id_col in frame.columns else [])
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    train_all = frame.loc[frame[season_col].between(2019, 2023)].copy()
    valid_2024 = frame.loc[frame[season_col].eq(2024)].copy()
    r_train = train_all.loc[train_all["game_type"].eq("R")].copy()
    f_train_all = train_all.loc[train_all["game_type"].eq("F")].copy()
    f_train_2023 = train_all.loc[train_all["game_type"].eq("F") & train_all[season_col].eq(2023)].copy()

    tqdm.write(
        f"Game-type domain suite | train<=2023={len(train_all):,} | valid2024={len(valid_2024):,} | "
        f"Rtrain={len(r_train):,} | Fall={len(f_train_all):,} | F2023={len(f_train_2023):,} | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | iterations={args.iterations} | catboost={catboost.__version__}"
    )

    # 1) Can pre-pitch covariates themselves identify R vs F in the next season?
    domain_train = train_all.copy()
    domain_valid = valid_2024.copy()
    domain_train["_domain_label"] = domain_train["game_type"].eq("R").astype(np.int8)
    domain_valid["_domain_label"] = domain_valid["game_type"].eq("R").astype(np.int8)

    base_no_gt = recent_core.feature_set("recent_drop_game_type")
    structural_features = [
        f for f in base_no_gt
        if f not in {"season", "pitcher_team_id", "batter_team_id"}
    ]
    full_domain_features = [f for f in base_no_gt if f != "season"]
    domain_specs = [
        ("DOMAIN_STRUCTURAL_NO_TEAM", structural_features),
        ("DOMAIN_WITH_TEAM", full_domain_features),
    ]

    domain_rows: list[dict] = []
    target_rows: list[dict] = []
    progress = tqdm(total=2 + 5, desc="game_type models", unit="model", dynamic_ncols=True)

    for name, features in domain_specs:
        seed_everything(seed)
        pred = fit_predict(
            train=domain_train,
            valid=domain_valid,
            label_col="_domain_label",
            features=features,
            params=params,
        )
        m = domain_metrics(domain_valid["_domain_label"].to_numpy(np.float64), pred)
        domain_rows.append({"experiment": name, "feature_count": len(features), **m})
        tqdm.write(
            f"{name:<24s} auc={m['auc']:.6f}  brier={m['brier']:.8f}  loss={m['loss']:.8f}"
        )
        progress.update(1)

    # 2) Cross-domain transfer matrix on the same 2024 rows.
    target_specs = [
        ("POOLED_WITH_GT", train_all, recent_core.feature_set("recent_raw_game_type")),
        ("POOLED_NO_GT", train_all, base_no_gt),
        ("R_ONLY", r_train, base_no_gt),
        ("F_ALL", f_train_all, base_no_gt),
        ("F_2023_ONLY", f_train_2023, base_no_gt),
    ]

    eval_masks = {
        "ALL_2024": np.ones(len(valid_2024), dtype=bool),
        "R_2024": valid_2024["game_type"].eq("R").to_numpy(),
        "F_2024": valid_2024["game_type"].eq("F").to_numpy(),
    }
    y_valid = pd.to_numeric(valid_2024[target_col], errors="raise").to_numpy(np.float64)

    for name, train_subset, features in target_specs:
        seed_everything(seed)
        pred = fit_predict(
            train=train_subset,
            valid=valid_2024,
            label_col=target_col,
            features=features,
            params=params,
        )
        for eval_name, mask in eval_masks.items():
            m = binary_metrics(y_valid[mask], pred[mask])
            target_rows.append(
                {
                    "model": name,
                    "train_rows": int(len(train_subset)),
                    "evaluation": eval_name,
                    **m,
                }
            )
            if eval_name in {"R_2024", "F_2024"}:
                tqdm.write(metric_line(f"{name}/{eval_name}", m))
        progress.update(1)

    progress.close()

    domain_df = pd.DataFrame(domain_rows)
    transfer_df = pd.DataFrame(target_rows)
    domain_df.to_csv(output_dir / "domain_separability.csv", index=False)
    transfer_df.to_csv(output_dir / "target_transfer_matrix.csv", index=False)

    # 3) Same-pitcher, matched pre-pitch context R/F gap.
    matched_df = matched_context_gap(frame, target_col, season_col)
    matched_df.to_csv(output_dir / "same_pitcher_matched_context_gap.csv", index=False)

    # 4) Within-season R/F response curve divergence.
    response_df, response_detail_df = response_divergence(frame, target_col, season_col)
    response_df.to_csv(output_dir / "response_curve_divergence.csv", index=False)
    response_detail_df.to_csv(output_dir / "response_curve_detail.csv", index=False)

    tqdm.write("\n[Same-pitcher matched context gaps]")
    if matched_df.empty:
        tqdm.write("no matched cells")
    else:
        tqdm.write(
            matched_df.to_string(
                index=False,
                formatters={
                    "weighted_gap_f_minus_r": "{:+.6f}".format,
                    "median_cell_gap": "{:+.6f}".format,
                    "positive_gap_share": "{:.4f}".format,
                },
            )
        )

    tqdm.write("\n[Top response-curve divergences]")
    if response_df.empty:
        tqdm.write("no response curves")
    else:
        top = response_df.groupby("season", group_keys=False).head(8)
        tqdm.write(
            top.to_string(
                index=False,
                formatters={
                    "weighted_mean_delta_f_minus_r": "{:+.6f}".format,
                    "weighted_mae_delta": "{:.6f}".format,
                    "weighted_rmse_delta": "{:.6f}".format,
                    "curve_corr": "{:.4f}".format,
                },
            )
        )

    save_json(
        {
            "experiment": "game_type R/F multi-view domain audit",
            "train_seasons_for_models": [2019, 2020, 2021, 2022, 2023],
            "validation_season": 2024,
            "f_recent_train_season": 2023,
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "gpu_ram_part": float(args.gpu_ram_part) if args.task_type == "GPU" else None,
            "pinned_memory_size": args.pinned_memory_size if args.task_type == "GPU" else None,
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
            "notes": {
                "domain_separability": "Predict R vs F from pre-pitch covariates only; no target labels used as features.",
                "transfer": "Compare pooled, R-only, all-F, and recent-F training on the same 2024 R/F subsets.",
                "matched_gap": "Compare F-R target rates within same-pitcher matched pre-pitch context cells, weighted by min(R,F) support.",
                "response_divergence": "Within-season shared quantile bins; compare target response curves for R and F.",
            },
        },
        output_dir / "run_config.json",
    )

    tqdm.write(f"\nsaved={output_dir}")


if __name__ == "__main__":
    main()
