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
import run_game_type_temporal_regime_ablation as regime_core
from src.utils import load_config, save_json, seed_everything


BASE_MEMORY = [
    "eng_prev_pitcher_n",
    "eng_prev_pitcher_same_gt_n",
]
CENTERED = [
    "eng_prev_pitcher_rel_all",
    "eng_prev_pitcher_rel_same_gt",
]
PERCENTILE = [
    "eng_prev_pitcher_pct_all_centered",
    "eng_prev_pitcher_pct_same_gt_centered",
]
SHRUNK = [
    "eng_prev_pitcher_rel_all_shrunk",
    "eng_prev_pitcher_rel_same_gt_shrunk",
]


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate integers: {values}")
    return sorted(values)


def _left_join_values(
    dest: pd.DataFrame,
    table: pd.DataFrame,
    keys: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    base = dest.loc[:, keys].copy()
    base["_join_order"] = np.arange(len(base), dtype=np.int64)
    merged = base.merge(table, how="left", on=keys, sort=False)
    merged = merged.sort_values("_join_order", kind="stable")
    return merged.loc[:, value_cols].reset_index(drop=True)


def _pitcher_tables(
    source: pd.DataFrame,
    *,
    pitcher_col: str,
    target_col: str,
    n_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    global_all = float(pd.to_numeric(source[target_col], errors="raise").mean())
    global_gt = (
        source.groupby("game_type", observed=True, sort=False)[target_col]
        .mean()
        .astype(float)
        .to_dict()
    )

    p_all = (
        source.groupby(pitcher_col, observed=True, sort=False)
        .agg(
            eng_prev_pitcher_n=(target_col, "size"),
            _success=(target_col, "mean"),
            _anchor_max_n=(n_col, "max"),
        )
        .reset_index()
    )
    p_all["eng_prev_pitcher_rel_all"] = p_all["_success"] - global_all
    p_all["eng_prev_pitcher_pct_all_centered"] = (
        p_all["_success"].rank(method="average", pct=True) - 0.5
    )
    # asof_n is the number of completed prior pitches before a row. The previous
    # season's final labeled pitch advances the end state by one beyond max asof_n.
    p_all["eng_prev_pitcher_end_n"] = pd.to_numeric(
        p_all["_anchor_max_n"], errors="raise"
    ).astype(np.int64) + 1

    p_gt = (
        source.groupby([pitcher_col, "game_type"], observed=True, sort=False)[target_col]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "eng_prev_pitcher_same_gt_n", "mean": "_success"})
    )
    p_gt["_global_gt"] = p_gt["game_type"].map(global_gt).astype(float)
    p_gt["eng_prev_pitcher_rel_same_gt"] = p_gt["_success"] - p_gt["_global_gt"]
    p_gt["eng_prev_pitcher_pct_same_gt_centered"] = (
        p_gt.groupby("game_type", observed=True, sort=False)["_success"]
        .rank(method="average", pct=True)
        - 0.5
    )

    keep_all = [
        pitcher_col,
        "eng_prev_pitcher_n",
        "eng_prev_pitcher_rel_all",
        "eng_prev_pitcher_pct_all_centered",
        "eng_prev_pitcher_end_n",
    ]
    keep_gt = [
        pitcher_col,
        "game_type",
        "eng_prev_pitcher_same_gt_n",
        "eng_prev_pitcher_rel_same_gt",
        "eng_prev_pitcher_pct_same_gt_centered",
    ]
    return p_all[keep_all], p_gt[keep_gt], {
        "ALL": global_all,
        "R": float(global_gt.get("R", np.nan)),
        "F": float(global_gt.get("F", np.nan)),
    }


def add_relative_memory_features(
    frame: pd.DataFrame,
    *,
    target_col: str,
    season_col: str,
    pitcher_col: str,
    n_col: str,
    shrink_lambda: float,
    decay_lambdas: list[int],
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """Strict previous-season relative-skill memory with optional progress decay.

    Season s only reads labeled target summaries from season s-1. Current-season
    progress is computed from the current row's asof_n minus the frozen end count
    from s-1. No row in season s contributes to any other row in season s.
    """
    required = {target_col, season_col, pitcher_col, n_col, "game_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing relative-memory columns: {missing}")

    base_cols = [
        *BASE_MEMORY,
        *CENTERED,
        *PERCENTILE,
        *SHRUNK,
        "eng_prev_pitcher_end_n",
        "eng_current_season_progress_n",
    ]
    decay_cols: list[str] = []
    for lam in decay_lambdas:
        decay_cols.extend(
            [
                f"eng_prev_memory_decay_w_{lam}",
                f"eng_prev_rel_all_decayed_{lam}",
                f"eng_prev_rel_same_gt_decayed_{lam}",
            ]
        )
    for col in [*base_cols, *decay_cols]:
        frame[col] = np.nan

    diagnostics: list[dict] = []
    global_rates: dict[int, dict[str, float]] = {}
    seasons = sorted(frame[season_col].unique().tolist())

    for season in seasons:
        idx = frame.index[frame[season_col].eq(season)]
        dest = frame.loc[idx]
        source = frame.loc[frame[season_col].eq(season - 1)]
        if source.empty:
            diagnostics.append(
                {
                    "season": int(season),
                    "source_season": int(season - 1),
                    "rows": int(len(dest)),
                    "pitcher_coverage": 0.0,
                    "same_gt_coverage": 0.0,
                    "progress_valid_share": 0.0,
                    "progress_negative_share": 0.0,
                    "median_progress_n": np.nan,
                }
            )
            continue

        p_all, p_gt, rates = _pitcher_tables(
            source,
            pitcher_col=pitcher_col,
            target_col=target_col,
            n_col=n_col,
        )
        global_rates[int(season - 1)] = rates

        all_cols = [
            "eng_prev_pitcher_n",
            "eng_prev_pitcher_rel_all",
            "eng_prev_pitcher_pct_all_centered",
            "eng_prev_pitcher_end_n",
        ]
        vals = _left_join_values(dest, p_all, [pitcher_col], all_cols)
        frame.loc[idx, all_cols] = vals.to_numpy(np.float32)

        gt_cols = [
            "eng_prev_pitcher_same_gt_n",
            "eng_prev_pitcher_rel_same_gt",
            "eng_prev_pitcher_pct_same_gt_centered",
        ]
        vals = _left_join_values(dest, p_gt, [pitcher_col, "game_type"], gt_cols)
        frame.loc[idx, gt_cols] = vals.to_numpy(np.float32)

        n_all = pd.to_numeric(frame.loc[idx, "eng_prev_pitcher_n"], errors="coerce").to_numpy(np.float64)
        n_gt = pd.to_numeric(frame.loc[idx, "eng_prev_pitcher_same_gt_n"], errors="coerce").to_numpy(np.float64)
        rel_all = pd.to_numeric(frame.loc[idx, "eng_prev_pitcher_rel_all"], errors="coerce").to_numpy(np.float64)
        rel_gt = pd.to_numeric(frame.loc[idx, "eng_prev_pitcher_rel_same_gt"], errors="coerce").to_numpy(np.float64)

        shrink_all = n_all / (n_all + float(shrink_lambda))
        shrink_gt = n_gt / (n_gt + float(shrink_lambda))
        frame.loc[idx, "eng_prev_pitcher_rel_all_shrunk"] = (rel_all * shrink_all).astype(np.float32)
        frame.loc[idx, "eng_prev_pitcher_rel_same_gt_shrunk"] = (rel_gt * shrink_gt).astype(np.float32)

        current_n = pd.to_numeric(dest[n_col], errors="raise").to_numpy(np.float64)
        end_n = pd.to_numeric(frame.loc[idx, "eng_prev_pitcher_end_n"], errors="coerce").to_numpy(np.float64)
        raw_progress = current_n - end_n
        available = np.isfinite(end_n)
        negative = available & np.isfinite(raw_progress) & (raw_progress < 0.0)
        valid_progress = available & np.isfinite(raw_progress) & (raw_progress >= 0.0)
        progress = np.full(len(dest), np.nan, dtype=np.float64)
        progress[valid_progress] = raw_progress[valid_progress]
        frame.loc[idx, "eng_current_season_progress_n"] = progress.astype(np.float32)

        shrunk_all = pd.to_numeric(
            frame.loc[idx, "eng_prev_pitcher_rel_all_shrunk"], errors="coerce"
        ).to_numpy(np.float64)
        shrunk_gt = pd.to_numeric(
            frame.loc[idx, "eng_prev_pitcher_rel_same_gt_shrunk"], errors="coerce"
        ).to_numpy(np.float64)
        for lam in decay_lambdas:
            w = np.full(len(dest), np.nan, dtype=np.float64)
            w[valid_progress] = float(lam) / (float(lam) + progress[valid_progress])
            frame.loc[idx, f"eng_prev_memory_decay_w_{lam}"] = w.astype(np.float32)
            frame.loc[idx, f"eng_prev_rel_all_decayed_{lam}"] = (shrunk_all * w).astype(np.float32)
            frame.loc[idx, f"eng_prev_rel_same_gt_decayed_{lam}"] = (shrunk_gt * w).astype(np.float32)

        diagnostics.append(
            {
                "season": int(season),
                "source_season": int(season - 1),
                "rows": int(len(dest)),
                "pitcher_coverage": float(np.isfinite(rel_all).mean()),
                "same_gt_coverage": float(np.isfinite(rel_gt).mean()),
                "progress_valid_share": float(valid_progress.mean()),
                "progress_negative_share": float(negative.mean()),
                "median_progress_n": float(np.nanmedian(progress)) if valid_progress.any() else np.nan,
                "p90_progress_n": float(np.nanpercentile(progress, 90)) if valid_progress.any() else np.nan,
                "source_rate_all": rates["ALL"],
                "source_rate_R": rates["R"],
                "source_rate_F": rates["F"],
            }
        )

    for col in [*base_cols, *decay_cols]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    return pd.DataFrame(diagnostics), global_rates


def feature_sets(base_features: list[str], decay_lambdas: list[int]) -> dict[str, list[str]]:
    regime = [*base_features, "eng_recent_f"]
    out: dict[str, list[str]] = {
        "A0_REGIME": regime,
        "A1_CENTERED": [*regime, *BASE_MEMORY, *CENTERED],
        "A2_PERCENTILE": [*regime, *BASE_MEMORY, *PERCENTILE],
        "A3_SHRUNK": [*regime, *BASE_MEMORY, *SHRUNK],
    }
    for index, lam in enumerate(decay_lambdas, start=4):
        out[f"A{index}_DECAY_{lam}"] = [
            *regime,
            *BASE_MEMORY,
            "eng_current_season_progress_n",
            f"eng_prev_memory_decay_w_{lam}",
            f"eng_prev_rel_all_decayed_{lam}",
            f"eng_prev_rel_same_gt_decayed_{lam}",
        ]
    for name, features in out.items():
        if len(features) != len(set(features)):
            raise RuntimeError(f"duplicate features in {name}")
    return out


def evaluate(y: np.ndarray, gt: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    masks = {
        "ALL": np.ones(len(y), dtype=bool),
        "R": gt == "R",
        "F": gt == "F",
    }
    baseline = {
        group: regime_core.binary_metrics(y[mask], predictions["A0_REGIME"][mask])
        for group, mask in masks.items()
    }
    rows: list[dict] = []
    for variant, pred in predictions.items():
        for group, mask in masks.items():
            metric = regime_core.binary_metrics(y[mask], pred[mask])
            rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "rows": int(mask.sum()),
                    **metric,
                    "delta_brier_vs_A0_same_group": float(
                        metric["brier"] - baseline[group]["brier"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _safe_int_ids(series: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(np.int64)
    if np.any(values.astype(np.float64) != numeric.to_numpy(np.float64)):
        raise ValueError(f"{name} cannot be represented losslessly as int64")
    return values


def save_deploy_artifact(
    frame: pd.DataFrame,
    *,
    source_year: int,
    target_col: str,
    season_col: str,
    pitcher_col: str,
    n_col: str,
    path: Path,
) -> dict[str, float | int]:
    source = frame.loc[frame[season_col].eq(source_year)].copy()
    if source.empty:
        raise ValueError(f"No source rows for {source_year}")
    p_all, p_gt, rates = _pitcher_tables(
        source,
        pitcher_col=pitcher_col,
        target_col=target_col,
        n_col=n_col,
    )
    r = p_gt.loc[p_gt["game_type"].eq("R")].drop(columns="game_type").rename(
        columns={
            "eng_prev_pitcher_same_gt_n": "r_n",
            "eng_prev_pitcher_rel_same_gt": "r_rel",
            "eng_prev_pitcher_pct_same_gt_centered": "r_pct",
        }
    )
    f = p_gt.loc[p_gt["game_type"].eq("F")].drop(columns="game_type").rename(
        columns={
            "eng_prev_pitcher_same_gt_n": "f_n",
            "eng_prev_pitcher_rel_same_gt": "f_rel",
            "eng_prev_pitcher_pct_same_gt_centered": "f_pct",
        }
    )
    p = p_all.merge(r, on=pitcher_col, how="left").merge(f, on=pitcher_col, how="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_year=np.asarray([source_year], dtype=np.int16),
        global_rate_all=np.asarray([rates["ALL"]], dtype=np.float32),
        global_rate_R=np.asarray([rates["R"]], dtype=np.float32),
        global_rate_F=np.asarray([rates["F"]], dtype=np.float32),
        pitcher_id=_safe_int_ids(p[pitcher_col], pitcher_col),
        pitcher_n=p["eng_prev_pitcher_n"].fillna(0).to_numpy(np.int32),
        pitcher_end_n=p["eng_prev_pitcher_end_n"].fillna(0).to_numpy(np.int32),
        pitcher_rel_all=p["eng_prev_pitcher_rel_all"].to_numpy(np.float32),
        pitcher_pct_all=p["eng_prev_pitcher_pct_all_centered"].to_numpy(np.float32),
        pitcher_r_n=p["r_n"].fillna(0).to_numpy(np.int32),
        pitcher_r_rel=p["r_rel"].to_numpy(np.float32),
        pitcher_r_pct=p["r_pct"].to_numpy(np.float32),
        pitcher_f_n=p["f_n"].fillna(0).to_numpy(np.int32),
        pitcher_f_rel=p["f_rel"].to_numpy(np.float32),
        pitcher_f_pct=p["f_pct"].to_numpy(np.float32),
    )
    size = int(path.stat().st_size)
    return {"pitchers": int(len(p)), "bytes": size, "kib": float(size / 1024.0)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict forward-fold relative previous-season memory. Replace raw last-season rates "
            "with season/domain-centered pitcher skill, optional empirical shrinkage, and a decay "
            "that fades memory as current-season asof_pitcher_n accumulates."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--shrink-lambda", type=float, default=200.0)
    parser.add_argument("--decay-lambdas", default="100,300,1000")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/relative_decayed_cheatsheet_probe")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    folds = parse_ints(args.folds)
    decay_lambdas = parse_ints(args.decay_lambdas)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.shrink_lambda <= 0:
        raise ValueError("--shrink-lambda must be positive")
    if any(x <= 0 for x in decay_lambdas):
        raise ValueError("decay lambdas must be positive")
    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05,1.0]")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    pitcher_col = "pitcher_id"
    n_col = "asof_pitcher_n"

    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame[n_col] = pd.to_numeric(frame[n_col], errors="raise").astype(np.int64)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()
    unexpected = sorted(set(frame["game_type"].dropna().unique()) - {"R", "F"})
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")

    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    memory_diag, global_rates = add_relative_memory_features(
        frame,
        target_col=target_col,
        season_col=season_col,
        pitcher_col=pitcher_col,
        n_col=n_col,
        shrink_lambda=float(args.shrink_lambda),
        decay_lambdas=decay_lambdas,
    )
    regime_core.add_regime_features(
        frame,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )

    base_features = recent_core.feature_set("recent_raw_game_type")
    variants = feature_sets(base_features, decay_lambdas)
    params = regime_core.build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_diag.to_csv(output_dir / "memory_diagnostics.csv", index=False)

    deploy_path = output_dir / "deploy_2025_relative_memory_from_2024.npz"
    deploy_info = save_deploy_artifact(
        frame,
        source_year=2024,
        target_col=target_col,
        season_col=season_col,
        pitcher_col=pitcher_col,
        n_col=n_col,
        path=deploy_path,
    )

    tqdm.write(
        f"Relative decayed cheatsheet | folds={folds} | rows={len(frame):,} | "
        f"shrink_lambda={args.shrink_lambda:g} | decay_lambdas={decay_lambdas} | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | iterations={args.iterations} | "
        f"catboost={catboost.__version__}"
    )
    tqdm.write(
        "STRICT RULE: season s memory is built from season s-1 labels only; current-season progress "
        "uses only the current row asof_n plus the frozen previous-season end count."
    )
    tqdm.write(
        f"compact deploy artifact={deploy_path.name} size={deploy_info['kib']:.1f} KiB "
        f"pitchers={deploy_info['pitchers']}"
    )
    tqdm.write("\n[Memory diagnostics]")
    for _, row in memory_diag.iterrows():
        tqdm.write(
            f"season={int(row['season'])} src={int(row['source_season'])} rows={int(row['rows']):,} "
            f"pitcher={float(row['pitcher_coverage']):.4f} sameGT={float(row['same_gt_coverage']):.4f} "
            f"progress={float(row['progress_valid_share']):.4f} neg={float(row['progress_negative_share']):.4f} "
            f"median_progress={float(row['median_progress_n']):.1f}"
        )

    all_results: list[pd.DataFrame] = []
    progress_bar = tqdm(
        total=len(folds) * len(variants),
        desc="relative-memory models",
        unit="model",
        dynamic_ncols=True,
    )
    for val_year in folds:
        train = frame.loc[frame[season_col] < val_year].copy()
        valid = frame.loc[frame[season_col].eq(val_year)].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train/valid")
        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt = valid["game_type"].astype(str).to_numpy()
        preds: dict[str, np.ndarray] = {}
        for variant, features in variants.items():
            seed_everything(seed)
            preds[variant] = regime_core.fit_predict(
                train=train,
                valid=valid,
                target_col=target_col,
                features=features,
                extra_categorical=set(),
                params=params,
            )
            progress_bar.update(1)
        fold_df = evaluate(y, gt, preds)
        fold_df.insert(0, "validation_year", int(val_year))
        all_results.append(fold_df)
        del train, valid, y, gt, preds
        gc.collect()
    progress_bar.close()

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(output_dir / "fold_results.csv", index=False)
    all_only = results.loc[results["group"].eq("ALL")]
    summary = (
        all_only.groupby("variant", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_brier=("brier", "mean"),
            mean_dB=("delta_brier_vs_A0_same_group", "mean"),
            worst_dB=("delta_brier_vs_A0_same_group", "max"),
            best_dB=("delta_brier_vs_A0_same_group", "min"),
            wins=("delta_brier_vs_A0_same_group", lambda x: int((x < 0).sum())),
        )
        .sort_values(["mean_dB", "worst_dB"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "experiment": "season/domain-centered previous-season pitcher memory with shrinkage and current-season decay",
            "folds": folds,
            "variants": variants,
            "shrink_lambda": float(args.shrink_lambda),
            "decay_lambdas": decay_lambdas,
            "global_rates_by_source_year": global_rates,
            "deploy_artifact": deploy_info,
            "strict_inference": "previous-season labels + current row asof_n only; no peer validation/test rows",
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Fold results | ALL]")
    for val_year in folds:
        part = results.loc[
            results["validation_year"].eq(val_year) & results["group"].eq("ALL")
        ].sort_values("brier")
        tqdm.write(f"fold={val_year}")
        for _, row in part.iterrows():
            tqdm.write(
                regime_core.metric_line(
                    str(row["variant"]),
                    {"score": float(row["score"]), "brier": float(row["brier"]), "loss": float(row["loss"])},
                    float(row["delta_brier_vs_A0_same_group"]),
                )
            )

    tqdm.write("\n[Cross-fold summary | ALL]")
    for _, row in summary.iterrows():
        tqdm.write(
            f"{str(row['variant']):<28s} mean_dB={float(row['mean_dB']):+.8f} "
            f"worst_dB={float(row['worst_dB']):+.8f} best_dB={float(row['best_dB']):+.8f} "
            f"wins={int(row['wins'])}/{int(row['folds'])}"
        )

    if 2024 in folds:
        tqdm.write("\n[2024 diagnostics | R/F]")
        for group in ("R", "F"):
            part = results.loc[
                results["validation_year"].eq(2024) & results["group"].eq(group)
            ].sort_values("brier")
            best = part.iloc[0]
            tqdm.write(
                regime_core.metric_line(
                    f"best/{group} {best['variant']}",
                    {"score": float(best["score"]), "brier": float(best["brier"]), "loss": float(best["loss"])},
                    float(best["delta_brier_vs_A0_same_group"]),
                )
            )
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
