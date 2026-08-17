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


VARIANTS = (
    "A0_REGIME",
    "A1_PITCHER_SHEET",
    "A2_PITCHER_RICH_SHEET",
    "A3_PLAYER_SHEET",
    "A4_PLAYER_MATCHUP_SHEET",
)

PITCHER_BASE = [
    "eng_prev_pitcher_n",
    "eng_prev_pitcher_success",
    "eng_prev_pitcher_success_minus_asof",
    "eng_prev_pitcher_same_gt_n",
    "eng_prev_pitcher_same_gt_success",
]

PITCHER_RICH = [
    "eng_prev_pitcher_r_n",
    "eng_prev_pitcher_r_success",
    "eng_prev_pitcher_f_n",
    "eng_prev_pitcher_f_success",
    "eng_prev_pitcher_late_n",
    "eng_prev_pitcher_late_success",
    "eng_prev_pitcher_late_minus_asof",
]

BATTER = [
    "eng_prev_batter_n",
    "eng_prev_batter_success",
    "eng_prev_batter_success_minus_asof",
    "eng_prev_batter_same_gt_n",
    "eng_prev_batter_same_gt_success",
]

MATCHUP = [
    "eng_prev_matchup_n",
    "eng_prev_matchup_success",
]

ALL_SHEET_FEATURES = PITCHER_BASE + PITCHER_RICH + BATTER + MATCHUP


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one fold is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate folds: {values}")
    return sorted(values)


def _mean_table(
    source: pd.DataFrame,
    keys: list[str],
    target_col: str,
    *,
    prefix: str,
) -> pd.DataFrame:
    table = (
        source.groupby(keys, observed=True, sort=False)[target_col]
        .agg(["size", "mean"])
        .rename(columns={"size": f"{prefix}_n", "mean": f"{prefix}_success"})
        .reset_index()
    )
    table[f"{prefix}_n"] = table[f"{prefix}_n"].astype(np.float32)
    table[f"{prefix}_success"] = table[f"{prefix}_success"].astype(np.float32)
    return table


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


def add_previous_season_sheet_features(
    frame: pd.DataFrame,
    *,
    target_col: str,
    season_col: str,
    pitcher_col: str,
    batter_col: str,
    late_month_min: int,
) -> pd.DataFrame:
    """Create strict previous-season target summaries.

    For every season s, rows in s receive lookup statistics built only from season s-1.
    No target from season s is used to featurize any row in season s. This exactly mirrors
    2025 deployment where the 2024 labeled training season is frozen as a compact lookup.
    """
    required = {
        target_col,
        season_col,
        pitcher_col,
        batter_col,
        "game_type",
        "game_month",
        "asof_pitcher_success_rate",
        "asof_batter_success_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing previous-season sheet columns: {missing}")

    for column in ALL_SHEET_FEATURES:
        frame[column] = np.nan

    diagnostics: list[dict] = []
    seasons = sorted(pd.to_numeric(frame[season_col], errors="raise").astype(int).unique().tolist())

    for season in seasons:
        dest_mask = frame[season_col].eq(season)
        dest_idx = frame.index[dest_mask]
        dest = frame.loc[dest_idx]
        source = frame.loc[frame[season_col].eq(season - 1)]

        if source.empty:
            diagnostics.append(
                {
                    "season": int(season),
                    "rows": int(len(dest)),
                    "source_season": int(season - 1),
                    "source_rows": 0,
                    "pitcher_coverage": 0.0,
                    "batter_coverage": 0.0,
                    "matchup_coverage": 0.0,
                }
            )
            continue

        # Pitcher overall.
        p_all = _mean_table(source, [pitcher_col], target_col, prefix="eng_prev_pitcher")
        vals = _left_join_values(
            dest,
            p_all,
            [pitcher_col],
            ["eng_prev_pitcher_n", "eng_prev_pitcher_success"],
        )
        frame.loc[dest_idx, vals.columns] = vals.to_numpy(np.float32)

        # Pitcher game-type-specific table. Store R/F separately plus current-row same-GT view.
        p_gt = _mean_table(source, [pitcher_col, "game_type"], target_col, prefix="_tmp_pitcher_gt")
        for gt, suffix in (("R", "r"), ("F", "f")):
            sub = p_gt.loc[p_gt["game_type"].eq(gt), [pitcher_col, "_tmp_pitcher_gt_n", "_tmp_pitcher_gt_success"]].copy()
            sub = sub.rename(
                columns={
                    "_tmp_pitcher_gt_n": f"eng_prev_pitcher_{suffix}_n",
                    "_tmp_pitcher_gt_success": f"eng_prev_pitcher_{suffix}_success",
                }
            )
            value_cols = [f"eng_prev_pitcher_{suffix}_n", f"eng_prev_pitcher_{suffix}_success"]
            vals = _left_join_values(dest, sub, [pitcher_col], value_cols)
            frame.loc[dest_idx, value_cols] = vals.to_numpy(np.float32)

        same_gt = p_gt.rename(
            columns={
                "_tmp_pitcher_gt_n": "eng_prev_pitcher_same_gt_n",
                "_tmp_pitcher_gt_success": "eng_prev_pitcher_same_gt_success",
            }
        )
        vals = _left_join_values(
            dest,
            same_gt,
            [pitcher_col, "game_type"],
            ["eng_prev_pitcher_same_gt_n", "eng_prev_pitcher_same_gt_success"],
        )
        frame.loc[dest_idx, vals.columns] = vals.to_numpy(np.float32)

        # Late-season pitcher form from only the previous season.
        late_source = source.loc[pd.to_numeric(source["game_month"], errors="coerce").ge(int(late_month_min))]
        if not late_source.empty:
            p_late = _mean_table(late_source, [pitcher_col], target_col, prefix="eng_prev_pitcher_late")
            vals = _left_join_values(
                dest,
                p_late,
                [pitcher_col],
                ["eng_prev_pitcher_late_n", "eng_prev_pitcher_late_success"],
            )
            frame.loc[dest_idx, vals.columns] = vals.to_numpy(np.float32)

        # Batter previous-season memory.
        b_all = _mean_table(source, [batter_col], target_col, prefix="eng_prev_batter")
        vals = _left_join_values(
            dest,
            b_all,
            [batter_col],
            ["eng_prev_batter_n", "eng_prev_batter_success"],
        )
        frame.loc[dest_idx, vals.columns] = vals.to_numpy(np.float32)

        b_gt = _mean_table(source, [batter_col, "game_type"], target_col, prefix="_tmp_batter_gt")
        b_same_gt = b_gt.rename(
            columns={
                "_tmp_batter_gt_n": "eng_prev_batter_same_gt_n",
                "_tmp_batter_gt_success": "eng_prev_batter_same_gt_success",
            }
        )
        vals = _left_join_values(
            dest,
            b_same_gt,
            [batter_col, "game_type"],
            ["eng_prev_batter_same_gt_n", "eng_prev_batter_same_gt_success"],
        )
        frame.loc[dest_idx, vals.columns] = vals.to_numpy(np.float32)

        # Exact pitcher-batter pair, previous season only. Sparse by design; n tells reliability.
        matchup = _mean_table(source, [pitcher_col, batter_col], target_col, prefix="eng_prev_matchup")
        vals = _left_join_values(
            dest,
            matchup,
            [pitcher_col, batter_col],
            ["eng_prev_matchup_n", "eng_prev_matchup_success"],
        )
        frame.loc[dest_idx, vals.columns] = vals.to_numpy(np.float32)

        # Differences to the current row's supplied long-run asof state.
        p_asof = pd.to_numeric(dest["asof_pitcher_success_rate"], errors="coerce").to_numpy(np.float32)
        b_asof = pd.to_numeric(dest["asof_batter_success_rate"], errors="coerce").to_numpy(np.float32)
        p_prev = pd.to_numeric(frame.loc[dest_idx, "eng_prev_pitcher_success"], errors="coerce").to_numpy(np.float32)
        p_late_rate = pd.to_numeric(frame.loc[dest_idx, "eng_prev_pitcher_late_success"], errors="coerce").to_numpy(np.float32)
        b_prev = pd.to_numeric(frame.loc[dest_idx, "eng_prev_batter_success"], errors="coerce").to_numpy(np.float32)
        frame.loc[dest_idx, "eng_prev_pitcher_success_minus_asof"] = (p_prev - p_asof).astype(np.float32)
        frame.loc[dest_idx, "eng_prev_pitcher_late_minus_asof"] = (p_late_rate - p_asof).astype(np.float32)
        frame.loc[dest_idx, "eng_prev_batter_success_minus_asof"] = (b_prev - b_asof).astype(np.float32)

        diagnostics.append(
            {
                "season": int(season),
                "rows": int(len(dest)),
                "source_season": int(season - 1),
                "source_rows": int(len(source)),
                "pitcher_coverage": float(frame.loc[dest_idx, "eng_prev_pitcher_success"].notna().mean()),
                "batter_coverage": float(frame.loc[dest_idx, "eng_prev_batter_success"].notna().mean()),
                "matchup_coverage": float(frame.loc[dest_idx, "eng_prev_matchup_success"].notna().mean()),
            }
        )

    for column in ALL_SHEET_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return pd.DataFrame(diagnostics)


def feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    regime = [*base_features, "eng_recent_f"]
    out = {
        "A0_REGIME": regime,
        "A1_PITCHER_SHEET": [*regime, *PITCHER_BASE],
        "A2_PITCHER_RICH_SHEET": [*regime, *PITCHER_BASE, *PITCHER_RICH],
        "A3_PLAYER_SHEET": [*regime, *PITCHER_BASE, *PITCHER_RICH, *BATTER],
        "A4_PLAYER_MATCHUP_SHEET": [*regime, *PITCHER_BASE, *PITCHER_RICH, *BATTER, *MATCHUP],
    }
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
                    "delta_brier_vs_A0_same_group": float(metric["brier"] - baseline[group]["brier"]),
                }
            )
    return pd.DataFrame(rows)


def _safe_int_ids(series: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(np.int64)
    if np.any(values.astype(np.float64) != numeric.to_numpy(np.float64)):
        raise ValueError(f"{name} cannot be represented losslessly as int64")
    return values


def save_compact_deploy_sheet(
    frame: pd.DataFrame,
    *,
    source_year: int,
    target_col: str,
    season_col: str,
    pitcher_col: str,
    batter_col: str,
    late_month_min: int,
    path: Path,
) -> dict[str, int | float | list[str]]:
    """Save only unique numeric lookup rows, never row-expanded CSV features."""
    source = frame.loc[frame[season_col].eq(source_year)].copy()
    if source.empty:
        raise ValueError(f"No source rows for deploy sheet year {source_year}")

    p_all = _mean_table(source, [pitcher_col], target_col, prefix="p")
    p_r = _mean_table(source.loc[source["game_type"].eq("R")], [pitcher_col], target_col, prefix="r")
    p_f = _mean_table(source.loc[source["game_type"].eq("F")], [pitcher_col], target_col, prefix="f")
    p_late = _mean_table(
        source.loc[pd.to_numeric(source["game_month"], errors="coerce").ge(int(late_month_min))],
        [pitcher_col],
        target_col,
        prefix="late",
    )
    p = p_all.merge(p_r, on=pitcher_col, how="left").merge(p_f, on=pitcher_col, how="left").merge(p_late, on=pitcher_col, how="left")

    b_all = _mean_table(source, [batter_col], target_col, prefix="b")
    b_r = _mean_table(source.loc[source["game_type"].eq("R")], [batter_col], target_col, prefix="r")
    b_f = _mean_table(source.loc[source["game_type"].eq("F")], [batter_col], target_col, prefix="f")
    b = b_all.merge(b_r, on=batter_col, how="left").merge(b_f, on=batter_col, how="left")

    m = _mean_table(source, [pitcher_col, batter_col], target_col, prefix="m")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        source_year=np.asarray([source_year], dtype=np.int16),
        pitcher_id=_safe_int_ids(p[pitcher_col], pitcher_col),
        pitcher_n=p["p_n"].fillna(0).to_numpy(np.int32),
        pitcher_success=p["p_success"].to_numpy(np.float32),
        pitcher_r_n=p["r_n"].fillna(0).to_numpy(np.int32),
        pitcher_r_success=p["r_success"].to_numpy(np.float32),
        pitcher_f_n=p["f_n"].fillna(0).to_numpy(np.int32),
        pitcher_f_success=p["f_success"].to_numpy(np.float32),
        pitcher_late_n=p["late_n"].fillna(0).to_numpy(np.int32),
        pitcher_late_success=p["late_success"].to_numpy(np.float32),
        batter_id=_safe_int_ids(b[batter_col], batter_col),
        batter_n=b["b_n"].fillna(0).to_numpy(np.int32),
        batter_success=b["b_success"].to_numpy(np.float32),
        batter_r_n=b["r_n"].fillna(0).to_numpy(np.int32),
        batter_r_success=b["r_success"].to_numpy(np.float32),
        batter_f_n=b["f_n"].fillna(0).to_numpy(np.int32),
        batter_f_success=b["f_success"].to_numpy(np.float32),
        matchup_pitcher_id=_safe_int_ids(m[pitcher_col], pitcher_col),
        matchup_batter_id=_safe_int_ids(m[batter_col], batter_col),
        matchup_n=m["m_n"].fillna(0).to_numpy(np.int32),
        matchup_success=m["m_success"].to_numpy(np.float32),
    )
    size = int(path.stat().st_size)
    return {
        "source_year": int(source_year),
        "pitchers": int(len(p)),
        "batters": int(len(b)),
        "matchups": int(len(m)),
        "bytes": size,
        "kib": float(size / 1024.0),
        "format": ["int64 ids", "int32 counts", "float32 rates", "npz compressed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict forward-fold previous-season target-memory probe. Each season receives only "
            "statistics aggregated from the immediately preceding labeled season. A compact 2024 "
            "numeric NPZ lookup is also written for eventual 2025 deployment; no row-expanded CSV."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--late-month-min", type=int, default=8)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/previous_season_cheatsheet_probe")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not (1 <= args.late_month_min <= 12):
        raise ValueError("--late-month-min must be in [1,12]")
    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05,1.0]")

    folds = parse_ints(args.folds)
    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)

    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    pitcher_col = "pitcher_id"
    batter_col = "batter_id"

    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()
    unexpected = sorted(set(frame["game_type"].dropna().unique()) - {"R", "F"})
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")

    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    sheet_diag = add_previous_season_sheet_features(
        frame,
        target_col=target_col,
        season_col=season_col,
        pitcher_col=pitcher_col,
        batter_col=batter_col,
        late_month_min=args.late_month_min,
    )
    regime_core.add_regime_features(
        frame,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )

    base_features = recent_core.feature_set("recent_raw_game_type")
    variants = feature_sets(base_features)
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
    sheet_diag.to_csv(output_dir / "coverage_metrics.csv", index=False)

    deploy_path = output_dir / "deploy_2025_cheatsheet_from_2024.npz"
    deploy_info = save_compact_deploy_sheet(
        frame,
        source_year=2024,
        target_col=target_col,
        season_col=season_col,
        pitcher_col=pitcher_col,
        batter_col=batter_col,
        late_month_min=args.late_month_min,
        path=deploy_path,
    )

    tqdm.write(
        f"Previous-season cheatsheet | folds={folds} | rows={len(frame):,} | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | iterations={args.iterations} | "
        f"catboost={catboost.__version__}"
    )
    tqdm.write(
        "STRICT RULE: season s features use target summaries from season s-1 only; "
        "validation rows never contribute to their own or peer-row lookup features."
    )
    tqdm.write(
        f"compact deploy artifact={deploy_path.name} size={deploy_info['kib']:.1f} KiB "
        f"pitchers={deploy_info['pitchers']} batters={deploy_info['batters']} matchups={deploy_info['matchups']}"
    )
    tqdm.write("\n[Coverage diagnostics]")
    for _, row in sheet_diag.iterrows():
        tqdm.write(
            f"season={int(row['season'])} src={int(row['source_season'])} rows={int(row['rows']):,} "
            f"pitcher={float(row['pitcher_coverage']):.4f} batter={float(row['batter_coverage']):.4f} "
            f"matchup={float(row['matchup_coverage']):.4f}"
        )

    all_results: list[pd.DataFrame] = []
    predictions_latest: dict[str, np.ndarray] = {}
    progress = tqdm(total=len(folds) * len(VARIANTS), desc="cheatsheet models", unit="model", dynamic_ncols=True)

    for val_year in folds:
        train = frame.loc[frame[season_col] < val_year].copy()
        valid = frame.loc[frame[season_col].eq(val_year)].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train/valid")
        y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt_valid = valid["game_type"].astype(str).to_numpy()

        fold_pred: dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            seed_everything(seed)
            pred = regime_core.fit_predict(
                train=train,
                valid=valid,
                target_col=target_col,
                features=variants[variant],
                extra_categorical=set(),
                params=params,
            )
            fold_pred[variant] = pred
            progress.update(1)

        fold_df = evaluate(y_valid, gt_valid, fold_pred)
        fold_df.insert(0, "validation_year", int(val_year))
        all_results.append(fold_df)
        if val_year == max(folds):
            predictions_latest = fold_pred

        del train, valid, y_valid, gt_valid, fold_pred
        gc.collect()

    progress.close()

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(output_dir / "fold_metrics.csv", index=False)

    all_only = results.loc[results["group"].eq("ALL")].copy()
    summary = (
        all_only.groupby("variant", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_brier=("brier", "mean"),
            mean_delta=("delta_brier_vs_A0_same_group", "mean"),
            worst_delta=("delta_brier_vs_A0_same_group", "max"),
            best_delta=("delta_brier_vs_A0_same_group", "min"),
            wins=("delta_brier_vs_A0_same_group", lambda x: int((x < 0).sum())),
        )
        .sort_values(["mean_delta", "worst_delta"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    latest_year = max(folds)
    valid_latest = frame.loc[frame[season_col].eq(latest_year)]
    if predictions_latest:
        np.savez_compressed(
            output_dir / f"predictions_{latest_year}.npz",
            target=pd.to_numeric(valid_latest[target_col], errors="raise").to_numpy(np.float32),
            game_type_f=valid_latest["game_type"].eq("F").to_numpy(np.uint8),
            **{name: pred.astype(np.float32) for name, pred in predictions_latest.items()},
        )

    save_json(
        {
            "experiment": "strict previous-season numeric cheatsheet",
            "folds": folds,
            "late_month_min": int(args.late_month_min),
            "variants": {k: v for k, v in variants.items()},
            "deploy_artifact": deploy_info,
            "deploy_artifact_path": str(deploy_path),
            "artifact_policy": "unique lookup rows only; int ids/counts + float32 rates in npz_compressed; no row-expanded feature CSV",
            "leakage_policy": "season s receives target aggregates from season s-1 only",
            "catboost_params": params,
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Fold results | ALL]")
    for val_year in folds:
        tqdm.write(f"fold={val_year}")
        subset = results.loc[(results["validation_year"].eq(val_year)) & (results["group"].eq("ALL"))].sort_values("brier")
        for _, row in subset.iterrows():
            metric = {"score": float(row["score"]), "brier": float(row["brier"]), "loss": float(row["loss"])}
            tqdm.write(regime_core.metric_line(str(row["variant"]), metric, float(row["delta_brier_vs_A0_same_group"])))

    tqdm.write("\n[Cross-fold summary | ALL]")
    for _, row in summary.iterrows():
        tqdm.write(
            f"{str(row['variant']):<27s} mean_dB={float(row['mean_delta']):+.8f} "
            f"worst_dB={float(row['worst_delta']):+.8f} best_dB={float(row['best_delta']):+.8f} "
            f"wins={int(row['wins'])}/{int(row['folds'])}"
        )

    tqdm.write(f"\n[{latest_year} diagnostics | R/F]")
    latest = results.loc[results["validation_year"].eq(latest_year)]
    for group in ("R", "F"):
        subset = latest.loc[latest["group"].eq(group)].sort_values("brier")
        best = subset.iloc[0]
        tqdm.write(
            f"best/{group} {str(best['variant']):<27s} score={float(best['score']):+9.2f} "
            f"brier={float(best['brier']):.8f} loss={float(best['loss']):.8f} "
            f"dB={float(best['delta_brier_vs_A0_same_group']):+.8f}"
        )
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
