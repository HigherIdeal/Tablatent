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


PREFIX_RATES = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}

VARIANTS = (
    "A0_REGIME",
    "A1_FROZEN_SUCCESS",
    "A2_FROZEN_MULTI",
    "A3_FROZEN_MULTI_SHIFT",
)


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one fold is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate folds: {values}")
    return sorted(values)


def _rounded_count(n: np.ndarray, rate: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    raw = n * rate
    rounded = np.rint(raw)
    valid = np.isfinite(raw) & (np.abs(raw - rounded) <= float(tolerance))
    return rounded, valid


def add_frozen_anchor_features(
    frame: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    n_col: str,
    count_tolerance: float,
) -> pd.DataFrame:
    """Build strict independent-row features using only a frozen earlier-season anchor.

    For every season s, all rows in s use the same per-pitcher snapshot selected
    from seasons < s. Rows inside s never update the anchor for other rows in s.
    Therefore a validation/test row can be featurized independently from every
    other validation/test row.

    The frozen snapshot is the latest observed pre-pitch cumulative state from
    prior seasons. Because that snapshot is itself pre-pitch, the delta interval
    can include the final pitch represented after the snapshot; this is deliberate
    and inference-safe. No validation/test label or another validation/test row is
    read during feature construction.
    """
    required = {season_col, pitcher_col, n_col, *PREFIX_RATES.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing frozen-anchor columns: {missing}")

    frame["_anchor_orig_order"] = np.arange(len(frame), dtype=np.int64)
    frame[n_col] = pd.to_numeric(frame[n_col], errors="raise").astype(np.int64)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)

    engineered = [
        "eng_anchor_available",
        "eng_anchor_gap_n",
    ]
    for short in PREFIX_RATES:
        engineered.extend(
            [
                f"eng_anchor_{short}_rate",
                f"eng_since_anchor_{short}_rate",
                f"eng_since_anchor_{short}_minus_long",
                f"eng_since_anchor_{short}_minus_anchor",
            ]
        )
    for column in engineered:
        frame[column] = np.nan

    diagnostics: list[dict] = []
    anchor: pd.DataFrame | None = None

    for season in sorted(frame[season_col].unique().tolist()):
        mask = frame[season_col].eq(season)
        idx = frame.index[mask]
        part = frame.loc[idx]

        if anchor is None or anchor.empty:
            frame.loc[idx, "eng_anchor_available"] = np.float32(0.0)
            diagnostics.append(
                {
                    "season": int(season),
                    "rows": int(len(part)),
                    "anchor_available_rows": 0,
                    "anchor_available_share": 0.0,
                    "positive_gap_rows": 0,
                    "positive_gap_share": 0.0,
                    "median_gap_n": np.nan,
                    "p95_gap_n": np.nan,
                    **{f"{short}_valid_share": 0.0 for short in PREFIX_RATES},
                }
            )
        else:
            pitcher = part[pitcher_col]
            anchor_n = pitcher.map(anchor[n_col]).to_numpy(np.float64)
            current_n = pd.to_numeric(part[n_col], errors="raise").to_numpy(np.float64)
            available = np.isfinite(anchor_n)
            gap = current_n - anchor_n
            positive_gap = available & np.isfinite(gap) & (gap > 0.0)

            frame.loc[idx, "eng_anchor_available"] = available.astype(np.float32)
            gap_values = np.full(len(part), np.nan, dtype=np.float32)
            gap_values[positive_gap] = gap[positive_gap].astype(np.float32)
            frame.loc[idx, "eng_anchor_gap_n"] = gap_values

            diag = {
                "season": int(season),
                "rows": int(len(part)),
                "anchor_available_rows": int(available.sum()),
                "anchor_available_share": float(available.mean()),
                "positive_gap_rows": int(positive_gap.sum()),
                "positive_gap_share": float(positive_gap.mean()),
                "median_gap_n": float(np.nanmedian(gap_values)) if positive_gap.any() else np.nan,
                "p95_gap_n": float(np.nanpercentile(gap_values, 95)) if positive_gap.any() else np.nan,
            }

            for short, column in PREFIX_RATES.items():
                current_rate = pd.to_numeric(part[column], errors="coerce").to_numpy(np.float64)
                anchor_rate = pitcher.map(anchor[column]).to_numpy(np.float64)

                current_count, current_valid = _rounded_count(current_n, current_rate, count_tolerance)
                anchor_count, anchor_valid = _rounded_count(anchor_n, anchor_rate, count_tolerance)
                delta_count = current_count - anchor_count
                rate_valid = (
                    positive_gap
                    & current_valid
                    & anchor_valid
                    & np.isfinite(delta_count)
                    & (delta_count >= 0.0)
                    & (delta_count <= gap)
                )

                anchor_out = np.full(len(part), np.nan, dtype=np.float32)
                anchor_out[available & np.isfinite(anchor_rate)] = anchor_rate[
                    available & np.isfinite(anchor_rate)
                ].astype(np.float32)

                since = np.full(len(part), np.nan, dtype=np.float32)
                since[rate_valid] = (delta_count[rate_valid] / gap[rate_valid]).astype(np.float32)

                minus_long = np.full(len(part), np.nan, dtype=np.float32)
                minus_anchor = np.full(len(part), np.nan, dtype=np.float32)
                finite_since = np.isfinite(since)
                finite_long = np.isfinite(current_rate)
                finite_anchor = np.isfinite(anchor_rate)
                ok_long = finite_since & finite_long
                ok_anchor = finite_since & finite_anchor
                minus_long[ok_long] = (since[ok_long] - current_rate[ok_long]).astype(np.float32)
                minus_anchor[ok_anchor] = (since[ok_anchor] - anchor_rate[ok_anchor]).astype(np.float32)

                frame.loc[idx, f"eng_anchor_{short}_rate"] = anchor_out
                frame.loc[idx, f"eng_since_anchor_{short}_rate"] = since
                frame.loc[idx, f"eng_since_anchor_{short}_minus_long"] = minus_long
                frame.loc[idx, f"eng_since_anchor_{short}_minus_anchor"] = minus_anchor
                diag[f"{short}_valid_share"] = float(rate_valid.mean())

            diagnostics.append(diag)

        # Update only after the complete season has been featurized. This is the
        # key independent-row rule: no row in this season can affect another row
        # in the same season.
        observed = frame.loc[idx, [pitcher_col, n_col, *PREFIX_RATES.values(), "_anchor_orig_order"]].copy()
        observed = observed.sort_values(
            [pitcher_col, n_col, "_anchor_orig_order"], kind="stable"
        )
        latest_this_season = observed.groupby(pitcher_col, sort=False).tail(1).set_index(pitcher_col)
        if anchor is None or anchor.empty:
            anchor = latest_this_season[[n_col, *PREFIX_RATES.values()]].copy()
        else:
            anchor = pd.concat(
                [anchor, latest_this_season[[n_col, *PREFIX_RATES.values()]]], axis=0
            )
            anchor = anchor.reset_index().sort_values(
                [pitcher_col, n_col], kind="stable"
            ).groupby(pitcher_col, sort=False).tail(1).set_index(pitcher_col)

    frame.drop(columns=["_anchor_orig_order"], inplace=True)
    return pd.DataFrame(diagnostics)


def feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    regime = [*base_features, "eng_recent_f"]

    success = [
        *regime,
        "eng_anchor_available",
        "eng_anchor_gap_n",
        "eng_anchor_success_rate",
        "eng_since_anchor_success_rate",
        "eng_since_anchor_success_minus_long",
    ]

    multi_extra: list[str] = []
    for short in ("reverse", "middle", "ball", "strike"):
        multi_extra.extend(
            [
                f"eng_anchor_{short}_rate",
                f"eng_since_anchor_{short}_rate",
                f"eng_since_anchor_{short}_minus_long",
            ]
        )
    multi = [*success, *multi_extra]

    shift_extra = [
        f"eng_since_anchor_{short}_minus_anchor" for short in PREFIX_RATES
    ]
    multi_shift = [*multi, *shift_extra]

    out = {
        "A0_REGIME": regime,
        "A1_FROZEN_SUCCESS": success,
        "A2_FROZEN_MULTI": multi,
        "A3_FROZEN_MULTI_SHIFT": multi_shift,
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
                    "delta_brier_vs_A0_same_group": float(
                        metric["brier"] - baseline[group]["brier"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict independent-row proxy for pitch-level state: freeze each pitcher's latest "
            "cumulative asof snapshot from earlier seasons, then combine that frozen training-only "
            "anchor with the current row's asof values. No validation row reads another validation row."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--count-tolerance", type=float, default=0.05)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/frozen_season_anchor_probe")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.count_tolerance <= 0.0:
        raise ValueError("--count-tolerance must be positive")
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
    n_col = "asof_pitcher_n"

    frame, invariant_check = recent_core.prepare_frame(config)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.strip().str.upper()

    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    anchor_diag = add_frozen_anchor_features(
        frame,
        season_col=season_col,
        pitcher_col=pitcher_col,
        n_col=n_col,
        count_tolerance=args.count_tolerance,
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
    anchor_diag.to_csv(output_dir / "anchor_diagnostics.csv", index=False)

    tqdm.write(
        f"Frozen season anchor | folds={folds} | rows={len(frame):,} | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | "
        f"iterations={args.iterations} | catboost={catboost.__version__}"
    )
    tqdm.write(
        "STRICT RULE: each season uses only anchors frozen from earlier seasons; "
        "no row in a validation season contributes to another validation row."
    )
    tqdm.write("\n[Anchor diagnostics]")
    for _, row in anchor_diag.iterrows():
        tqdm.write(
            f"season={int(row['season'])} rows={int(row['rows']):,} "
            f"anchor={float(row['anchor_available_share']):.4f} "
            f"gap>0={float(row['positive_gap_share']):.4f} "
            f"median_gap={float(row['median_gap_n']):.1f} "
            f"success_valid={float(row['success_valid_share']):.4f}"
        )

    all_results: list[pd.DataFrame] = []
    progress = tqdm(total=len(folds) * len(VARIANTS), desc="frozen anchor models", unit="model", dynamic_ncols=True)

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
        fold_df.insert(0, "fold", int(val_year))
        all_results.append(fold_df)

    progress.close()
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(output_dir / "fold_results.csv", index=False)

    all_only = results.loc[results["group"].eq("ALL")].copy()
    summary = (
        all_only.groupby("variant", as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_brier=("brier", "mean"),
            mean_delta_brier=("delta_brier_vs_A0_same_group", "mean"),
            worst_delta_brier=("delta_brier_vs_A0_same_group", "max"),
            best_delta_brier=("delta_brier_vs_A0_same_group", "min"),
            wins=("delta_brier_vs_A0_same_group", lambda x: int((x < 0).sum())),
        )
        .sort_values(["mean_delta_brier", "worst_delta_brier"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "experiment": "strict frozen earlier-season anchor + current-row asof state",
            "folds": folds,
            "regime_start_year": int(args.regime_start_year),
            "count_tolerance": float(args.count_tolerance),
            "iterations": int(args.iterations),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "feature_sets": variants,
            "independent_row_inference": True,
            "anchor_policy": (
                "for season s, use latest pre-pitch cumulative snapshot from seasons < s; "
                "update anchors only after all rows of s are featurized"
            ),
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Fold results | ALL]")
    for val_year in folds:
        tqdm.write(f"fold={val_year}")
        subset = results.loc[
            results["fold"].eq(val_year) & results["group"].eq("ALL")
        ].sort_values("brier")
        for _, row in subset.iterrows():
            metric = {
                "score": float(row["score"]),
                "brier": float(row["brier"]),
                "loss": float(row["loss"]),
            }
            tqdm.write(
                regime_core.metric_line(
                    str(row["variant"]),
                    metric,
                    float(row["delta_brier_vs_A0_same_group"]),
                )
            )

    tqdm.write("\n[Cross-fold summary | ALL]")
    for _, row in summary.iterrows():
        tqdm.write(
            f"{str(row['variant']):<24s} mean_dB={float(row['mean_delta_brier']):+.8f} "
            f"worst_dB={float(row['worst_delta_brier']):+.8f} "
            f"best_dB={float(row['best_delta_brier']):+.8f} "
            f"wins={int(row['wins'])}/{int(row['folds'])}"
        )

    tqdm.write("\n[2024 diagnostics]")
    subset_2024 = results.loc[results["fold"].eq(2024)].sort_values(["group", "brier"])
    for group in ("ALL", "R", "F"):
        part = subset_2024.loc[subset_2024["group"].eq(group)]
        if part.empty:
            continue
        best = part.iloc[0]
        tqdm.write(
            f"best/{group:<3s} {str(best['variant']):<24s} "
            f"score={float(best['score']):+9.2f} brier={float(best['brier']):.8f} "
            f"loss={float(best['loss']):.8f} dB={float(best['delta_brier_vs_A0_same_group']):+.8f}"
        )

    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
