from __future__ import annotations

import argparse
import json
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
    "T0_REGIME",
    "T1_SUCCESS_CURVATURE",
    "T2_SUCCESS_TREND_CAT",
    "T3_MIDDLE_TRAJECTORY",
    "T4_JOINT_TRAJECTORY",
    "T5_PERSISTENCE",
)

SUCCESS_SHAPE = [
    "eng_ps_curvature_135",
    "eng_ps_abs_d13",
    "eng_ps_abs_d35",
    "eng_ps_d13_x_d35",
]

SUCCESS_CAT = ["eng_ps_trend_state"]

MIDDLE_SHAPE = [
    "eng_pm_curvature_135",
    "eng_pm_abs_d13",
    "eng_pm_abs_d35",
    "eng_pm_d13_x_d35",
]

MIDDLE_CAT = ["eng_pm_trend_state"]

JOINT = [
    "eng_ps_pm_trend_state",
    "eng_ps_pm_d13_product",
    "eng_ps_pm_curvature_product",
    "eng_ps_pm_d13_minus",
    "eng_ps_pm_curvature_minus",
    "eng_ps_pm_direction_agreement",
]

PERSISTENCE = [
    "eng_ps_persistence_signed",
    "eng_ps_reversal_signed",
    "eng_pm_persistence_signed",
    "eng_pm_reversal_signed",
]

TRAJECTORY_CATEGORICAL = {
    "eng_ps_trend_state",
    "eng_pm_trend_state",
    "eng_ps_pm_trend_state",
}


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one fold is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate folds: {values}")
    return sorted(values)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(np.float32)


def trend_state(d13: pd.Series, d35: pd.Series, eps: float) -> pd.Series:
    a = pd.to_numeric(d13, errors="coerce").to_numpy(np.float32)
    b = pd.to_numeric(d35, errors="coerce").to_numpy(np.float32)
    finite = np.isfinite(a) & np.isfinite(b)

    out = np.full(len(a), "MISSING", dtype=object)
    flat_a = np.abs(a) <= eps
    flat_b = np.abs(b) <= eps

    out[finite & flat_a & flat_b] = "FLAT"
    out[finite & (a > eps) & (b > eps)] = "UP"
    out[finite & (a < -eps) & (b < -eps)] = "DOWN"
    out[finite & (a > eps) & (b < -eps)] = "REV_UP"
    out[finite & (a < -eps) & (b > eps)] = "REV_DOWN"

    unresolved = finite & (out == "MISSING")
    out[unresolved] = "MIXED"
    return pd.Series(out, index=d13.index, dtype="string")


def signed_persistence(d13: pd.Series, d35: pd.Series) -> tuple[pd.Series, pd.Series]:
    a = pd.to_numeric(d13, errors="coerce").to_numpy(np.float32)
    b = pd.to_numeric(d35, errors="coerce").to_numpy(np.float32)
    finite = np.isfinite(a) & np.isfinite(b)
    magnitude = np.minimum(np.abs(a), np.abs(b)).astype(np.float32)

    persistence = np.full(len(a), np.nan, dtype=np.float32)
    reversal = np.full(len(a), np.nan, dtype=np.float32)
    persistence[finite] = 0.0
    reversal[finite] = 0.0

    same = finite & ((a * b) > 0)
    flip = finite & ((a * b) < 0)
    persistence[same] = np.sign(a[same]).astype(np.float32) * magnitude[same]
    reversal[flip] = np.sign(a[flip]).astype(np.float32) * magnitude[flip]

    return (
        pd.Series(persistence, index=d13.index, dtype=np.float32),
        pd.Series(reversal, index=d13.index, dtype=np.float32),
    )


def direction_code(d13: pd.Series, eps: float) -> np.ndarray:
    x = pd.to_numeric(d13, errors="coerce").to_numpy(np.float32)
    out = np.zeros(len(x), dtype=np.float32)
    out[x > eps] = 1.0
    out[x < -eps] = -1.0
    out[~np.isfinite(x)] = np.nan
    return out


def add_pitcher_trajectory_features(frame: pd.DataFrame, *, trend_eps: float) -> None:
    required = {
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing trajectory source columns: {missing}")

    ps1 = numeric(frame, "asof_pitcher_prev1_game_success_rate")
    ps3 = numeric(frame, "asof_pitcher_prev3_game_success_rate")
    ps5 = numeric(frame, "asof_pitcher_prev5_game_success_rate")
    pm1 = numeric(frame, "asof_pitcher_prev1_game_middle_rate")
    pm3 = numeric(frame, "asof_pitcher_prev3_game_middle_rate")
    pm5 = numeric(frame, "asof_pitcher_prev5_game_middle_rate")

    ps_d13 = ps1 - ps3
    ps_d35 = ps3 - ps5
    pm_d13 = pm1 - pm3
    pm_d35 = pm3 - pm5

    frame["eng_ps_curvature_135"] = (ps_d13 - ps_d35).astype(np.float32)
    frame["eng_ps_abs_d13"] = ps_d13.abs().astype(np.float32)
    frame["eng_ps_abs_d35"] = ps_d35.abs().astype(np.float32)
    frame["eng_ps_d13_x_d35"] = (ps_d13 * ps_d35).astype(np.float32)
    frame["eng_ps_trend_state"] = trend_state(ps_d13, ps_d35, trend_eps)

    frame["eng_pm_curvature_135"] = (pm_d13 - pm_d35).astype(np.float32)
    frame["eng_pm_abs_d13"] = pm_d13.abs().astype(np.float32)
    frame["eng_pm_abs_d35"] = pm_d35.abs().astype(np.float32)
    frame["eng_pm_d13_x_d35"] = (pm_d13 * pm_d35).astype(np.float32)
    frame["eng_pm_trend_state"] = trend_state(pm_d13, pm_d35, trend_eps)

    frame["eng_ps_pm_trend_state"] = (
        frame["eng_ps_trend_state"].astype("string").fillna("MISSING")
        + "__"
        + frame["eng_pm_trend_state"].astype("string").fillna("MISSING")
    )
    frame["eng_ps_pm_d13_product"] = (ps_d13 * pm_d13).astype(np.float32)
    frame["eng_ps_pm_curvature_product"] = (
        frame["eng_ps_curvature_135"] * frame["eng_pm_curvature_135"]
    ).astype(np.float32)
    frame["eng_ps_pm_d13_minus"] = (ps_d13 - pm_d13).astype(np.float32)
    frame["eng_ps_pm_curvature_minus"] = (
        frame["eng_ps_curvature_135"] - frame["eng_pm_curvature_135"]
    ).astype(np.float32)

    ps_dir = direction_code(ps_d13, trend_eps)
    pm_dir = direction_code(pm_d13, trend_eps)
    agreement = np.full(len(frame), np.nan, dtype=np.float32)
    finite = np.isfinite(ps_dir) & np.isfinite(pm_dir)
    agreement[finite] = ps_dir[finite] * pm_dir[finite]
    frame["eng_ps_pm_direction_agreement"] = agreement

    ps_persist, ps_reverse = signed_persistence(ps_d13, ps_d35)
    pm_persist, pm_reverse = signed_persistence(pm_d13, pm_d35)
    frame["eng_ps_persistence_signed"] = ps_persist
    frame["eng_ps_reversal_signed"] = ps_reverse
    frame["eng_pm_persistence_signed"] = pm_persist
    frame["eng_pm_reversal_signed"] = pm_reverse


def feature_sets(base_features: list[str]) -> dict[str, tuple[list[str], set[str]]]:
    regime = [*base_features, "eng_recent_f"]
    out: dict[str, tuple[list[str], set[str]]] = {
        "T0_REGIME": (regime, set()),
        "T1_SUCCESS_CURVATURE": (
            [*regime, *SUCCESS_SHAPE],
            set(),
        ),
        "T2_SUCCESS_TREND_CAT": (
            [*regime, *SUCCESS_SHAPE, *SUCCESS_CAT],
            set(SUCCESS_CAT),
        ),
        "T3_MIDDLE_TRAJECTORY": (
            [*regime, *SUCCESS_SHAPE, *SUCCESS_CAT, *MIDDLE_SHAPE, *MIDDLE_CAT],
            set(SUCCESS_CAT + MIDDLE_CAT),
        ),
        "T4_JOINT_TRAJECTORY": (
            [
                *regime,
                *SUCCESS_SHAPE,
                *SUCCESS_CAT,
                *MIDDLE_SHAPE,
                *MIDDLE_CAT,
                *JOINT,
            ],
            set(SUCCESS_CAT + MIDDLE_CAT + ["eng_ps_pm_trend_state"]),
        ),
        "T5_PERSISTENCE": (
            [
                *regime,
                *SUCCESS_SHAPE,
                *SUCCESS_CAT,
                *MIDDLE_SHAPE,
                *MIDDLE_CAT,
                *JOINT,
                *PERSISTENCE,
            ],
            set(SUCCESS_CAT + MIDDLE_CAT + ["eng_ps_pm_trend_state"]),
        ),
    }
    for name, (features, extra_cat) in out.items():
        if len(features) != len(set(features)):
            dup = sorted({f for f in features if features.count(f) > 1})
            raise RuntimeError(f"duplicate features in {name}: {dup}")
        unknown_cat = extra_cat - set(features)
        if unknown_cat:
            raise RuntimeError(f"categoricals absent from {name}: {sorted(unknown_cat)}")
    return out


def quartile_masks(valid: pd.DataFrame) -> dict[str, np.ndarray]:
    values = pd.to_numeric(valid["asof_pitcher_n"], errors="coerce")
    masks: dict[str, np.ndarray] = {}
    finite = values.notna()
    if int(finite.sum()) < 4:
        return masks
    try:
        codes = pd.qcut(values.loc[finite], q=4, labels=False, duplicates="drop")
    except ValueError:
        return masks
    if codes.empty:
        return masks
    unique_codes = sorted(pd.Series(codes).dropna().astype(int).unique().tolist())
    for code in unique_codes:
        mask = np.zeros(len(valid), dtype=bool)
        positions = np.flatnonzero(finite.to_numpy())
        local = pd.Series(codes).to_numpy()
        selected_positions = positions[np.asarray(local == code)]
        mask[selected_positions] = True
        masks[f"NQ{code + 1}"] = mask
    return masks


def evaluate_fold(
    *,
    fold: int,
    valid: pd.DataFrame,
    target_col: str,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    gt = valid["game_type"].astype("string").str.upper().to_numpy()
    masks: dict[str, np.ndarray] = {
        "ALL": np.ones(len(valid), dtype=bool),
        "R": gt == "R",
        "F": gt == "F",
        **quartile_masks(valid),
    }

    baseline: dict[str, dict[str, float]] = {}
    for group, mask in masks.items():
        if not mask.any():
            continue
        baseline[group] = regime_core.binary_metrics(y[mask], predictions["T0_REGIME"][mask])

    rows: list[dict] = []
    for variant, pred in predictions.items():
        for group, mask in masks.items():
            if group not in baseline or not mask.any():
                continue
            metric = regime_core.binary_metrics(y[mask], pred[mask])
            n_values = pd.to_numeric(valid.loc[mask, "asof_pitcher_n"], errors="coerce")
            rows.append(
                {
                    "fold": int(fold),
                    "variant": variant,
                    "group": group,
                    "rows": int(mask.sum()),
                    "n_median": float(n_values.median()) if n_values.notna().any() else np.nan,
                    **metric,
                    "delta_brier_vs_T0_same_group": float(
                        metric["brier"] - baseline[group]["brier"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def state_distribution(frame: pd.DataFrame, season_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for season, group in frame.groupby(season_col, sort=True):
        for column in ["eng_ps_trend_state", "eng_pm_trend_state"]:
            counts = group[column].astype("string").fillna("MISSING").value_counts(dropna=False)
            total = int(counts.sum())
            for state, count in counts.items():
                rows.append(
                    {
                        "season": int(season),
                        "feature": column,
                        "state": str(state),
                        "rows": int(count),
                        "share": float(count / total) if total else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict independent-row pitcher trajectory probe. Builds success/middle curvature, "
            "trend-shape categories, joint trajectory and persistence from supplied prev1/3/5-game "
            "rates only; no validation-row history or player lookup is used."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument(
        "--trend-eps",
        type=float,
        default=0.005,
        help="Absolute delta <= eps is treated as flat for categorical trend states.",
    )
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/pitcher_trajectory_probe")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.trend_eps < 0.0:
        raise ValueError("--trend-eps must be >= 0")
    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05,1.0]")

    folds = parse_ints(args.folds)
    config = load_config(ROOT / args.config)
    seed_everything(int(config["seed"]))
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")

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

    add_pitcher_trajectory_features(frame, trend_eps=float(args.trend_eps))
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

    state_dist = state_distribution(frame, season_col)
    state_dist.to_csv(output_dir / "trajectory_state_distribution.csv", index=False)

    tqdm.write(
        f"Pitcher trajectory probe | folds={folds} | rows={len(frame):,} | "
        f"trend_eps={args.trend_eps:g} | GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | "
        f"iterations={args.iterations} | catboost={catboost.__version__}"
    )
    tqdm.write(
        "STRICT RULE: every trajectory feature is computed from columns already present in the current row; "
        "no target lookup, validation-row rolling state, or peer-row dependency."
    )
    tqdm.write(
        "baseline=T0_REGIME = canonical + existing SUCCESS_STATE + game_type + eng_recent_f. "
        "T1-T5 are cumulative trajectory additions."
    )

    fold_metric_frames: list[pd.DataFrame] = []
    progress = tqdm(total=len(folds) * len(VARIANTS), desc="trajectory models", unit="model")

    for fold in folds:
        train = frame.loc[frame[season_col] < fold].copy()
        valid = frame.loc[frame[season_col] == fold].copy()
        if train.empty or valid.empty:
            raise ValueError(f"empty train/valid split for fold={fold}")

        predictions: dict[str, np.ndarray] = {}
        for variant in VARIANTS:
            features, extra_cat = variants[variant]
            predictions[variant] = regime_core.fit_predict(
                train=train,
                valid=valid,
                target_col=target_col,
                features=features,
                extra_categorical=extra_cat,
                params=params,
            )
            progress.update(1)

        fold_metrics = evaluate_fold(
            fold=fold,
            valid=valid,
            target_col=target_col,
            predictions=predictions,
        )
        fold_metric_frames.append(fold_metrics)

    progress.close()
    metrics = pd.concat(fold_metric_frames, ignore_index=True)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    all_rows = metrics.loc[metrics["group"].eq("ALL")].copy()
    summary = (
        all_rows.groupby("variant", observed=True)["delta_brier_vs_T0_same_group"]
        .agg(mean_dB="mean", worst_dB="max", best_dB="min", wins=lambda x: int((x < 0).sum()))
        .reset_index()
    )
    summary["folds"] = all_rows.groupby("variant", observed=True).size().reindex(summary["variant"]).to_numpy()
    summary = summary.sort_values(["mean_dB", "worst_dB"], ascending=[True, True]).reset_index(drop=True)
    summary.to_csv(output_dir / "crossfold_summary.csv", index=False)

    manifest = {
        "folds": folds,
        "iterations": int(args.iterations),
        "regime_start_year": int(args.regime_start_year),
        "trend_eps": float(args.trend_eps),
        "variants": {
            name: {
                "features": features,
                "extra_categorical": sorted(extra_cat),
                "feature_count": len(features),
            }
            for name, (features, extra_cat) in variants.items()
        },
        "canonical_invariants": invariant_check,
        "strict_independent_row": True,
        "notes": [
            "prev1/3/5 are treated as nested recent-history summaries, not literal equally-spaced point samples",
            "curvature is a trend-contrast feature, not a physical time derivative",
            "experience quartiles are diagnostics only and do not route or train separate models",
        ],
    }
    save_json(manifest, output_dir / "manifest.json")

    print("\n[Fold results | ALL | lower dB is better]")
    for fold in folds:
        sub = metrics.loc[(metrics["fold"] == fold) & metrics["group"].eq("ALL")].copy()
        sub = sub.sort_values("brier")
        print(f"fold={fold}")
        for row in sub.itertuples(index=False):
            print(
                f"  {row.variant:<28s} brier={row.brier:.8f} score={row.score:+9.2f} "
                f"dB={row.delta_brier_vs_T0_same_group:+.8f}"
            )

    print("\n[Cross-fold summary | ALL]")
    for row in summary.itertuples(index=False):
        print(
            f"{row.variant:<28s} mean_dB={row.mean_dB:+.8f} worst_dB={row.worst_dB:+.8f} "
            f"best_dB={row.best_dB:+.8f} wins={int(row.wins)}/{int(row.folds)}"
        )

    latest_fold = max(folds)
    latest = metrics.loc[metrics["fold"].eq(latest_fold)].copy()
    print(f"\n[{latest_fold} diagnostics | best by group]")
    groups = ["R", "F"] + sorted([g for g in latest["group"].unique() if str(g).startswith("NQ")])
    for group in groups:
        sub = latest.loc[latest["group"].eq(group)].sort_values("brier")
        if sub.empty:
            continue
        row = sub.iloc[0]
        print(
            f"{group:<4s} {row['variant']:<28s} brier={row['brier']:.8f} score={row['score']:+9.2f} "
            f"dB={row['delta_brier_vs_T0_same_group']:+.8f} n_med={row['n_median']:.1f}"
        )

    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
