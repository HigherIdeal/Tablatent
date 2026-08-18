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
SUCCESS_WINDOWS = (3, 5, 10, 20)
AUX_WINDOWS = (5, 20)
EXTRA_SUCCESS_WINDOWS = (2, 30, 50)
EXTRA_AUX_WINDOWS = (3, 10, 50)
VARIANTS = (
    "A0_REGIME",
    "A1_PREV1_SUCCESS",
    "A2_PITCH_SUCCESS_STATE",
    "A3_MULTI_PREFIX_STATE",
    "A4_MULTI_SCALE_STATE",
    "A5_SEQUENCE_STATE",
)


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one fold is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate folds: {values}")
    return sorted(values)


def _safe_integer_count(n: np.ndarray, rate: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    raw = n * rate
    rounded = np.rint(raw)
    valid = np.isfinite(raw) & (np.abs(raw - rounded) <= float(tolerance))
    return rounded, valid


def add_prefix_inversion_features(
    frame: pd.DataFrame,
    *,
    pitcher_col: str,
    n_col: str,
    count_tolerance: float,
) -> dict[str, float | int]:
    """Invert cumulative asof rates into prior-pitch states without reading target labels.

    At a row with cumulative pre-pitch state (n_t, rate_t), and the immediately
    preceding row for the same pitcher (n_t-1, rate_t-1), the difference of the
    rounded cumulative counts recovers the outcome/state of the preceding pitch.
    Only transitions with delta n == 1 are accepted. Rolling features are confined
    to contiguous-n segments, so they never jump across missing observations.
    """
    required = {pitcher_col, n_col, *PREFIX_RATES.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prefix-inversion columns: {missing}")

    work = frame.loc[:, [pitcher_col, n_col, *PREFIX_RATES.values()]].copy()
    work["_orig_index"] = np.arange(len(work), dtype=np.int64)
    work[n_col] = pd.to_numeric(work[n_col], errors="raise").astype(np.int64)
    work = work.sort_values([pitcher_col, n_col, "_orig_index"], kind="stable").reset_index(drop=True)

    duplicate_n = int(work.duplicated([pitcher_col, n_col], keep=False).sum())
    prev_pitcher = work[pitcher_col].shift(1)
    prev_n = work[n_col].shift(1)
    same_pitcher = work[pitcher_col].eq(prev_pitcher)
    contiguous = same_pitcher & work[n_col].eq(prev_n + 1)

    # Start a new segment whenever the observed cumulative count is not contiguous.
    segment_break = (~contiguous).astype(np.int64)
    work["_segment"] = segment_break.groupby(work[pitcher_col], sort=False).cumsum()

    diagnostics: dict[str, float | int] = {
        "rows": int(len(work)),
        "duplicate_pitcher_n_rows": duplicate_n,
        "contiguous_transition_rows": int(contiguous.sum()),
        "contiguous_transition_share": float(contiguous.mean()),
    }

    reconstructed_names: list[str] = []
    for short, column in PREFIX_RATES.items():
        rate = pd.to_numeric(work[column], errors="coerce").to_numpy(np.float64)
        n = work[n_col].to_numpy(np.float64)
        count, count_valid = _safe_integer_count(n, rate, count_tolerance)
        prev_count = np.roll(count, 1)
        prev_valid = np.roll(count_valid, 1)
        prev_valid[0] = False

        delta = count - prev_count
        valid = (
            contiguous.to_numpy(bool)
            & count_valid
            & prev_valid
            & np.isfinite(delta)
            & ((delta == 0.0) | (delta == 1.0))
        )
        values = np.full(len(work), np.nan, dtype=np.float32)
        values[valid] = delta[valid].astype(np.float32)
        name = f"eng_prev_pitch_{short}_recon"
        work[name] = values
        reconstructed_names.append(name)
        diagnostics[f"{short}_recon_rows"] = int(valid.sum())
        diagnostics[f"{short}_recon_share"] = float(valid.mean())

    for short in PREFIX_RATES:
        source = f"eng_prev_pitch_{short}_recon"
        for lag in (2, 3, 4, 5):
            work[f"eng_pitch_{short}_lag{lag}"] = work[source].groupby(
                [work[pitcher_col], work["_segment"]], sort=False
            ).shift(lag - 1).astype(np.float32)

    # Pitch-level recent success state. For row t, eng_prev_pitch_success_recon is
    # the outcome of t-1; rolling windows ending at t therefore use only pitches
    # strictly before the row being predicted.
    success_name = "eng_prev_pitch_success_recon"
    seg_keys = [work[pitcher_col], work["_segment"]]
    for window in SUCCESS_WINDOWS:
        out = f"eng_pitch_success_last{window}"
        work[out] = (
            work[success_name]
            .groupby(seg_keys, sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
            .astype(np.float32)
        )
    for window in EXTRA_SUCCESS_WINDOWS:
        work[f"eng_pitch_success_last{window}"] = (
            work[success_name].groupby(seg_keys, sort=False).rolling(window, min_periods=1)
            .mean().reset_index(level=[0, 1], drop=True).astype(np.float32)
        )

    # Relative-to-long features give shallow trees a direct local-vs-career state.
    long_success = pd.to_numeric(work["asof_pitcher_success_rate"], errors="coerce").astype(np.float32)
    for window in (3, 10, 20):
        work[f"eng_pitch_success_last{window}_minus_long"] = (
            work[f"eng_pitch_success_last{window}"] - long_success
        ).astype(np.float32)

    # Auxiliary reconstructed states: keep the expansion small and interpretable.
    for short in ("reverse", "middle", "ball", "strike"):
        source = f"eng_prev_pitch_{short}_recon"
        for window in AUX_WINDOWS:
            out = f"eng_pitch_{short}_last{window}"
            work[out] = (
                work[source]
                .groupby(seg_keys, sort=False)
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
                .astype(np.float32)
            )
        for window in EXTRA_AUX_WINDOWS:
            work[f"eng_pitch_{short}_last{window}"] = (
                work[source].groupby(seg_keys, sort=False).rolling(window, min_periods=1)
                .mean().reset_index(level=[0, 1], drop=True).astype(np.float32)
            )

    new_columns = [c for c in work.columns if c.startswith("eng_prev_pitch_") or c.startswith("eng_pitch_")]
    restored = work.sort_values("_orig_index", kind="stable")
    for column in new_columns:
        frame[column] = restored[column].to_numpy()

    diagnostics["engineered_feature_count"] = int(len(new_columns))
    return diagnostics


def reconstruction_audit(
    frame: pd.DataFrame,
    *,
    pitcher_col: str,
    n_col: str,
    target_col: str,
    season_col: str,
) -> pd.DataFrame:
    """Diagnostic only: compare target-free reconstruction with the known prior label."""
    cols = [pitcher_col, n_col, target_col, season_col, "eng_prev_pitch_success_recon"]
    work = frame.loc[:, cols].copy()
    work["_orig_index"] = np.arange(len(work), dtype=np.int64)
    work[n_col] = pd.to_numeric(work[n_col], errors="raise").astype(np.int64)
    work = work.sort_values([pitcher_col, n_col, "_orig_index"], kind="stable").reset_index(drop=True)

    same = work[pitcher_col].eq(work[pitcher_col].shift(1))
    contiguous = same & work[n_col].eq(work[n_col].shift(1) + 1)
    prev_y = pd.to_numeric(work[target_col], errors="raise").shift(1)
    recon = pd.to_numeric(work["eng_prev_pitch_success_recon"], errors="coerce")
    available = contiguous & recon.notna() & prev_y.notna()
    exact = available & recon.eq(prev_y)

    rows: list[dict] = []
    for label, mask in [("ALL", pd.Series(True, index=work.index))] + [
        (str(int(year)), work[season_col].eq(year)) for year in sorted(work[season_col].unique())
    ]:
        eligible = contiguous & mask
        used = available & mask
        correct = exact & mask
        rows.append(
            {
                "season": label,
                "eligible_contiguous": int(eligible.sum()),
                "reconstructed": int(used.sum()),
                "coverage_of_contiguous": float(used.sum() / eligible.sum()) if eligible.sum() else np.nan,
                "exact_matches": int(correct.sum()),
                "accuracy": float(correct.sum() / used.sum()) if used.sum() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    regime = [*base_features, "eng_recent_f"]
    prev1 = [*regime, "eng_prev_pitch_success_recon"]
    success_state = [
        *prev1,
        *[f"eng_pitch_success_last{w}" for w in SUCCESS_WINDOWS],
        "eng_pitch_success_last3_minus_long",
        "eng_pitch_success_last10_minus_long",
        "eng_pitch_success_last20_minus_long",
    ]
    multi = [
        *success_state,
        "eng_prev_pitch_reverse_recon",
        "eng_prev_pitch_middle_recon",
        "eng_prev_pitch_ball_recon",
        "eng_prev_pitch_strike_recon",
        *[
            f"eng_pitch_{state}_last{window}"
            for state in ("reverse", "middle", "ball", "strike")
            for window in AUX_WINDOWS
        ],
    ]
    out = {
        "A0_REGIME": regime,
        "A1_PREV1_SUCCESS": prev1,
        "A2_PITCH_SUCCESS_STATE": success_state,
        "A3_MULTI_PREFIX_STATE": multi,
        "A4_MULTI_SCALE_STATE": [
            *multi,
            *[f"eng_pitch_success_last{w}" for w in EXTRA_SUCCESS_WINDOWS],
            *[f"eng_pitch_{state}_last{window}" for state in ("reverse", "middle", "ball", "strike") for window in EXTRA_AUX_WINDOWS],
        ],
        "A5_SEQUENCE_STATE": [
            *multi,
            *[f"eng_pitch_success_last{w}" for w in EXTRA_SUCCESS_WINDOWS],
            *[f"eng_pitch_{state}_last{window}" for state in ("reverse", "middle", "ball", "strike") for window in EXTRA_AUX_WINDOWS],
            *[f"eng_pitch_{state}_lag{lag}" for state in PREFIX_RATES for lag in (2, 3, 4, 5)],
        ],
    }
    for name, features in out.items():
        if len(features) != len(set(features)):
            raise RuntimeError(f"duplicate features in {name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "High-leverage probe: invert supplied cumulative asof pitcher rates into prior-pitch "
            "states using feature columns only, then test whether pitch-level recent state improves "
            "the regime-aware CatBoost. No current-row or future target is used to construct features."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--objective", choices=["logloss", "brier"], default="logloss")
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--count-tolerance", type=float, default=0.05)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/asof_prefix_inversion_probe")
    args = parser.parse_args()
    selected_variants = tuple(x.strip() for x in args.variants.split(",") if x.strip())
    if not selected_variants or not set(selected_variants) <= set(VARIANTS):
        raise ValueError(f"bad --variants: {selected_variants}")

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

    # Construct the new state from feature columns only. The target column is not
    # passed to add_prefix_inversion_features and is used only afterward for audit/scoring.
    inversion_diag = add_prefix_inversion_features(
        frame,
        pitcher_col=pitcher_col,
        n_col=n_col,
        count_tolerance=args.count_tolerance,
    )
    regime_core.add_regime_features(
        frame,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )

    audit = reconstruction_audit(
        frame,
        pitcher_col=pitcher_col,
        n_col=n_col,
        target_col=target_col,
        season_col=season_col,
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
    params["depth"] = args.depth
    params["_regression"] = args.objective == "brier"

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "reconstruction_audit.csv", index=False)

    tqdm.write(
        f"ASOF prefix inversion | folds={folds} | rows={len(frame):,} | "
        f"contiguous={inversion_diag['contiguous_transition_share']:.4f} | "
        f"success_recon={inversion_diag['success_recon_share']:.4f} | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | "
        f"iterations={args.iterations} | catboost={catboost.__version__}"
    )
    tqdm.write("feature construction uses pitcher_id + asof_n + cumulative asof rates only; target is audit/scoring only.")
    tqdm.write("\n[Reconstruction audit: previous control_success]")
    for _, row in audit.iterrows():
        tqdm.write(
            f"season={str(row['season']):>4s} eligible={int(row['eligible_contiguous']):>8,d} "
            f"coverage={float(row['coverage_of_contiguous']):.6f} accuracy={float(row['accuracy']):.6f}"
        )

    rows: list[dict] = []
    predictions_2024: dict[str, np.ndarray] = {}
    total_models = len(folds) * len(selected_variants)
    progress = tqdm(total=total_models, desc="prefix inversion models", unit="model", dynamic_ncols=True)

    for val_year in folds:
        train = frame.loc[frame[season_col] < val_year].copy()
        valid = frame.loc[frame[season_col] == val_year].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train/valid")
        y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt_valid = valid["game_type"].astype(str).to_numpy()
        recon_available = valid["eng_prev_pitch_success_recon"].notna().to_numpy()

        fold_predictions: dict[str, np.ndarray] = {}
        for variant in selected_variants:
            seed_everything(seed)
            pred = regime_core.fit_predict(
                train=train,
                valid=valid,
                target_col=target_col,
                features=variants[variant],
                extra_categorical=set(),
                params=params,
            )
            fold_predictions[variant] = pred
            progress.update(1)

        baseline = fold_predictions["A0_REGIME"]
        masks = {
            "ALL": np.ones(len(valid), dtype=bool),
            "R": gt_valid == "R",
            "F": gt_valid == "F",
            "RECON_AVAILABLE": recon_available,
            "RECON_MISSING": ~recon_available,
        }
        baseline_metrics = {
            group: regime_core.binary_metrics(y_valid[mask], baseline[mask])
            for group, mask in masks.items()
            if int(mask.sum()) > 0
        }

        for variant, pred in fold_predictions.items():
            for group, mask in masks.items():
                if int(mask.sum()) == 0:
                    continue
                metric = regime_core.binary_metrics(y_valid[mask], pred[mask])
                rows.append(
                    {
                        "validation_year": int(val_year),
                        "variant": variant,
                        "group": group,
                        "rows": int(mask.sum()),
                        "feature_count": int(len(variants[variant])),
                        "score": metric["score"],
                        "brier": metric["brier"],
                        "loss": metric["loss"],
                        "delta_brier_vs_A0_same_group": (
                            metric["brier"] - baseline_metrics[group]["brier"]
                        ),
                    }
                )

        if True:
            if val_year == 2024:
                predictions_2024 = fold_predictions
            pred_frame = pd.DataFrame(
                {
                    "target": y_valid,
                    "game_type": gt_valid,
                    "recon_available": recon_available.astype(np.int8),
                    **{f"{k}_probability": v for k, v in fold_predictions.items()},
                }
            )
            if row_id_col in valid.columns:
                pred_frame.insert(0, row_id_col, valid[row_id_col].to_numpy())
            pred_frame.to_csv(output_dir / f"validation_{val_year}_predictions.csv", index=False)

        del train, valid, y_valid, fold_predictions
        gc.collect()

    progress.close()
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "metrics.csv", index=False)

    overall = results.loc[results["group"].eq("ALL")].copy()
    summary = (
        overall.groupby("variant", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_brier=("brier", "mean"),
            mean_delta=("delta_brier_vs_A0_same_group", "mean"),
            worst_delta=("delta_brier_vs_A0_same_group", "max"),
            best_delta=("delta_brier_vs_A0_same_group", "min"),
            wins=("delta_brier_vs_A0_same_group", lambda s: int((s < 0).sum())),
        )
        .sort_values(["mean_delta", "worst_delta"])
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    save_json(
        {
            "experiment": "target-free inversion of cumulative asof pitcher prefixes into prior-pitch state",
            "folds": folds,
            "regime_start_year": int(args.regime_start_year),
            "count_tolerance": float(args.count_tolerance),
            "prefix_rates": PREFIX_RATES,
            "success_windows": list(SUCCESS_WINDOWS),
            "aux_windows": list(AUX_WINDOWS),
            "variants": variants,
            "inversion_diagnostics": inversion_diag,
            "iterations": int(args.iterations),
            "depth": int(args.depth),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "gpu_ram_part": float(args.gpu_ram_part) if args.task_type == "GPU" else None,
            "pinned_memory_size": args.pinned_memory_size if args.task_type == "GPU" else None,
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
            "leakage_guard": (
                "engineered prefix-inversion features are computed without target labels; "
                "target is consulted only after construction for reconstruction audit and validation scoring"
            ),
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Fold results | ALL]")
    for val_year in folds:
        subset = overall.loc[overall["validation_year"].eq(val_year)].sort_values("brier")
        tqdm.write(f"fold={val_year}")
        for _, row in subset.iterrows():
            tqdm.write(
                regime_core.metric_line(
                    str(row["variant"]),
                    {"score": float(row["score"]), "brier": float(row["brier"]), "loss": float(row["loss"])},
                    float(row["delta_brier_vs_A0_same_group"]),
                )
            )

    tqdm.write("\n[Cross-fold summary | lower delta is better]")
    for _, row in summary.iterrows():
        tqdm.write(
            f"{str(row['variant']):<24s} mean_dB={float(row['mean_delta']):+.8f} "
            f"worst_dB={float(row['worst_delta']):+.8f} best_dB={float(row['best_delta']):+.8f} "
            f"wins={int(row['wins'])}/{int(row['folds'])}"
        )

    if predictions_2024:
        tqdm.write("\n[2024 diagnostics | best ALL variant]")
        v2024 = results.loc[(results["validation_year"].eq(2024)) & (results["group"].eq("ALL"))]
        best_name = str(v2024.sort_values("brier").iloc[0]["variant"])
        for group in ("ALL", "R", "F", "RECON_AVAILABLE", "RECON_MISSING"):
            row = results.loc[
                results["validation_year"].eq(2024)
                & results["variant"].eq(best_name)
                & results["group"].eq(group)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            tqdm.write(
                f"{best_name}/{group:<15s} score={float(r['score']):+9.2f} "
                f"brier={float(r['brier']):.8f} loss={float(r['loss']):.8f} "
                f"dB={float(r['delta_brier_vs_A0_same_group']):+.8f}"
            )

    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
