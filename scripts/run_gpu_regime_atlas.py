from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
import run_phone_regime_atlas as atlas
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything

DEFAULT_FOLDS = [2022, 2023, 2024]
BASELINE_VARIANTS = {
    "raw_game_type": "recent_raw_game_type",
    "drop_game_type": "recent_drop_game_type",
}


def _parse_years(value: str) -> list[int]:
    years = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not years:
        raise ValueError("at least one fold year is required")
    if years != sorted(set(years)):
        raise ValueError(f"fold years must be sorted and unique: {years}")
    return years


def _signal_names(frame: pd.DataFrame) -> list[str]:
    scalars = [name for name in atlas.SCALAR_SIGNALS if name in frame.columns]
    return scalars + atlas._available_interactions(frame)


def _signal_groups(frame: pd.DataFrame, signal: str, bins: int) -> tuple[pd.Series, str, int]:
    if signal in atlas.SCALAR_SIGNALS:
        categorical = signal in atlas.CATEGORICAL_HINTS
        return atlas._make_groups(frame[signal], categorical=categorical, bins=bins)
    groups = atlas._build_interaction(frame, signal, bins=bins)
    return groups, "interaction", int(groups.nunique(dropna=False))


def _fit_temporal_oof(
    frame: pd.DataFrame,
    *,
    features: list[str],
    fold_years: list[int],
    target_col: str,
    season_col: str,
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    from catboost import CatBoostClassifier, Pool

    predictions = np.full(len(frame), np.nan, dtype=np.float64)
    rows: list[dict] = []
    season = pd.to_numeric(frame[season_col], errors="raise").astype(int)

    for year in fold_years:
        train_mask = season.lt(year)
        valid_mask = season.eq(year)
        if not bool(train_mask.any()) or not bool(valid_mask.any()):
            raise RuntimeError(
                f"empty temporal fold {year}: train={int(train_mask.sum()):,}, valid={int(valid_mask.sum()):,}"
            )

        train = frame.loc[train_mask]
        valid = frame.loc[valid_mask]
        x_train, categorical = context_core.prepare_x(train, features)
        x_valid, valid_categorical = context_core.prepare_x(valid, features)
        if categorical != valid_categorical:
            raise RuntimeError(f"categorical mismatch in fold {year}")

        y_train = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.float32)
        y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        params = context_core.catboost_params(
            config=config,
            iterations=iterations,
            task_type=task_type,
            devices=devices,
            verbose=verbose,
        )
        train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
        valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, verbose=verbose)
        pred = np.asarray(model.predict_proba(valid_pool)[:, 1], dtype=np.float64)
        predictions[np.flatnonzero(valid_mask.to_numpy())] = pred
        metrics = probability_metrics(y_valid, pred)
        rows.append(
            {
                "fold_year": int(year),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "iterations": int(iterations),
                "feature_count": int(len(features)),
                **metrics,
            }
        )
        print(
            f"    fold={year} train={int(train_mask.sum()):,} valid={int(valid_mask.sum()):,} "
            f"brier={metrics['brier']:.8f} raw_score={metrics['raw_score']:+.2f}"
        )

        del model, train_pool, valid_pool, x_train, x_valid, y_train, y_valid, pred
        gc.collect()

    return predictions, pd.DataFrame(rows)


def _residual_profile(
    groups: pd.Series,
    residual: np.ndarray,
    season: pd.Series,
    year: int,
) -> pd.DataFrame:
    mask = season.eq(year).to_numpy() & np.isfinite(residual)
    if not mask.any():
        return pd.DataFrame(columns=["season", "group", "count", "mean", "effect", "season_prior"])
    r = residual[mask]
    g = groups.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    prior = float(np.mean(r))
    temp = pd.DataFrame({"group": g.to_numpy(), "r": r})
    prof = temp.groupby("group", dropna=False, sort=False)["r"].agg(["count", "mean"]).reset_index()
    prof["effect"] = prof["mean"] - prior
    prof["season"] = int(year)
    prof["season_prior"] = prior
    return prof[["season", "group", "count", "mean", "effect", "season_prior"]]


def _recent_residual_structure(
    p23: pd.DataFrame,
    p24: pd.DataFrame,
    *,
    min_year_count: int,
) -> dict[str, float | int]:
    a = p23.set_index("group")[["count", "effect"]].rename(columns={"count": "n23", "effect": "e23"})
    b = p24.set_index("group")[["count", "effect"]].rename(columns={"count": "n24", "effect": "e24"})
    joined = a.join(b, how="inner")
    joined = joined[(joined["n23"] >= min_year_count) & (joined["n24"] >= min_year_count)]
    if joined.empty:
        return {
            "recent_supported_groups": 0,
            "recent_residual_rms": float("nan"),
            "recent_residual_corr": float("nan"),
            "recent_same_sign_rate": float("nan"),
        }

    w = np.minimum(joined["n23"].to_numpy(float), joined["n24"].to_numpy(float))
    e23 = joined["e23"].to_numpy(float)
    e24 = joined["e24"].to_numpy(float)
    persistent = 0.5 * (e23 + e24)
    recent_rms = atlas._weighted_rmse(persistent, w)
    corr = atlas._safe_corr(e23, e24)
    strong = (np.abs(e23) >= 0.001) & (np.abs(e24) >= 0.001)
    strong_w = float(w[strong].sum())
    same = strong & (e23 * e24 > 0.0)
    same_sign = float(w[same].sum() / strong_w) if strong_w > 0 else 0.0
    return {
        "recent_supported_groups": int(len(joined)),
        "recent_residual_rms": recent_rms,
        "recent_residual_corr": corr,
        "recent_same_sign_rate": same_sign,
    }


def _audit_residual_signal(
    signal: str,
    groups: pd.Series,
    frame: pd.DataFrame,
    residual: np.ndarray,
    *,
    min_year_count: int,
) -> tuple[dict, pd.DataFrame]:
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    profiles = {
        year: _residual_profile(groups, residual, season, year)
        for year in [2022, 2023, 2024]
    }
    shock = atlas._pair_rmse(profiles[2022], profiles[2023], min_year_count)
    post = atlas._pair_rmse(profiles[2023], profiles[2024], min_year_count)
    recent = _recent_residual_structure(
        profiles[2023], profiles[2024], min_year_count=min_year_count
    )

    recent_rms = float(recent["recent_residual_rms"])
    corr = float(recent["recent_residual_corr"])
    same_sign = float(recent["recent_same_sign_rate"])
    if np.isfinite(shock):
        shock_ratio = float(shock / max(post if np.isfinite(post) else 0.0, 0.002))
    else:
        shock_ratio = float("nan")

    if np.isfinite(recent_rms):
        score = float(
            recent_rms
            * np.clip(shock_ratio if np.isfinite(shock_ratio) else 1.0, 0.5, 8.0)
            * (0.5 + (same_sign if np.isfinite(same_sign) else 0.0))
            * np.clip((corr + 1.0) / 2.0 if np.isfinite(corr) else 0.5, 0.25, 1.0)
        )
    else:
        score = float("nan")

    if not np.isfinite(recent_rms):
        classification = "insufficient_support"
    elif recent_rms >= 0.003 and np.isfinite(corr) and corr >= 0.25 and same_sign >= 0.60:
        classification = "persistent_recent_residual"
    elif (
        np.isfinite(shock)
        and shock >= 0.005
        and np.isfinite(post)
        and post <= 0.8 * shock
        and recent_rms >= 0.002
    ):
        classification = "changepoint_residual"
    elif recent_rms >= 0.0025:
        classification = "residual_signal"
    else:
        classification = "weak"

    long = pd.concat(
        [profiles[y].assign(signal=signal) for y in [2022, 2023, 2024]],
        ignore_index=True,
    )
    summary = {
        "signal": signal,
        "groups": int(groups.nunique(dropna=False)),
        "shock_2022_2023_residual_rmse": shock,
        "post_2023_2024_residual_rmse": post,
        "residual_shock_ratio": shock_ratio,
        **recent,
        "residual_regime_score": score,
        "residual_classification": classification,
    }
    return summary, long


def _target_atlas(
    frame: pd.DataFrame,
    *,
    signals: list[str],
    bins: int,
    min_era_count: int,
    min_year_count: int,
    same_pitcher_min_old: int,
    same_pitcher_min_recent: int,
) -> tuple[pd.DataFrame, dict]:
    cohort_mask, cohort_info = atlas._same_entity_mask(
        frame,
        "pitcher_id",
        min_old=same_pitcher_min_old,
        min_recent=same_pitcher_min_recent,
    )
    rows: list[dict] = []
    for idx, signal in enumerate(signals, start=1):
        groups, grouping, group_count = _signal_groups(frame, signal, bins)
        summary, _ = atlas._audit_one(
            signal,
            groups,
            frame,
            min_era_count=min_era_count,
            min_year_count=min_year_count,
            cohort_mask=cohort_mask,
        )
        summary["grouping"] = grouping
        summary["groups"] = group_count
        rows.append(summary)
        print(
            f"  [target {idx:02d}/{len(signals):02d}] {signal:<36} "
            f"class={summary['classification']:<22} score={summary['regime_score']:.6f}"
        )
        del groups
        gc.collect()
    result = pd.DataFrame(rows).sort_values("regime_score", ascending=False, na_position="last")
    return result, cohort_info


def _residual_atlas(
    frame: pd.DataFrame,
    residual: np.ndarray,
    *,
    signals: list[str],
    bins: int,
    min_year_count: int,
    baseline_name: str,
    keep_top_group_profiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    longs: list[pd.DataFrame] = []
    for idx, signal in enumerate(signals, start=1):
        groups, grouping, group_count = _signal_groups(frame, signal, bins)
        summary, long = _audit_residual_signal(
            signal,
            groups,
            frame,
            residual,
            min_year_count=min_year_count,
        )
        summary["grouping"] = grouping
        summary["groups"] = group_count
        summary["baseline"] = baseline_name
        summaries.append(summary)
        long["baseline"] = baseline_name
        longs.append(long)
        print(
            f"  [resid {idx:02d}/{len(signals):02d}] {signal:<36} "
            f"class={summary['residual_classification']:<26} "
            f"rms={summary['recent_residual_rms']:.5f}"
        )
        del groups, long
        gc.collect()

    summary_df = pd.DataFrame(summaries).sort_values(
        "residual_regime_score", ascending=False, na_position="last"
    )
    if keep_top_group_profiles <= 0:
        return summary_df, pd.DataFrame()

    top_signals = set(summary_df.head(keep_top_group_profiles)["signal"])
    selected = [part for part in longs if not part.empty and part["signal"].iloc[0] in top_signals]
    long_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    return summary_df, long_df


def _evidence_label(row: pd.Series) -> str:
    target_class = str(row.get("classification", ""))
    raw_class = str(row.get("raw_residual_classification", ""))
    raw_rms = row.get("raw_recent_residual_rms", np.nan)
    drop_rms = row.get("drop_recent_residual_rms", np.nan)

    residual_strong = raw_class in {"persistent_recent_residual", "changepoint_residual"}
    target_strong = target_class == "regime_candidate"
    if target_strong and residual_strong:
        return "STRONG_NEW_EXPERT_CANDIDATE"
    if residual_strong:
        return "UNEXPLAINED_RECENT_SIGNAL"
    if target_strong and np.isfinite(raw_rms) and np.isfinite(drop_rms) and drop_rms > 1.20 * max(raw_rms, 1e-9):
        return "RAW_GAME_TYPE_ABSORBS_SHIFT"
    if target_strong:
        return "TARGET_SHIFT_MOSTLY_MODELED"
    return "LOW_PRIORITY"


def _combine_rankings(
    target_df: pd.DataFrame,
    residual_by_baseline: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    result = target_df.copy()
    for baseline_name, residual_df in residual_by_baseline.items():
        prefix = "raw" if baseline_name == "raw_game_type" else "drop"
        keep = residual_df[
            [
                "signal",
                "recent_residual_rms",
                "recent_residual_corr",
                "recent_same_sign_rate",
                "shock_2022_2023_residual_rmse",
                "post_2023_2024_residual_rmse",
                "residual_shock_ratio",
                "residual_regime_score",
                "residual_classification",
            ]
        ].rename(columns={c: f"{prefix}_{c}" for c in residual_df.columns if c != "signal"})
        result = result.merge(keep, on="signal", how="left")

    target_pct = result["regime_score"].rank(pct=True, method="average")
    raw_pct = result.get("raw_residual_regime_score", pd.Series(np.nan, index=result.index)).rank(
        pct=True, method="average"
    )
    result["combined_rank_score"] = 0.55 * target_pct.fillna(0.0) + 0.45 * raw_pct.fillna(0.0)
    result["expert_evidence"] = result.apply(_evidence_label, axis=1)
    evidence_rank = {
        "STRONG_NEW_EXPERT_CANDIDATE": 0,
        "UNEXPLAINED_RECENT_SIGNAL": 1,
        "RAW_GAME_TYPE_ABSORBS_SHIFT": 2,
        "TARGET_SHIFT_MOSTLY_MODELED": 3,
        "LOW_PRIORITY": 4,
    }
    result["_evidence_rank"] = result["expert_evidence"].map(evidence_rank).fillna(9)
    return result.sort_values(
        ["_evidence_rank", "combined_rank_score"], ascending=[True, False]
    ).drop(columns=["_evidence_rank"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GPU regime discovery for Colab. First builds a model-free target atlas, then trains "
            "strict temporal CatBoost OOF baselines and searches their residuals for persistent "
            "post-2023 signals that can justify a third/fourth expert."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--baselines", default="raw_game_type,drop_game_type")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-era-count", type=int, default=500)
    parser.add_argument("--min-year-count", type=int, default=100)
    parser.add_argument("--same-pitcher-min-old", type=int, default=200)
    parser.add_argument("--same-pitcher-min-recent", type=int, default=100)
    parser.add_argument("--keep-top-group-profiles", type=int, default=12)
    parser.add_argument("--save-oof", action="store_true")
    parser.add_argument("--output-dir", default="outputs/gpu_regime_atlas")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.bins < 3:
        raise ValueError("--bins must be >= 3")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    fold_years = _parse_years(args.folds)
    baseline_names = [x.strip() for x in args.baselines.split(",") if x.strip()]
    unknown = sorted(set(baseline_names) - set(BASELINE_VARIANTS))
    if unknown:
        raise ValueError(f"unknown baselines: {unknown}")

    frame, invariant_check = recent_core.prepare_frame(config)
    sort_cols = [season_col, "game_month"]
    if row_id_col in frame.columns:
        sort_cols.append(row_id_col)
    frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    frame[season_col] = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    if season_col != "season":
        frame["season"] = frame[season_col]
    if target_col != "control_success":
        frame["control_success"] = pd.to_numeric(frame[target_col], errors="raise")

    signals = _signal_names(frame)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[GPU Regime Atlas]")
    print(f"  rows        : {len(frame):,}")
    print(f"  folds       : {fold_years}")
    print(f"  iterations  : {args.iterations}")
    print(f"  task_type   : {args.task_type}")
    print(f"  baselines   : {baseline_names}")
    print(f"  signals     : {len(signals)}")
    print("  purpose     : find regime signals that remain in temporal OOF residuals")

    print("\n[1/3] Model-free target regime atlas")
    target_df, cohort_info = _target_atlas(
        frame,
        signals=signals,
        bins=args.bins,
        min_era_count=args.min_era_count,
        min_year_count=args.min_year_count,
        same_pitcher_min_old=args.same_pitcher_min_old,
        same_pitcher_min_recent=args.same_pitcher_min_recent,
    )
    target_df.to_csv(output_dir / "target_regime_atlas.csv", index=False)

    print("\n[2/3] Strict temporal OOF CatBoost + residual atlas")
    fold_metric_parts: list[pd.DataFrame] = []
    residual_summaries: dict[str, pd.DataFrame] = {}
    residual_profiles: list[pd.DataFrame] = []
    oof_payload: dict[str, np.ndarray] = {}

    y = pd.to_numeric(frame[target_col], errors="raise").to_numpy(np.float64)
    for baseline_name in baseline_names:
        variant = BASELINE_VARIANTS[baseline_name]
        features = recent_core.feature_set(variant)
        print(f"\n  [{baseline_name}] features={len(features)}")
        seed_everything(seed)
        pred, metrics_df = _fit_temporal_oof(
            frame,
            features=features,
            fold_years=fold_years,
            target_col=target_col,
            season_col=season_col,
            config=config,
            iterations=args.iterations,
            task_type=args.task_type,
            devices=args.devices,
            verbose=args.verbose,
        )
        metrics_df.insert(0, "baseline", baseline_name)
        fold_metric_parts.append(metrics_df)
        residual = y - pred
        oof_payload[baseline_name] = pred
        summary_df, profile_df = _residual_atlas(
            frame,
            residual,
            signals=signals,
            bins=args.bins,
            min_year_count=args.min_year_count,
            baseline_name=baseline_name,
            keep_top_group_profiles=args.keep_top_group_profiles,
        )
        residual_summaries[baseline_name] = summary_df
        summary_df.to_csv(output_dir / f"residual_regime_atlas_{baseline_name}.csv", index=False)
        if not profile_df.empty:
            residual_profiles.append(profile_df)
        del residual, pred
        gc.collect()

    fold_metrics = pd.concat(fold_metric_parts, ignore_index=True)
    fold_metrics.to_csv(output_dir / "oof_fold_metrics.csv", index=False)
    if residual_profiles:
        pd.concat(residual_profiles, ignore_index=True).to_csv(
            output_dir / "top_residual_group_profiles.csv", index=False
        )

    print("\n[3/3] Combined expert-candidate ranking")
    combined = _combine_rankings(target_df, residual_summaries)
    combined.to_csv(output_dir / "expert_candidate_ranking.csv", index=False)

    if args.save_oof:
        oof = pd.DataFrame(
            {
                "season": frame[season_col].to_numpy(),
                "control_success": y,
            }
        )
        if row_id_col in frame.columns:
            oof.insert(0, row_id_col, frame[row_id_col].to_numpy())
        for name, pred in oof_payload.items():
            oof[f"pred_{name}"] = pred
            oof[f"residual_{name}"] = y - pred
        valid_any = np.zeros(len(frame), dtype=bool)
        for name in oof_payload:
            valid_any |= np.isfinite(oof_payload[name])
        oof.loc[valid_any].to_csv(output_dir / "temporal_oof_predictions.csv", index=False)

    top_cols = [
        "signal",
        "expert_evidence",
        "classification",
        "regime_score",
        "raw_residual_classification",
        "raw_recent_residual_rms",
        "raw_recent_residual_corr",
        "raw_recent_same_sign_rate",
        "combined_rank_score",
    ]
    top_cols = [c for c in top_cols if c in combined.columns]
    print("\n[Top expert candidates]")
    print(combined.head(20)[top_cols].to_string(index=False))

    payload = {
        "folds": fold_years,
        "iterations": int(args.iterations),
        "task_type": args.task_type,
        "devices": args.devices,
        "baselines": baseline_names,
        "bins": int(args.bins),
        "min_era_count": int(args.min_era_count),
        "min_year_count": int(args.min_year_count),
        "same_pitcher_cohort": cohort_info,
        "signals": signals,
        "canonical_invariants": invariant_check,
        "interpretation": {
            "STRONG_NEW_EXPERT_CANDIDATE": "target relation shifts and the raw-game_type temporal model still leaves persistent recent residual structure",
            "UNEXPLAINED_RECENT_SIGNAL": "persistent recent residual structure even without a strong marginal target-regime label",
            "RAW_GAME_TYPE_ABSORBS_SHIFT": "target relation shifts, but raw game_type materially reduces the residual structure versus dropping it",
            "TARGET_SHIFT_MOSTLY_MODELED": "marginal target shift exists but current raw baseline largely absorbs it",
            "LOW_PRIORITY": "no current evidence for a separate expert",
        },
    }
    save_json(payload, output_dir / "run_config.json")
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
