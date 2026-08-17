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
import run_frozen_season_anchor_probe as frozen_core
import run_game_type_temporal_regime_ablation as regime_core
from src.utils import load_config, save_json, seed_everything


MODEL_NAMES = ("REGIME", "FROZEN_MULTI", "STABLE_NO_GT")


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise ValueError("at least one fold is required")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate folds: {values}")
    return sorted(values)


def alpha_grid(step: float) -> np.ndarray:
    if not (0.0 < step <= 1.0):
        raise ValueError("--alpha-step must be in (0,1]")
    values = np.arange(0.0, 1.0 + step * 0.5, step, dtype=np.float64)
    return np.unique(np.clip(np.append(values, 1.0), 0.0, 1.0))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return regime_core.binary_metrics(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64))


def routed_prediction(
    gt: np.ndarray,
    p_regime: np.ndarray,
    p_frozen: np.ndarray,
    p_stable: np.ndarray,
    alpha_r: float,
) -> np.ndarray:
    """F always uses regime; R blends frozen-state expert with stable expert."""
    pred = np.asarray(p_regime, dtype=np.float64).copy()
    r = np.asarray(gt).astype(str) == "R"
    pred[r] = float(alpha_r) * p_frozen[r] + (1.0 - float(alpha_r)) * p_stable[r]
    return pred


def best_alpha_for_rows(
    y: np.ndarray,
    gt: np.ndarray,
    p_regime: np.ndarray,
    p_frozen: np.ndarray,
    p_stable: np.ndarray,
    alphas: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_alpha = 1.0
    best_metric: dict[str, float] | None = None
    for alpha in alphas:
        p = routed_prediction(gt, p_regime, p_frozen, p_stable, float(alpha))
        m = metrics(y, p)
        if best_metric is None or m["brier"] < best_metric["brier"]:
            best_alpha = float(alpha)
            best_metric = m
    assert best_metric is not None
    return best_alpha, best_metric


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test the emerging conditional architecture: F uses the regime expert only; "
            "R uses frozen-season state plus a stable full-history expert. Includes a strict "
            "temporal alpha selection test where 2024 alpha_R is chosen only from earlier folds."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--stable-iterations", type=int, default=300)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--count-tolerance", type=float, default=0.05)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument("--output-dir", default="outputs/rf_conditional_frozen_stable_gate")
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if args.iterations <= 0 or args.stable_iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.count_tolerance <= 0.0:
        raise ValueError("--count-tolerance must be positive")
    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05,1.0]")

    folds = parse_ints(args.folds)
    alphas = alpha_grid(args.alpha_step)
    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)

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

    anchor_diag = frozen_core.add_frozen_anchor_features(
        frame,
        season_col=season_col,
        pitcher_col="pitcher_id",
        n_col="asof_pitcher_n",
        count_tolerance=args.count_tolerance,
    )
    regime_core.add_regime_features(
        frame,
        season_col=season_col,
        regime_start_year=args.regime_start_year,
    )

    base_features = recent_core.feature_set("recent_raw_game_type")
    frozen_sets = frozen_core.feature_sets(base_features)
    feature_sets = {
        "REGIME": frozen_sets["A0_REGIME"],
        "FROZEN_MULTI": frozen_sets["A2_FROZEN_MULTI"],
        "STABLE_NO_GT": recent_core.feature_set("recent_drop_game_type"),
    }

    main_params = regime_core.build_params(
        config=config,
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )
    stable_params = regime_core.build_params(
        config=config,
        iterations=args.stable_iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_diag.to_csv(output_dir / "anchor_diagnostics.csv", index=False)

    tqdm.write(
        f"RF conditional gate | folds={folds} | rows={len(frame):,} | alpha_step={args.alpha_step:.3f} "
        f"| GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | trees main/stable="
        f"{args.iterations}/{args.stable_iterations} | catboost={catboost.__version__}"
    )
    tqdm.write(
        "architecture: F -> REGIME only; R -> alpha*FROZEN_MULTI + (1-alpha)*STABLE_NO_GT. "
        "Frozen features obey independent-row inference."
    )

    fold_payload: dict[int, dict[str, np.ndarray]] = {}
    progress = tqdm(total=len(folds) * len(MODEL_NAMES), desc="fit conditional experts", unit="model", dynamic_ncols=True)

    for val_year in folds:
        train = frame.loc[frame[season_col] < val_year].copy()
        valid = frame.loc[frame[season_col].eq(val_year)].copy()
        if train.empty or valid.empty:
            raise ValueError(f"Fold {val_year}: empty train/valid")

        y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
        gt = valid["game_type"].astype(str).to_numpy()
        payload: dict[str, np.ndarray] = {"y": y, "gt": gt}
        if row_id_col in valid.columns:
            payload["row_id"] = valid[row_id_col].to_numpy()

        for name in MODEL_NAMES:
            seed_everything(seed)
            params = stable_params if name == "STABLE_NO_GT" else main_params
            pred = regime_core.fit_predict(
                train=train,
                valid=valid,
                target_col=target_col,
                features=feature_sets[name],
                extra_categorical=set(),
                params=params,
            )
            payload[name] = pred
            progress.update(1)

        fold_payload[int(val_year)] = payload
        del train, valid
        gc.collect()

    progress.close()

    # Per-fold diagnostics and alpha sweep.
    sweep_rows: list[dict] = []
    summary_rows: list[dict] = []
    for val_year in folds:
        d = fold_payload[val_year]
        y, gt = d["y"], d["gt"]
        r = gt == "R"
        f = gt == "F"

        candidates = {
            "B0_REGIME": d["REGIME"],
            "B2_FROZEN_MULTI": d["FROZEN_MULTI"],
            "STABLE_NO_GT": d["STABLE_NO_GT"],
            "G1_ROUTE_R_FROZEN_F_REGIME": np.where(r, d["FROZEN_MULTI"], d["REGIME"]),
        }
        for name, pred in candidates.items():
            for group, mask in (("ALL", np.ones(len(y), bool)), ("R", r), ("F", f)):
                m = metrics(y[mask], pred[mask])
                summary_rows.append({"fold": val_year, "name": name, "group": group, "alpha_r": np.nan, **m})

        for alpha in alphas:
            pred = routed_prediction(gt, d["REGIME"], d["FROZEN_MULTI"], d["STABLE_NO_GT"], float(alpha))
            m_all = metrics(y, pred)
            m_r = metrics(y[r], pred[r])
            sweep_rows.append(
                {
                    "fold": val_year,
                    "alpha_r": float(alpha),
                    "all_brier": m_all["brier"],
                    "all_score": m_all["score"],
                    "r_brier": m_r["brier"],
                    "r_score": m_r["score"],
                }
            )

        best_alpha, best_m = best_alpha_for_rows(
            y, gt, d["REGIME"], d["FROZEN_MULTI"], d["STABLE_NO_GT"], alphas
        )
        best_pred = routed_prediction(gt, d["REGIME"], d["FROZEN_MULTI"], d["STABLE_NO_GT"], best_alpha)
        for group, mask in (("ALL", np.ones(len(y), bool)), ("R", r), ("F", f)):
            m = metrics(y[mask], best_pred[mask])
            summary_rows.append({"fold": val_year, "name": "G3_ORACLE_ALPHA_R", "group": group, "alpha_r": best_alpha, **m})

    sweep_df = pd.DataFrame(sweep_rows)
    summary_df = pd.DataFrame(summary_rows)
    sweep_df.to_csv(output_dir / "alpha_sweep_by_fold.csv", index=False)
    summary_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    # Robust alpha: minimize row-weighted Brier across all folds.
    pooled_grid_rows: list[dict] = []
    for alpha in alphas:
        se_sum = 0.0
        n_sum = 0
        for val_year in folds:
            d = fold_payload[val_year]
            p = routed_prediction(d["gt"], d["REGIME"], d["FROZEN_MULTI"], d["STABLE_NO_GT"], float(alpha))
            se_sum += float(np.sum((d["y"] - p) ** 2))
            n_sum += int(len(d["y"]))
        pooled_grid_rows.append({"alpha_r": float(alpha), "pooled_brier": se_sum / n_sum})
    pooled_grid = pd.DataFrame(pooled_grid_rows).sort_values("pooled_brier").reset_index(drop=True)
    pooled_grid.to_csv(output_dir / "pooled_alpha_sweep.csv", index=False)
    pooled_alpha = float(pooled_grid.iloc[0]["alpha_r"])

    # Strict temporal test for the latest fold: choose alpha only from earlier validation folds.
    latest = max(folds)
    earlier = [year for year in folds if year < latest]
    temporal_rows: list[dict] = []
    temporal_alpha = np.nan
    if earlier:
        alpha_scores: list[tuple[float, float]] = []
        for alpha in alphas:
            se_sum = 0.0
            n_sum = 0
            for year in earlier:
                d = fold_payload[year]
                p = routed_prediction(d["gt"], d["REGIME"], d["FROZEN_MULTI"], d["STABLE_NO_GT"], float(alpha))
                se_sum += float(np.sum((d["y"] - p) ** 2))
                n_sum += int(len(d["y"]))
            alpha_scores.append((float(alpha), se_sum / n_sum))
        temporal_alpha = min(alpha_scores, key=lambda x: x[1])[0]
        d = fold_payload[latest]
        p = routed_prediction(d["gt"], d["REGIME"], d["FROZEN_MULTI"], d["STABLE_NO_GT"], temporal_alpha)
        r = d["gt"] == "R"
        f = d["gt"] == "F"
        base = d["REGIME"]
        for group, mask in (("ALL", np.ones(len(d["y"]), bool)), ("R", r), ("F", f)):
            m = metrics(d["y"][mask], p[mask])
            b = metrics(d["y"][mask], base[mask])
            temporal_rows.append(
                {
                    "validation_fold": latest,
                    "alpha_selected_from_folds": ",".join(map(str, earlier)),
                    "alpha_r": temporal_alpha,
                    "group": group,
                    **m,
                    "delta_brier_vs_regime": m["brier"] - b["brier"],
                }
            )
    temporal_df = pd.DataFrame(temporal_rows)
    temporal_df.to_csv(output_dir / "temporal_holdout_latest.csv", index=False)

    # Save latest-fold predictions for later ensemble work.
    latest_payload = fold_payload[latest]
    np.savez_compressed(
        output_dir / f"validation_{latest}_predictions.npz",
        y=latest_payload["y"],
        gt=latest_payload["gt"],
        regime=latest_payload["REGIME"],
        frozen_multi=latest_payload["FROZEN_MULTI"],
        stable=latest_payload["STABLE_NO_GT"],
    )

    save_json(
        {
            "experiment": "R/F conditional regime + frozen-state + stable gating",
            "folds": folds,
            "regime_start_year": int(args.regime_start_year),
            "alpha_step": float(args.alpha_step),
            "pooled_best_alpha_r": pooled_alpha,
            "latest_temporal_holdout_fold": latest,
            "latest_temporal_alpha_r": None if np.isnan(temporal_alpha) else float(temporal_alpha),
            "latest_temporal_alpha_selected_from": earlier,
            "routing": "F=REGIME; R=alpha*FROZEN_MULTI+(1-alpha)*STABLE_NO_GT",
            "iterations": int(args.iterations),
            "stable_iterations": int(args.stable_iterations),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Per-fold core models]")
    for year in folds:
        rows = summary_df[(summary_df["fold"] == year) & (summary_df["group"] == "ALL")]
        for _, row in rows.sort_values("brier").iterrows():
            alpha_text = "" if pd.isna(row["alpha_r"]) else f" alphaR={float(row['alpha_r']):.3f}"
            tqdm.write(
                f"fold={year} {str(row['name']):<28s}{alpha_text:<14s} "
                f"score={float(row['score']):+9.2f} brier={float(row['brier']):.8f}"
            )

    tqdm.write(f"\n[Pooled alpha across folds] alpha_R={pooled_alpha:.3f}")
    if not temporal_df.empty:
        tqdm.write(
            f"[Strict temporal latest-fold test] alpha_R={temporal_alpha:.3f} selected only from folds={earlier}"
        )
        for _, row in temporal_df.iterrows():
            tqdm.write(
                f"{str(row['group']):<3s} score={float(row['score']):+9.2f} "
                f"brier={float(row['brier']):.8f} dB_vs_REGIME={float(row['delta_brier_vs_regime']):+.8f}"
            )

    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
