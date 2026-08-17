from __future__ import annotations

import argparse
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


PAIR_SPECS = {
    "OLD": "CURRENT_RECENT_2023_GT",
    "FULL": "FULL_GT",
    "REGIME": "FULL_GT_RECENT_F",
}
STABLE_NAME = "STABLE_NO_GT"


def alpha_grid(step: float) -> np.ndarray:
    if not (0.0 < step <= 1.0):
        raise ValueError("--alpha-step must be in (0, 1]")
    values = np.arange(0.0, 1.0 + 0.5 * step, step, dtype=np.float64)
    return np.unique(np.clip(np.r_[values, 1.0], 0.0, 1.0))


def analytic_alpha(y: np.ndarray, p_expert: np.ndarray, p_stable: np.ndarray) -> float:
    """Exact Brier-optimal alpha for p=alpha*expert+(1-alpha)*stable."""
    y = np.asarray(y, dtype=np.float64)
    p_expert = np.asarray(p_expert, dtype=np.float64)
    p_stable = np.asarray(p_stable, dtype=np.float64)
    direction = p_expert - p_stable
    denom = float(np.dot(direction, direction))
    if denom <= 0.0:
        return 0.0
    value = float(np.dot(y - p_stable, direction) / denom)
    return float(np.clip(value, 0.0, 1.0))


def eval_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return regime_core.binary_metrics(y, p)


def fmt(name: str, alpha: float, metric: dict[str, float]) -> str:
    return (
        f"{name:<10s} alpha={alpha:5.3f}  "
        f"score={metric['score']:+9.2f}  "
        f"brier={metric['brier']:.8f}  "
        f"loss={metric['loss']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the current/pooled/regime-aware experts once, then sweep the blend weight "
            "against the stable expert in memory. The deployed alpha is selected on ALL 2024 rows; "
            "R/F-specific optima are reported only as diagnostics."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--valid-year", type=int, default=2024)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--recent-iterations", type=int, default=400)
    parser.add_argument("--regime-iterations", type=int, default=400)
    parser.add_argument("--stable-iterations", type=int, default=300)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument(
        "--output-dir",
        default="outputs/dual_track_recent_f_alpha_sweep",
    )
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05,1.0]")

    alphas = alpha_grid(float(args.alpha_step))
    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)

    target_col = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id_col = config["data"].get("row_id_col", "row_id")
    valid_year = int(args.valid_year)

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

    regime_core.add_regime_features(
        frame,
        season_col=season_col,
        regime_start_year=int(args.regime_start_year),
    )

    full_train = frame.loc[frame[season_col].between(2019, valid_year - 1)].copy()
    recent_train = frame.loc[frame[season_col].eq(valid_year - 1)].copy()
    valid = frame.loc[frame[season_col].eq(valid_year)].copy()
    if full_train.empty or recent_train.empty or valid.empty:
        raise ValueError("train/validation split is empty")

    recent_f_expected = recent_train["game_type"].eq("F").astype(np.float32).to_numpy()
    recent_f_actual = recent_train["eng_recent_f"].to_numpy(np.float32)
    if not np.array_equal(recent_f_expected, recent_f_actual):
        raise RuntimeError("eng_recent_f should be redundant in the one-season recent expert")

    base_gt_features = recent_core.feature_set("recent_raw_game_type")
    stable_features = recent_core.feature_set("recent_drop_game_type")
    regime_features = [*base_gt_features, "eng_recent_f"]

    recent_params = regime_core.build_params(
        config=config,
        iterations=int(args.recent_iterations),
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=float(args.gpu_ram_part),
        pinned_memory_size=args.pinned_memory_size,
    )
    regime_params = regime_core.build_params(
        config=config,
        iterations=int(args.regime_iterations),
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=float(args.gpu_ram_part),
        pinned_memory_size=args.pinned_memory_size,
    )
    stable_params = regime_core.build_params(
        config=config,
        iterations=int(args.stable_iterations),
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=float(args.gpu_ram_part),
        pinned_memory_size=args.pinned_memory_size,
    )

    expert_specs = {
        "CURRENT_RECENT_2023_GT": (recent_train, base_gt_features, recent_params),
        "FULL_GT": (full_train, base_gt_features, regime_params),
        "FULL_GT_RECENT_F": (full_train, regime_features, regime_params),
        STABLE_NAME: (full_train, stable_features, stable_params),
    }

    tqdm.write(
        f"Recent_F alpha sweep | full_train=2019-{valid_year-1} ({len(full_train):,}) | "
        f"recent_train={valid_year-1} ({len(recent_train):,}) | valid={valid_year} ({len(valid):,}) | "
        f"alpha_step={args.alpha_step:g} ({len(alphas)} points) | "
        f"GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | "
        f"trees recent/regime/stable={args.recent_iterations}/{args.regime_iterations}/{args.stable_iterations} | "
        f"catboost={catboost.__version__}"
    )

    expert_pred: dict[str, np.ndarray] = {}
    fit_bar = tqdm(total=len(expert_specs), desc="fit experts", unit="model", dynamic_ncols=True)
    for name, (train_part, features, params) in expert_specs.items():
        seed_everything(seed)
        expert_pred[name] = regime_core.fit_predict(
            train=train_part,
            valid=valid,
            target_col=target_col,
            features=features,
            extra_categorical=set(),
            params=params,
        )
        fit_bar.update(1)
    fit_bar.close()

    y = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    gt = valid["game_type"].astype(str).to_numpy()
    masks = {
        "ALL": np.ones(len(valid), dtype=bool),
        "R": gt == "R",
        "F": gt == "F",
    }

    p_stable = expert_pred[STABLE_NAME]
    sweep_rows: list[dict] = []
    total_sims = len(PAIR_SPECS) * len(alphas)
    sweep_bar = tqdm(total=total_sims, desc="simulate alpha", unit="blend", dynamic_ncols=True)
    for pair, expert_name in PAIR_SPECS.items():
        p_expert = expert_pred[expert_name]
        for alpha in alphas:
            pred = float(alpha) * p_expert + (1.0 - float(alpha)) * p_stable
            for group, mask in masks.items():
                metric = eval_metric(y[mask], pred[mask])
                sweep_rows.append(
                    {
                        "pair": pair,
                        "expert": expert_name,
                        "group": group,
                        "alpha": float(alpha),
                        "rows": int(mask.sum()),
                        "score": metric["score"],
                        "brier": metric["brier"],
                        "loss": metric["loss"],
                    }
                )
            sweep_bar.update(1)
    sweep_bar.close()

    sweep_df = pd.DataFrame(sweep_rows)

    # Exact Brier optimum for each pair/group. Group-specific values are diagnostics only.
    analytic_rows: list[dict] = []
    for pair, expert_name in PAIR_SPECS.items():
        p_expert = expert_pred[expert_name]
        for group, mask in masks.items():
            a = analytic_alpha(y[mask], p_expert[mask], p_stable[mask])
            pred = a * p_expert[mask] + (1.0 - a) * p_stable[mask]
            metric = eval_metric(y[mask], pred)
            analytic_rows.append(
                {
                    "pair": pair,
                    "expert": expert_name,
                    "group": group,
                    "alpha": a,
                    "score": metric["score"],
                    "brier": metric["brier"],
                    "loss": metric["loss"],
                }
            )
    analytic_df = pd.DataFrame(analytic_rows)

    # Select one deployable alpha per pair strictly on ALL rows, then report R/F at that same alpha.
    selected_rows: list[dict] = []
    for pair in PAIR_SPECS:
        all_rows = sweep_df.loc[(sweep_df["pair"].eq(pair)) & (sweep_df["group"].eq("ALL"))]
        best = all_rows.sort_values(["brier", "loss", "alpha"]).iloc[0]
        selected_alpha = float(best["alpha"])
        same_alpha = sweep_df.loc[
            sweep_df["pair"].eq(pair) & np.isclose(sweep_df["alpha"], selected_alpha)
        ].copy()
        same_alpha["selected_on"] = "ALL"
        selected_rows.extend(same_alpha.to_dict("records"))
    selected_df = pd.DataFrame(selected_rows)

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_dir / "alpha_sweep.csv", index=False)
    analytic_df.to_csv(output_dir / "analytic_optima.csv", index=False)
    selected_df.to_csv(output_dir / "selected_on_all.csv", index=False)

    pred_payload: dict[str, np.ndarray] = {
        "target": y.astype(np.float32),
        "game_type_is_f": (gt == "F").astype(np.int8),
    }
    if row_id_col in valid.columns:
        pred_payload["row_id"] = valid[row_id_col].to_numpy()
    for name, pred in expert_pred.items():
        pred_payload[name] = pred.astype(np.float32)
    np.savez_compressed(output_dir / "expert_predictions.npz", **pred_payload)

    best_regime_all = selected_df.loc[
        selected_df["pair"].eq("REGIME") & selected_df["group"].eq("ALL")
    ].iloc[0]
    save_json(
        {
            "experiment": "dual-track alpha sweep after explicit recent_F integration",
            "validation_year": valid_year,
            "full_train_seasons": list(range(2019, valid_year)),
            "recent_train_seasons": [valid_year - 1],
            "regime_start_year": int(args.regime_start_year),
            "alpha_step": float(args.alpha_step),
            "alpha_points": [float(x) for x in alphas],
            "pair_semantics": {
                "OLD": "alpha*CURRENT_RECENT_2023_GT + (1-alpha)*STABLE_NO_GT",
                "FULL": "alpha*FULL_GT + (1-alpha)*STABLE_NO_GT",
                "REGIME": "alpha*FULL_GT_RECENT_F + (1-alpha)*STABLE_NO_GT",
            },
            "selection_rule": "choose alpha by minimum 2024 ALL-row Brier; R/F optima are diagnostic only",
            "recommended_pair": "REGIME",
            "recommended_alpha_if_regime_pair_is_retained": float(best_regime_all["alpha"]),
            "recommended_validation_brier": float(best_regime_all["brier"]),
            "recent_iterations": int(args.recent_iterations),
            "regime_iterations": int(args.regime_iterations),
            "stable_iterations": int(args.stable_iterations),
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "gpu_ram_part": float(args.gpu_ram_part) if args.task_type == "GPU" else None,
            "pinned_memory_size": args.pinned_memory_size if args.task_type == "GPU" else None,
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Best grid alpha selected on ALL]")
    for pair in ("OLD", "FULL", "REGIME"):
        rows = selected_df.loc[selected_df["pair"].eq(pair)]
        a = float(rows.iloc[0]["alpha"])
        for group in ("ALL", "R", "F"):
            row = rows.loc[rows["group"].eq(group)].iloc[0]
            metric = {
                "score": float(row["score"]),
                "brier": float(row["brier"]),
                "loss": float(row["loss"]),
            }
            tqdm.write(fmt(f"{pair}/{group}", a, metric))

    tqdm.write("\n[Exact analytic alpha | diagnostic]")
    for pair in ("OLD", "FULL", "REGIME"):
        for group in ("ALL", "R", "F"):
            row = analytic_df.loc[
                analytic_df["pair"].eq(pair) & analytic_df["group"].eq(group)
            ].iloc[0]
            metric = {
                "score": float(row["score"]),
                "brier": float(row["brier"]),
                "loss": float(row["loss"]),
            }
            tqdm.write(fmt(f"{pair}/{group}", float(row["alpha"]), metric))

    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
