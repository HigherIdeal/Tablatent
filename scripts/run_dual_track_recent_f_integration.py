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


EXPERTS = (
    "CURRENT_RECENT_2023_GT",
    "FULL_GT",
    "FULL_GT_RECENT_F",
    "STABLE_NO_GT",
)


def evaluate_groups(
    *,
    y: np.ndarray,
    game_type: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline_name: str,
) -> pd.DataFrame:
    masks = {
        "ALL": np.ones(len(y), dtype=bool),
        "R": game_type == "R",
        "F": game_type == "F",
    }
    baseline = {
        group: regime_core.binary_metrics(y[mask], predictions[baseline_name][mask])
        for group, mask in masks.items()
    }
    rows: list[dict] = []
    for name, pred in predictions.items():
        for group, mask in masks.items():
            metric = regime_core.binary_metrics(y[mask], pred[mask])
            rows.append(
                {
                    "name": name,
                    "group": group,
                    "rows": int(mask.sum()),
                    "score": metric["score"],
                    "brier": metric["brier"],
                    "loss": metric["loss"],
                    "delta_brier_vs_baseline_same_group": (
                        metric["brier"] - baseline[group]["brier"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def print_group_table(df: pd.DataFrame, group: str) -> None:
    tqdm.write(f"\n[{group}]")
    subset = df.loc[df["group"].eq(group)].sort_values("brier")
    for _, row in subset.iterrows():
        metric = {
            "score": float(row["score"]),
            "brier": float(row["brier"]),
            "loss": float(row["loss"]),
        }
        tqdm.write(
            regime_core.metric_line(
                str(row["name"]),
                metric,
                float(row["delta_brier_vs_baseline_same_group"]),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate the observed 2023+ F regime into the existing dual-track design. "
            "Current recent expert stays 2023-only + game_type; stable expert stays full-history "
            "without game_type. A full-history game_type expert with/without recent_F is added "
            "to isolate whether recent_F can improve the dual-track blend."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--valid-year", type=int, default=2024)
    parser.add_argument("--regime-start-year", type=int, default=2023)
    parser.add_argument("--recent-iterations", type=int, default=400)
    parser.add_argument("--regime-iterations", type=int, default=400)
    parser.add_argument("--stable-iterations", type=int, default=300)
    parser.add_argument("--alpha-recent", type=float, default=0.425)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="2", help="Default 2 = third GPU")
    parser.add_argument("--gpu-ram-part", type=float, default=0.95)
    parser.add_argument("--pinned-memory-size", default="4GB")
    parser.add_argument(
        "--output-dir",
        default="outputs/dual_track_recent_f_integration",
    )
    args = parser.parse_args()

    try:
        import catboost
    except ImportError as exc:
        raise RuntimeError("catboost is required") from exc

    if not (0.0 <= args.alpha_recent <= 1.0):
        raise ValueError("--alpha-recent must be in [0,1]")
    if not (0.05 <= args.gpu_ram_part <= 1.0):
        raise ValueError("--gpu-ram-part must be in [0.05,1.0]")

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
        regime_start_year=args.regime_start_year,
    )

    full_train = frame.loc[frame[season_col].between(2019, valid_year - 1)].copy()
    recent_train = frame.loc[frame[season_col].eq(valid_year - 1)].copy()
    valid = frame.loc[frame[season_col].eq(valid_year)].copy()
    if full_train.empty or recent_train.empty or valid.empty:
        raise ValueError("train/validation split is empty")

    # Critical design check: in the current 2023-only recent expert, recent_F is
    # exactly the same information as F itself. Therefore adding recent_F directly
    # to that expert cannot encode the temporal break. The treatment must see old F
    # and recent F together, hence the full-history regime-aware expert below.
    recent_f_expected = recent_train["game_type"].eq("F").astype(np.float32).to_numpy()
    recent_f_actual = recent_train["eng_recent_f"].to_numpy(np.float32)
    recent_f_redundant_in_recent_track = bool(np.array_equal(recent_f_expected, recent_f_actual))
    if not recent_f_redundant_in_recent_track:
        raise RuntimeError(
            "Expected eng_recent_f to be redundant with game_type=F in the one-season recent track"
        )

    base_gt_features = recent_core.feature_set("recent_raw_game_type")
    stable_features = recent_core.feature_set("recent_drop_game_type")
    regime_features = [*base_gt_features, "eng_recent_f"]

    recent_params = regime_core.build_params(
        config=config,
        iterations=args.recent_iterations,
        task_type=args.task_type,
        devices=args.devices,
        gpu_ram_part=args.gpu_ram_part,
        pinned_memory_size=args.pinned_memory_size,
    )
    regime_params = regime_core.build_params(
        config=config,
        iterations=args.regime_iterations,
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

    tqdm.write(
        f"Dual-track recent_F integration | full_train=2019-{valid_year-1} ({len(full_train):,}) | "
        f"recent_train={valid_year-1} ({len(recent_train):,}) | valid={valid_year} ({len(valid):,}) | "
        f"alpha={args.alpha_recent:.3f} | GPU={args.devices if args.task_type == 'GPU' else 'CPU'} | "
        f"trees recent/regime/stable={args.recent_iterations}/{args.regime_iterations}/{args.stable_iterations} | "
        f"catboost={catboost.__version__}"
    )
    tqdm.write(
        "note: recent_F is redundant inside the one-season current recent expert; "
        "treatment uses full-history + game_type + recent_F."
    )

    expert_specs = {
        "CURRENT_RECENT_2023_GT": (recent_train, base_gt_features, recent_params),
        "FULL_GT": (full_train, base_gt_features, regime_params),
        "FULL_GT_RECENT_F": (full_train, regime_features, regime_params),
        "STABLE_NO_GT": (full_train, stable_features, stable_params),
    }

    expert_pred: dict[str, np.ndarray] = {}
    progress = tqdm(total=len(EXPERTS), desc="dual-track recent_F", unit="model", dynamic_ncols=True)
    for name in EXPERTS:
        train_part, features, params = expert_specs[name]
        seed_everything(seed)
        pred = regime_core.fit_predict(
            train=train_part,
            valid=valid,
            target_col=target_col,
            features=features,
            extra_categorical=set(),
            params=params,
        )
        expert_pred[name] = pred
        progress.update(1)
    progress.close()

    alpha = float(args.alpha_recent)
    p_stable = expert_pred["STABLE_NO_GT"]
    blend_pred = {
        "BASE_DUAL": alpha * expert_pred["CURRENT_RECENT_2023_GT"] + (1.0 - alpha) * p_stable,
        "FULL_GT_DUAL": alpha * expert_pred["FULL_GT"] + (1.0 - alpha) * p_stable,
        "RECENT_F_DUAL": alpha * expert_pred["FULL_GT_RECENT_F"] + (1.0 - alpha) * p_stable,
    }

    y_valid = pd.to_numeric(valid[target_col], errors="raise").to_numpy(np.float64)
    gt_valid = valid["game_type"].astype(str).to_numpy()

    expert_df = evaluate_groups(
        y=y_valid,
        game_type=gt_valid,
        predictions=expert_pred,
        baseline_name="CURRENT_RECENT_2023_GT",
    )
    blend_df = evaluate_groups(
        y=y_valid,
        game_type=gt_valid,
        predictions=blend_pred,
        baseline_name="BASE_DUAL",
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expert_df.to_csv(output_dir / "expert_metrics.csv", index=False)
    blend_df.to_csv(output_dir / "blend_metrics.csv", index=False)

    pred_frame = pd.DataFrame(
        {
            "target": y_valid,
            "game_type": gt_valid,
            **{f"expert_{k}": v for k, v in expert_pred.items()},
            **{f"blend_{k}": v for k, v in blend_pred.items()},
        }
    )
    if row_id_col in valid.columns:
        pred_frame.insert(0, row_id_col, valid[row_id_col].to_numpy())
    pred_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    save_json(
        {
            "experiment": "dual-track integration of explicit 2023+ F regime",
            "validation_year": valid_year,
            "full_train_seasons": list(range(2019, valid_year)),
            "current_recent_train_seasons": [valid_year - 1],
            "regime_start_year": int(args.regime_start_year),
            "alpha_recent": alpha,
            "recent_iterations": int(args.recent_iterations),
            "regime_iterations": int(args.regime_iterations),
            "stable_iterations": int(args.stable_iterations),
            "recent_f_redundant_in_current_recent_track": recent_f_redundant_in_recent_track,
            "experts": {
                "CURRENT_RECENT_2023_GT": "one-season recent expert + raw game_type (current dual-track)",
                "FULL_GT": "full-history pooled + raw game_type; control for training-window change",
                "FULL_GT_RECENT_F": "full-history pooled + raw game_type + 1[F and season>=2023]",
                "STABLE_NO_GT": "full-history stable expert with game_type removed",
            },
            "blends": {
                "BASE_DUAL": "alpha*CURRENT_RECENT_2023_GT + (1-alpha)*STABLE_NO_GT",
                "FULL_GT_DUAL": "alpha*FULL_GT + (1-alpha)*STABLE_NO_GT",
                "RECENT_F_DUAL": "alpha*FULL_GT_RECENT_F + (1-alpha)*STABLE_NO_GT",
            },
            "task_type": args.task_type,
            "devices": args.devices if args.task_type == "GPU" else None,
            "gpu_ram_part": float(args.gpu_ram_part) if args.task_type == "GPU" else None,
            "pinned_memory_size": args.pinned_memory_size if args.task_type == "GPU" else None,
            "catboost_version": catboost.__version__,
            "canonical_invariants": invariant_check,
        },
        output_dir / "run_config.json",
    )

    tqdm.write("\n[Experts | overall]")
    for _, row in expert_df.loc[expert_df["group"].eq("ALL")].sort_values("brier").iterrows():
        metric = {
            "score": float(row["score"]),
            "brier": float(row["brier"]),
            "loss": float(row["loss"]),
        }
        tqdm.write(
            regime_core.metric_line(
                str(row["name"]),
                metric,
                float(row["delta_brier_vs_baseline_same_group"]),
            )
        )

    print_group_table(blend_df, "ALL")
    print_group_table(blend_df, "R")
    print_group_table(blend_df, "F")
    tqdm.write(f"saved={output_dir}")


if __name__ == "__main__":
    main()
