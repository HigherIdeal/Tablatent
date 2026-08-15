from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_recent_regime_submissions as recent_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.utils import load_config, save_json, seed_everything


TRAIN_SEASONS = [2023, 2024]
VARIANTS = ["recent_raw_game_type", "recent_drop_game_type"]


def _season_diagnostics(train: pd.DataFrame, season_col: str, target: str) -> list[dict]:
    rows: list[dict] = []
    for year in TRAIN_SEASONS:
        group = train.loc[train[season_col].eq(year)]
        game_type = group["game_type"].astype(str)
        unexpected = sorted(set(game_type.unique()) - {"F", "R"})
        if unexpected:
            raise ValueError(f"Season {year}: unexpected game_type values {unexpected}")
        is_f = game_type.eq("F")
        y = pd.to_numeric(group[target], errors="raise")
        f_rate = float(y.loc[is_f].mean())
        r_rate = float(y.loc[~is_f].mean())
        rows.append(
            {
                "season": int(year),
                "rows": int(len(group)),
                "target_rate": float(y.mean()),
                "f_share": float(is_f.mean()),
                "f_target_rate": f_rate,
                "r_target_rate": r_rate,
                "f_minus_r_target_gap": f_rate - r_rate,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and persist the two 2023+2024 CatBoost regime probes. "
            "No validation and no submission packaging are performed here."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/recent_regime_models")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target = config["data"]["target_col"]
    season_col = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")

    frame, invariant_check = recent_core.prepare_frame(config)
    train = frame.loc[frame[season_col].isin(TRAIN_SEASONS)].copy()
    if train.empty:
        raise RuntimeError("No rows found for seasons 2023 and 2024")
    observed = sorted(train[season_col].unique().tolist())
    if observed != TRAIN_SEASONS:
        raise RuntimeError(f"Expected train seasons {TRAIN_SEASONS}, got {observed}")

    # The shared CatBoost policy uses has_time=True. Make the order explicit
    # instead of relying on whatever order the processed pickle happens to have.
    sort_columns = [season_col]
    if "game_month" in train.columns:
        sort_columns.append("game_month")
    if row_id in train.columns:
        sort_columns.append(row_id)
    train = train.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    diagnostics = _season_diagnostics(train, season_col, target)
    print(
        f"[Recent Regime Training] seasons={TRAIN_SEASONS}, rows={len(train):,}, "
        f"target_rate={pd.to_numeric(train[target], errors='raise').mean():.6f}, "
        f"iterations={args.iterations}, task_type={args.task_type}"
    )
    for row in diagnostics:
        print(
            f"  {row['season']}: rows={row['rows']:,} target={row['target_rate']:.6f} "
            f"F_share={row['f_share']:.6f} F_rate={row['f_target_rate']:.6f} "
            f"R_rate={row['r_target_rate']:.6f} F-R={row['f_minus_r_target_gap']:+.6f}"
        )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary: dict = {
        "train_seasons": TRAIN_SEASONS,
        "no_validation": True,
        "seed": seed,
        "iterations": int(args.iterations),
        "task_type": args.task_type,
        "devices": args.devices if args.task_type == "GPU" else None,
        "training_order": sort_columns,
        "season_diagnostics": diagnostics,
        "canonical_invariants": invariant_check,
        "variants": {},
    }

    for variant in VARIANTS:
        seed_everything(seed)
        features = recent_core.feature_set(variant)
        categorical = [feature for feature in features if feature in set(CANONICAL_CATEGORICAL)]
        model_path = output_dir / f"{variant}.cbm"
        metadata_path = output_dir / f"{variant}.json"

        print(
            f"\n[{variant}] TRAIN START rows={len(train):,} "
            f"features={len(features)} cat={len(categorical)} iterations={args.iterations}"
        )
        stats = recent_core.train_variant(
            train=train,
            target=target,
            features=features,
            config=config,
            iterations=args.iterations,
            task_type=args.task_type,
            devices=args.devices,
            verbose=args.verbose,
            model_path=model_path,
        )

        metadata = {
            "variant": variant,
            "train_seasons": TRAIN_SEASONS,
            "no_validation": True,
            "features": features,
            "categorical": categorical,
            "training_stats": stats,
            "season_diagnostics": diagnostics,
            "canonical_invariants": invariant_check,
            "purpose": "2025 hidden leaderboard probe for post-2023 regime continuity",
        }
        save_json(metadata, metadata_path)
        run_summary["variants"][variant] = {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            **stats,
        }
        print(
            f"[{variant}] TRAIN DONE model={model_path} "
            f"size={stats['model_bytes'] / (1024 * 1024):.2f} MiB"
        )

    save_json(run_summary, output_dir / "training_summary.json")
    (output_dir / "feature_sets.json").write_text(
        json.dumps(
            {variant: recent_core.feature_set(variant) for variant in VARIANTS},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n[Recent Regime Training] COMPLETE")
    print(f"  raw : {output_dir / 'recent_raw_game_type.cbm'}")
    print(f"  drop: {output_dir / 'recent_drop_game_type.cbm'}")
    print(f"  summary: {output_dir / 'training_summary.json'}")
    print("Models are persisted; packaging can be done later without retraining.")


if __name__ == "__main__":
    main()
