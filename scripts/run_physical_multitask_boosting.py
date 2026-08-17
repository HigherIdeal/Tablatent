from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (ROOT, SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_game_type_temporal_regime_ablation as metric_core
import run_multitask_outcome_boosting as multi_core
import run_physical_regime_suite as physical_core
import run_regime_feature_prediction_suite as regime_core
import run_frozen_season_anchor_probe as anchor_core
import run_offset_residual_boosting as offset_core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prev", "current"], required=True)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--rich", action="store_true")
    args = ap.parse_args()
    from catboost import CatBoostRegressor, Pool

    frame = pd.read_parquet(physical_core.DEFAULT_DATA)
    physical_core.add_derived_features(frame)
    frame.season = pd.to_numeric(frame.season, errors="raise").astype(int)
    frame.game_type = frame.game_type.astype("string").str.upper()
    rich = []
    if args.rich:
        anchor_core.add_frozen_anchor_features(frame, season_col="season", pitcher_col="pitcher_id", n_col="asof_pitcher_n", count_tolerance=.05)
        rich += ["eng_anchor_available", "eng_anchor_gap_n", *[x for short in ("success", "reverse", "middle", "ball", "strike") for x in (f"eng_anchor_{short}_rate", f"eng_since_anchor_{short}_rate", f"eng_since_anchor_{short}_minus_long")]]
        rich += offset_core.add_batter_anchor(frame)
        rich += offset_core.add_anchor_cross(frame)
        rich += offset_core.add_frozen_matchup(frame)
        rich += offset_core.add_frozen_count_profiles(frame)
        rich += offset_core.add_frozen_pressure_profiles(frame)
        rich += offset_core.add_frozen_domain_profiles(frame)
    context_core.add_context_interactions(frame)
    paths = path_core.add_paths(frame, "pitcher_id", "season", "control_success") + path_core.add_paths(frame, "batter_id", "season", "control_success")
    paths = [x for x in paths if not x.endswith("_rate")]
    path_cats = {x for x in paths if x.endswith(("last_gt", "current_x_last"))}
    regime_core.EXTRA_CATEGORICAL.update(path_cats | {"tm_profile_source"})
    train, valid = frame[frame.season < 2024].copy(), frame[frame.season == 2024].copy()
    regime_core.add_regime_features(train, valid, season_col="season", recent_start=2023)
    features = [*physical_core.PHYSICAL_FEATURES, regime_core.RECENT_FLAG, *regime_core.FAST_CONT, *regime_core.RANGE_CONT, *context_core.INTERACTION_COLUMNS, *paths, *rich]
    aux = multi_core.auxiliary_targets(train)
    if args.mode == "prev":
        aux_names, repeats = ["prev1_success", "prev3_success", "prev5_success"], 6
    else:
        aux_names, repeats = ["reverse", "middle", "ball", "strike"], 8
    keep = aux[aux_names].notna().all(axis=1).to_numpy()
    success = train.loc[keep, "control_success"].to_numpy(np.float32)
    labels = np.column_stack([*[success] * repeats, aux.loc[keep, aux_names].to_numpy(np.float32)])
    x_train, cats = regime_core.prepare_x(train.loc[keep], features)
    x_valid, _ = regime_core.prepare_x(valid, features)
    weights = np.where(train.loc[keep, "game_type"].astype(str).to_numpy() == "F", 2.0 if args.mode == "current" else 1.0, 1.0).astype(np.float32)
    pool = Pool(x_train, labels, weight=weights, cat_features=cats, feature_names=features)
    vpool = Pool(x_valid, cat_features=cats, feature_names=features)
    model = CatBoostRegressor(iterations=600, learning_rate=.03, depth=8, loss_function="MultiRMSE", l2_leaf_reg=20, random_strength=.5, bootstrap_type="Bayesian", bagging_temperature=.5, border_count=128, random_seed=42, has_time=True, one_hot_max_size=10, allow_writing_files=False, task_type="GPU", devices=args.devices, verbose=False)
    model.fit(pool)
    y = valid.control_success.to_numpy(float); out = ROOT / f"outputs/physical_multitask_{args.mode}{'_rich' if args.rich else ''}"; out.mkdir(parents=True, exist_ok=True)
    saved = {"y": y, "gt": valid.game_type.astype(str).to_numpy()}
    for trees in (200, 400, 600):
        p = np.clip(model.predict(vpool, ntree_end=trees)[:, 0], 0, 1); saved[f"t{trees}"] = p
        m = metric_core.binary_metrics(y, p); print(f"t{trees}: s={m['score']:.1f} b={m['brier']:.3e}")
    np.savez_compressed(out / "predictions.npz", **saved)


if __name__ == "__main__":
    main()
