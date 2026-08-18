from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_game_type_temporal_regime_ablation as metric_core
import run_regime_feature_prediction_suite as regime_core
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", default="200,400,600,800")
    ap.add_argument("--devices", default="0")
    ap.add_argument("--output-dir", default="outputs/direct_brier_boosting")
    args = ap.parse_args()
    from catboost import CatBoostRegressor, Pool

    config = load_config(ROOT / "configs/default.yaml")
    target, season = config["data"]["target_col"], config["data"]["season_col"]
    frame, _ = recent_core.prepare_frame(config)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.upper()
    context_core.add_context_interactions(frame)
    paths = path_core.add_paths(frame, "pitcher_id", season, target) + path_core.add_paths(frame, "batter_id", season, target)
    paths = [x for x in paths if not x.endswith("_rate")]
    path_cats = {x for x in paths if x.endswith(("last_gt", "current_x_last"))}
    regime_core.EXTRA_CATEGORICAL.update(path_cats)
    train, valid = frame[frame[season] < 2024].copy(), frame[frame[season] == 2024].copy()
    regime_core.add_regime_features(train, valid, season_col=season, recent_start=2023)
    features = [
        *recent_core.feature_set("recent_raw_game_type"), regime_core.RECENT_FLAG,
        *regime_core.FAST_CONT, *regime_core.RANGE_CONT,
        *context_core.INTERACTION_COLUMNS, *paths,
    ]
    x_train, cats = regime_core.prepare_x(train, features)
    x_valid, _ = regime_core.prepare_x(valid, features)
    y_train, y_valid = train[target].to_numpy(np.float32), valid[target].to_numpy(np.float64)
    train_pool = Pool(x_train, y_train, cat_features=cats, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=cats, feature_names=features)
    trees = sorted({int(x) for x in args.trees.split(",")})
    model = CatBoostRegressor(
        iterations=max(trees), learning_rate=.03, depth=8, loss_function="RMSE",
        l2_leaf_reg=20, random_strength=.5, bootstrap_type="Bayesian",
        bagging_temperature=.5, border_count=128, random_seed=42, has_time=True,
        one_hot_max_size=10, allow_writing_files=False, task_type="GPU",
        devices=args.devices, verbose=False,
    )
    model.fit(train_pool)
    rows = []
    predictions = {}
    for n in trees:
        p = np.clip(model.predict(valid_pool, ntree_end=n), 0, 1)
        m = metric_core.binary_metrics(y_valid, p)
        rows.append({"trees": n, **m})
        predictions[f"t{n}"] = p
        print(f"t{n}: s={m['score']:.1f} b={m['brier']:.3e}")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(out / "predictions.npz", y=y_valid, **predictions)


if __name__ == "__main__":
    main()
