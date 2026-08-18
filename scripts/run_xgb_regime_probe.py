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
import run_game_type_temporal_regime_ablation as metric_core
import run_regime_feature_prediction_suite as regime_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.utils import load_config


def encode(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = train[features].copy(), valid[features].copy()
    categorical = set(CANONICAL_CATEGORICAL) | set(context_core.INTERACTION_COLUMNS)
    for column in features:
        if column in categorical:
            categories = pd.Index(a[column].astype("string").fillna("<MISSING>").unique())
            a[column] = pd.Categorical(a[column].astype("string").fillna("<MISSING>"), categories=categories)
            b[column] = pd.Categorical(b[column].astype("string").fillna("<MISSING>"), categories=categories)
        else:
            a[column] = pd.to_numeric(a[column], errors="coerce").astype(np.float32)
            b[column] = pd.to_numeric(b[column], errors="coerce").astype(np.float32)
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--trees", default="300,500,800,1200")
    ap.add_argument("--output-dir", default="outputs/xgb_regime_probe")
    args = ap.parse_args()

    import xgboost as xgb

    config = load_config(ROOT / args.config)
    target, season = config["data"]["target_col"], config["data"]["season_col"]
    frame, _ = recent_core.prepare_frame(config)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.upper()
    context_core.add_context_interactions(frame)
    train, valid = frame[frame[season] < 2024].copy(), frame[frame[season] == 2024].copy()
    regime_core.add_regime_features(train, valid, season_col=season, recent_start=2023)
    features = [
        *recent_core.feature_set("recent_raw_game_type"), regime_core.RECENT_FLAG,
        *regime_core.FAST_CONT, *regime_core.RANGE_CONT, *context_core.INTERACTION_COLUMNS,
    ]
    x_train, x_valid = encode(train, valid, features)
    y_train = train[target].to_numpy(np.float32)
    y_valid = valid[target].to_numpy(np.float64)
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, enable_categorical=True, max_bin=256)
    dvalid = xgb.QuantileDMatrix(x_valid, ref=dtrain, enable_categorical=True, max_bin=256)
    trees = sorted({int(x) for x in args.trees.split(",")})
    model = xgb.train(
        {
            "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "device": "cuda", "eta": 0.03, "max_depth": args.max_depth,
            "min_child_weight": 50, "subsample": 0.85, "colsample_bytree": 0.85,
            "lambda": 20, "alpha": 0.2, "max_bin": 256, "seed": 42,
        },
        dtrain,
        num_boost_round=max(trees),
        verbose_eval=False,
    )
    rows = []
    for n in trees:
        p = model.predict(dvalid, iteration_range=(0, n))
        m = metric_core.binary_metrics(y_valid, p)
        rows.append({"depth": args.max_depth, "trees": n, **m})
        print(f"d{args.max_depth} t{n}: s={m['score']:.1f} b={m['brier']:.3e}")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"depth_{args.max_depth}.csv", index=False)
    np.savez_compressed(out / f"depth_{args.max_depth}_pred.npz", y=y_valid, probability=model.predict(dvalid, iteration_range=(0, min(rows, key=lambda r:r['brier'])["trees"])))


if __name__ == "__main__":
    main()
