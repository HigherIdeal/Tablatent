from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import build_recent_regime_submissions as recent_core
import run_asof_prefix_inversion_probe as prefix_core
import run_game_type_temporal_regime_ablation as regime_core
from run_multitask_outcome_boosting import auxiliary_targets
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--output-dir", default="outputs/prefix_joint_outcome")
    a = ap.parse_args()

    cfg = load_config(ROOT / "configs/default.yaml")
    frame, _ = recent_core.prepare_frame(cfg)
    frame["season"] = pd.to_numeric(frame.season, errors="raise").astype(int)
    frame["game_type"] = frame.game_type.astype("string").str.upper()
    prefix_core.add_prefix_inversion_features(frame, pitcher_col="pitcher_id", n_col="asof_pitcher_n", count_tolerance=.05)
    regime_core.add_regime_features(frame, season_col="season", regime_start_year=2023)
    features = prefix_core.feature_sets(recent_core.feature_set("recent_raw_game_type"))["A4_MULTI_SCALE_STATE"]

    train, valid = frame[frame.season < a.fold].copy(), frame[frame.season == a.fold].copy()
    aux = auxiliary_targets(train)[["reverse", "middle", "ball", "strike"]]
    keep = aux.notna().all(axis=1).to_numpy()
    bits = aux.loc[keep].to_numpy(np.int8)
    codes = bits @ (1 << np.arange(4, dtype=np.int16))
    classes = np.unique(codes)
    labels = np.searchsorted(classes, codes)
    xt, cats = regime_core.prepare_x(train.loc[keep], features)
    xv, _ = regime_core.prepare_x(valid, features)
    model = CatBoostClassifier(iterations=a.iterations, depth=a.depth, learning_rate=.03,
        loss_function="MultiClass", l2_leaf_reg=20, random_strength=.5,
        bootstrap_type="Bayesian", bagging_temperature=.5, border_count=128,
        has_time=True, one_hot_max_size=10, allow_writing_files=False,
        task_type="GPU", devices=a.devices, verbose=False, random_seed=42)
    model.fit(Pool(xt, labels, cat_features=cats, feature_names=features))
    probs = model.predict_proba(Pool(xv, cat_features=cats, feature_names=features))

    target = train.loc[keep, "control_success"].to_numpy(float)
    gt = train.loc[keep, "game_type"].astype(str).to_numpy()
    sy = train.loc[keep, "season"].to_numpy()
    q = np.zeros((2, len(classes)))
    for gi, domain in enumerate(("R", "F")):
        dm = gt == domain
        if domain == "F" and np.any(dm & (sy >= 2023)):
            dm &= sy >= 2023
        for ci in range(len(classes)):
            cm = dm & (labels == ci)
            q[gi, ci] = target[cm].mean() if cm.any() else target[dm].mean()
    vg = (valid.game_type.astype(str).to_numpy() == "F").astype(int)
    pred = np.sum(probs * q[vg], axis=1)
    y = valid.control_success.to_numpy(float)
    brier = np.mean((pred-y)**2); score=1e5*(1-brier/.2498069)
    out=ROOT/a.output_dir; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"predictions.npz", y=y, gt=valid.game_type.astype(str).to_numpy(), pred=pred, probs=probs, classes=classes, q=q)
    model.save_model(str(out/"model.cbm"))
    print(f"s={score:.1f} b={brier:.3e}")


if __name__ == "__main__": main()
