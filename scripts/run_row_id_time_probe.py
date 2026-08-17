from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_game_type_temporal_regime_ablation as metric_core
import run_regime_feature_prediction_suite as regime_core
from src.utils import load_config


def add_row_time(frame: pd.DataFrame) -> None:
    raw = pd.to_numeric(frame.row_id.astype(str).str.rsplit("_", n=1).str[-1])
    start = raw.groupby(frame.season).transform("min")
    frame["row_time"] = (raw - start) / 250_000.0
    frame["row_time20"] = np.floor(frame.row_time * 20).clip(0, 19).astype("string")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=2024); ap.add_argument("--devices", default="0"); args = ap.parse_args()
    from catboost import CatBoostClassifier, Pool
    frame, _ = recent_core.prepare_frame(load_config(ROOT / "configs/default.yaml")); frame.season = pd.to_numeric(frame.season).astype(int); frame.game_type = frame.game_type.astype("string").str.upper(); add_row_time(frame); context_core.add_context_interactions(frame)
    paths = path_core.add_paths(frame, "pitcher_id", "season", "control_success") + path_core.add_paths(frame, "batter_id", "season", "control_success"); paths = [x for x in paths if not x.endswith("_rate")]
    regime_core.EXTRA_CATEGORICAL.update(["row_time20", *(x for x in paths if x.endswith(("last_gt", "current_x_last")))])
    tr, va = frame[frame.season < args.fold].copy(), frame[frame.season == args.fold].copy(); regime_core.add_regime_features(tr, va, season_col="season", recent_start=2023)
    base = [*recent_core.feature_set("recent_raw_game_type"), regime_core.RECENT_FLAG, *regime_core.FAST_CONT, *regime_core.RANGE_CONT, *context_core.INTERACTION_COLUMNS, *paths]
    y=va.control_success.to_numpy(float); out=ROOT/f"outputs/row_id_time_f{args.fold}";out.mkdir(parents=True,exist_ok=True); saved={"y":y,"gt":va.game_type.astype(str).to_numpy()}
    for name,extra in (("base",[]),("time",["row_time"]),("time20",["row_time","row_time20"])):
        feats=base+extra;x,c=regime_core.prepare_x(tr,feats);v,_=regime_core.prepare_x(va,feats);m=CatBoostClassifier(iterations=600,learning_rate=.03,depth=8,loss_function="Logloss",l2_leaf_reg=20,random_strength=.5,bootstrap_type="Bayesian",bagging_temperature=.5,border_count=128,random_seed=42,has_time=True,one_hot_max_size=10,allow_writing_files=False,task_type="GPU",devices=args.devices,verbose=False).fit(Pool(x,tr.control_success,cat_features=c,feature_names=feats));p=m.predict_proba(Pool(v,cat_features=c,feature_names=feats),ntree_end=400)[:,1];saved[name]=p;q=metric_core.binary_metrics(y,p);print(f"{name}: s={q['score']:.1f} b={q['brier']:.3e}")
    np.savez_compressed(out/"predictions.npz",**saved)


if __name__ == "__main__": main()
