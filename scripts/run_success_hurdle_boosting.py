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
import run_frozen_season_anchor_probe as anchor_core
import run_game_type_temporal_regime_ablation as metric_core
import run_multitask_outcome_boosting as multi_core
import run_regime_feature_prediction_suite as regime_core
import run_offset_residual_boosting as offset_core
from src.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--batter-anchor", action="store_true")
    ap.add_argument("--anchor-cross", action="store_true")
    ap.add_argument("--matchup", action="store_true")
    ap.add_argument("--count-profile", action="store_true")
    ap.add_argument("--pressure-profile", action="store_true")
    ap.add_argument("--domain-profile", action="store_true")
    ap.add_argument("--aux-profile", action="store_true")
    ap.add_argument("--conditional-profile", action="store_true")
    ap.add_argument("--conditional-batter", action="store_true")
    ap.add_argument("--conditional-loss", choices=["logloss", "brier"], default="logloss")
    ap.add_argument("--f-weight", type=float, default=1.0)
    ap.add_argument("--gate-loss", choices=["logloss", "brier"], default="logloss")
    args = ap.parse_args()
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    frame, _ = recent_core.prepare_frame(load_config(ROOT / "configs/default.yaml"))
    frame.season = pd.to_numeric(frame.season).astype(int)
    frame.game_type = frame.game_type.astype("string").str.upper()
    if args.anchor:
        anchor_core.add_frozen_anchor_features(frame, season_col="season", pitcher_col="pitcher_id", n_col="asof_pitcher_n", count_tolerance=.05)
    banchor = offset_core.add_batter_anchor(frame) if args.batter_anchor else []
    across = offset_core.add_anchor_cross(frame) if args.anchor_cross else []
    matchup = offset_core.add_frozen_matchup(frame) if args.matchup else []
    count_profile = offset_core.add_frozen_count_profiles(frame) if args.count_profile else []
    pressure_profile = offset_core.add_frozen_pressure_profiles(frame) if args.pressure_profile else []
    domain_profile = offset_core.add_frozen_domain_profiles(frame) if args.domain_profile else []
    aux_profile = offset_core.add_frozen_aux_profiles(frame, multi_core.auxiliary_targets(frame), "pressure") if args.aux_profile else []
    conditional_profile = offset_core.add_frozen_conditional_profiles(frame, multi_core.auxiliary_targets(frame), args.conditional_batter) if args.conditional_profile else []
    context_core.add_context_interactions(frame)
    paths = path_core.add_paths(frame, "pitcher_id", "season", "control_success") + path_core.add_paths(frame, "batter_id", "season", "control_success")
    paths = [x for x in paths if not x.endswith("_rate")]
    regime_core.EXTRA_CATEGORICAL.update(x for x in paths if x.endswith(("last_gt", "current_x_last")))
    train, valid = frame[frame.season < args.fold].copy(), frame[frame.season == args.fold].copy()
    regime_core.add_regime_features(train, valid, season_col="season", recent_start=2023)
    features = [*recent_core.feature_set("recent_raw_game_type"), regime_core.RECENT_FLAG, *regime_core.FAST_CONT, *regime_core.RANGE_CONT, *context_core.INTERACTION_COLUMNS, *paths]
    if args.anchor:
        features += ["eng_anchor_available", "eng_anchor_gap_n", "eng_anchor_success_rate", "eng_since_anchor_success_rate", "eng_since_anchor_success_minus_long"]
    features += banchor
    features += across
    features += matchup
    features += count_profile
    features += pressure_profile + domain_profile + aux_profile + conditional_profile
    aux = multi_core.auxiliary_targets(train)
    keep = aux[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    gate = ((aux.reverse == 0) & (aux.middle == 0)).to_numpy()
    x, cats = regime_core.prepare_x(train.loc[keep], features)
    xv, _ = regime_core.prepare_x(valid, features)
    vp = Pool(xv, cat_features=cats, feature_names=features)
    common = dict(iterations=600, learning_rate=.03, depth=8, loss_function="Logloss", l2_leaf_reg=20, random_strength=.5, bootstrap_type="Bayesian", bagging_temperature=.5, border_count=128, random_seed=args.seed, has_time=True, one_hot_max_size=10, allow_writing_files=False, task_type="GPU", devices=args.devices, verbose=False)
    gw = np.where(train.loc[keep, "game_type"].astype(str).to_numpy() == "F", args.f_weight, 1.0)
    if args.gate_loss == "logloss":
        g = CatBoostClassifier(**common).fit(Pool(x, gate[keep].astype(np.int8), weight=gw, cat_features=cats, feature_names=features))
    else:
        gp = {**common, "loss_function": "RMSE"}; g = CatBoostRegressor(**gp).fit(Pool(x, gate[keep].astype(np.float32), weight=gw, cat_features=cats, feature_names=features))
    cond = keep & gate
    xc, cc = regime_core.prepare_x(train.loc[cond], features)
    if args.conditional_loss == "logloss":
        cw = np.where(train.loc[cond, "game_type"].astype(str).to_numpy() == "F", args.f_weight, 1.0)
        h = CatBoostClassifier(**common).fit(Pool(xc, train.loc[cond, "control_success"].astype(np.int8), weight=cw, cat_features=cc, feature_names=features))
    else:
        cw = np.where(train.loc[cond, "game_type"].astype(str).to_numpy() == "F", args.f_weight, 1.0)
        hp = {**common, "loss_function": "RMSE"}; h = CatBoostRegressor(**hp).fit(Pool(xc, train.loc[cond, "control_success"].astype(np.float32), weight=cw, cat_features=cc, feature_names=features))
    y = valid.control_success.to_numpy(float)
    out = ROOT / f"outputs/success_hurdle_f{args.fold}_s{args.seed}{'_anchor' if args.anchor else ''}{'_banchor' if args.batter_anchor else ''}{'_cross' if args.anchor_cross else ''}{'_match' if args.matchup else ''}{'_count' if args.count_profile else ''}{'_pressure' if args.pressure_profile else ''}{'_domain' if args.domain_profile else ''}{'_auxprof' if args.aux_profile else ''}{'_condprof' if args.conditional_profile else ''}{'_cbatter' if args.conditional_batter else ''}_{args.gate_loss}_{args.conditional_loss}_fw{args.f_weight:g}"; out.mkdir(parents=True, exist_ok=True)
    saved = {"y": y, "gt": valid.game_type.astype(str).to_numpy()}
    for t in (200, 400, 600):
        gp = g.predict_proba(vp, ntree_end=t)[:, 1] if args.gate_loss == "logloss" else np.clip(g.predict(vp, ntree_end=t), 0, 1); hp = h.predict_proba(vp, ntree_end=t)[:, 1] if args.conditional_loss == "logloss" else np.clip(h.predict(vp, ntree_end=t), 0, 1); p = gp * hp
        saved[f"t{t}"] = p; saved[f"gate{t}"] = gp; saved[f"cond{t}"] = hp; m = metric_core.binary_metrics(y, p)
        print(f"t{t}: s={m['score']:.1f} b={m['brier']:.3e}")
    np.savez_compressed(out / "predictions.npz", **saved)


if __name__ == "__main__":
    main()
