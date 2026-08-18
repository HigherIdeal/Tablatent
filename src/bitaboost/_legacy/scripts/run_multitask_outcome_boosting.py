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
import run_asof_prefix_inversion_probe as inversion_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_frozen_season_anchor_probe as anchor_core
import run_game_type_temporal_regime_ablation as metric_core
import run_regime_feature_prediction_suite as regime_core
import run_offset_residual_boosting as offset_core
from src.utils import load_config


def auxiliary_targets(train: pd.DataFrame) -> pd.DataFrame:
    work = train[["pitcher_id", "asof_pitcher_n", "control_success", *inversion_core.PREFIX_RATES.values()]].copy()
    work["_index"] = np.arange(len(work))
    work = work.sort_values(["pitcher_id", "asof_pitcher_n", "_index"], kind="stable").reset_index(drop=True)
    same = work.pitcher_id.eq(work.pitcher_id.shift(-1))
    contiguous = work.asof_pitcher_n.shift(-1).eq(work.asof_pitcher_n + 1)
    valid_transition = (same & contiguous).to_numpy()
    out = pd.DataFrame(index=np.arange(len(train)))
    n = work.asof_pitcher_n.to_numpy(float)
    for short, column in inversion_core.PREFIX_RATES.items():
        count = np.rint(n * pd.to_numeric(work[column], errors="coerce").to_numpy(float))
        delta = np.roll(count, -1) - count
        valid = valid_transition & np.isfinite(delta) & ((delta == 0) | (delta == 1))
        values = np.full(len(work), np.nan, np.float32)
        values[valid] = delta[valid]
        restored = np.full(len(train), np.nan, np.float32)
        restored[work._index.to_numpy()] = values
        out[short] = restored
    previous = work.groupby("pitcher_id", sort=False)["control_success"].shift(1)
    for window in (1, 3, 5):
        values = previous.groupby(work["pitcher_id"], sort=False).rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
        restored = np.full(len(train), np.nan, np.float32)
        restored[work._index.to_numpy()] = values.to_numpy(np.float32)
        out[f"prev{window}_success"] = restored
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--success-repeats", type=int, default=1)
    ap.add_argument("--aux", default="reverse,middle,ball,strike")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f-weight", type=float, default=1.0)
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--batter-anchor", action="store_true")
    ap.add_argument("--multi-anchor", action="store_true")
    ap.add_argument("--anchor-cross", action="store_true")
    ap.add_argument("--matchup", action="store_true")
    ap.add_argument("--count-profile", action="store_true")
    ap.add_argument("--pressure-profile", action="store_true")
    ap.add_argument("--opponent-profile", action="store_true")
    ap.add_argument("--domain-profile", action="store_true")
    ap.add_argument("--pair-profile", action="store_true")
    ap.add_argument("--leverage-profile", action="store_true")
    ap.add_argument("--anchor-form", action="store_true")
    ap.add_argument("--arsenal-context", action="store_true")
    ap.add_argument("--aux-profile", action="store_true")
    ap.add_argument("--aux-profile-extra", choices=["none", "gt", "pressure", "inning", "both"], default="both")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--joint-class", action="store_true")
    ap.add_argument("--joint-success", action="store_true")
    ap.add_argument("--season-basis", action="store_true")
    ap.add_argument("--context-lattice", action="store_true")
    args = ap.parse_args()
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    config = load_config(ROOT / "configs/default.yaml")
    target, season = config["data"]["target_col"], config["data"]["season_col"]
    frame, _ = recent_core.prepare_frame(config)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame.game_type = frame.game_type.astype("string").str.upper()
    if args.anchor:
        anchor_core.add_frozen_anchor_features(frame, season_col=season, pitcher_col="pitcher_id", n_col="asof_pitcher_n", count_tolerance=.05)
    banchor = offset_core.add_batter_anchor(frame) if args.batter_anchor else []
    across = offset_core.add_anchor_cross(frame) if args.anchor_cross else []
    season_basis = offset_core.add_rowlocal_season_basis(frame) if args.season_basis else []
    context_lattice = offset_core.add_frozen_context_lattice(frame) if args.context_lattice else []
    matchup = offset_core.add_frozen_matchup(frame) if args.matchup else []
    count_profile = offset_core.add_frozen_count_profiles(frame) if args.count_profile else []
    pressure_profile = offset_core.add_frozen_pressure_profiles(frame) if args.pressure_profile else []
    opponent_profile = offset_core.add_frozen_opponent_profiles(frame) if args.opponent_profile else []
    domain_profile = offset_core.add_frozen_domain_profiles(frame) if args.domain_profile else []
    pair_profile = offset_core.add_frozen_pair_profile(frame) if args.pair_profile else []
    leverage_profile = offset_core.add_frozen_leverage_profiles(frame) if args.leverage_profile else []
    anchor_form = offset_core.add_anchor_form(frame) if args.anchor_form else []
    arsenal_context = offset_core.add_arsenal_context(frame) if args.arsenal_context else []
    aux_profile = offset_core.add_frozen_aux_profiles(frame, auxiliary_targets(frame), args.aux_profile_extra) if args.aux_profile else []
    regime_core.EXTRA_CATEGORICAL.update(arsenal_context)
    context_core.add_context_interactions(frame)
    paths = path_core.add_paths(frame, "pitcher_id", season, target) + path_core.add_paths(frame, "batter_id", season, target)
    paths = [x for x in paths if not x.endswith("_rate")]
    path_cats = {x for x in paths if x.endswith(("last_gt", "current_x_last"))}
    regime_core.EXTRA_CATEGORICAL.update(path_cats)
    train, valid = frame[frame[season] < args.fold].copy(), frame[frame[season] == args.fold].copy()
    regime_core.add_regime_features(train, valid, season_col=season, recent_start=2023)
    features = [
        *recent_core.feature_set("recent_raw_game_type"), regime_core.RECENT_FLAG,
        *regime_core.FAST_CONT, *regime_core.RANGE_CONT,
        *context_core.INTERACTION_COLUMNS, *paths,
    ]
    if args.anchor:
        features += ["eng_anchor_available", "eng_anchor_gap_n", "eng_anchor_success_rate", "eng_since_anchor_success_rate", "eng_since_anchor_success_minus_long"]
    if args.multi_anchor:
        features += [x for short in ("reverse", "middle", "ball", "strike") for x in (f"eng_anchor_{short}_rate", f"eng_since_anchor_{short}_rate", f"eng_since_anchor_{short}_minus_long")]
    features += banchor
    features += across
    features += season_basis
    features += context_lattice
    features += matchup
    features += count_profile
    features += pressure_profile
    features += opponent_profile
    features += domain_profile
    features += pair_profile
    features += leverage_profile
    features += anchor_form
    features += arsenal_context
    features += aux_profile
    aux = auxiliary_targets(train)
    aux_names = [x.strip() for x in args.aux.split(",") if x.strip()]
    keep = aux[aux_names].notna().all(axis=1).to_numpy()
    success = train.loc[keep, target].to_numpy(np.float32)
    aux_values = aux.loc[keep, aux_names].to_numpy(np.int8)
    labels = np.column_stack([*[success] * args.success_repeats, aux_values.astype(np.float32)])
    x_train, cats = regime_core.prepare_x(train.loc[keep], features)
    x_valid, _ = regime_core.prepare_x(valid, features)
    weights = np.where(train.loc[keep, "game_type"].astype(str).to_numpy() == "F", args.f_weight, 1.0).astype(np.float32)
    if args.joint_class:
        codes = aux_values @ (1 << np.arange(len(aux_names), dtype=np.int16))
        if args.joint_success:
            codes = codes + success.astype(np.int16) * (1 << len(aux_names))
        classes = np.unique(codes)
        class_index = np.searchsorted(classes, codes)
        pool = Pool(x_train, class_index, weight=weights, cat_features=cats, feature_names=features)
    else:
        pool = Pool(x_train, labels, weight=weights, cat_features=cats, feature_names=features)
    vpool = Pool(x_valid, cat_features=cats, feature_names=features)
    model_cls = CatBoostClassifier if args.joint_class else CatBoostRegressor
    model = model_cls(
        iterations=args.iterations, learning_rate=.03, depth=args.depth,
        loss_function="MultiClass" if args.joint_class else "MultiRMSE",
        l2_leaf_reg=20, random_strength=.5, bootstrap_type="Bayesian",
        bagging_temperature=.5, border_count=128, has_time=True,
        one_hot_max_size=10, allow_writing_files=False, task_type="GPU", devices="0", verbose=False,
        random_seed=args.seed,
    )
    model.fit(pool)
    y = valid[target].to_numpy(float)
    out = ROOT / f"outputs/{'joint_success' if args.joint_success else 'joint_outcome' if args.joint_class else 'multitask_outcome_boosting'}_d{args.depth}_r{args.success_repeats}_{'-'.join(aux_names)}_f{args.fold}_s{args.seed}_fw{args.f_weight:g}{'_anchor' if args.anchor else ''}{'_multi' if args.multi_anchor else ''}{'_banchor' if args.batter_anchor else ''}{'_cross4' if args.anchor_cross else ''}{'_sbasis' if args.season_basis else ''}{'_lattice' if args.context_lattice else ''}{'_match' if args.matchup else ''}{'_count' if args.count_profile else ''}{'_pressure' if args.pressure_profile else ''}{'_opp' if args.opponent_profile else ''}{'_domain' if args.domain_profile else ''}{'_pair' if args.pair_profile else ''}{'_lev' if args.leverage_profile else ''}{'_form' if args.anchor_form else ''}{'_arsenal' if args.arsenal_context else ''}{f'_auxprof_{args.aux_profile_extra}' if args.aux_profile else ''}_i{args.iterations}"
    out.mkdir(parents=True, exist_ok=True)
    rows, saved = {"y": y, "gt": valid.game_type.astype(str).to_numpy()}.copy(), None
    metric_rows = []
    for trees in sorted({200, 400, 600, 800, args.iterations}):
        if trees > args.iterations:
            continue
        all_p = np.clip(model.predict_proba(vpool, ntree_end=trees), 0, 1) if args.joint_class else np.clip(model.predict(vpool, ntree_end=trees), 0, 1)
        if args.joint_class:
            if args.joint_success:
                p = all_p[:, classes >= (1 << len(aux_names))].sum(axis=1)
            else:
                q = np.zeros((2, len(classes)), dtype=float)
                gt_train = train.loc[keep, "game_type"].astype(str).to_numpy()
                season_train = train.loc[keep, season].to_numpy()
                for gi, g in enumerate(("R", "F")):
                    gm = gt_train == g
                    if g == "F" and np.any(gm & (season_train >= 2023)):
                        gm &= season_train >= 2023
                    for ci in range(len(classes)):
                        cm = gm & (class_index == ci)
                        q[gi, ci] = success[cm].mean() if cm.any() else success[gm].mean()
                vg = (valid.game_type.astype(str).to_numpy() == "F").astype(int)
                p = np.sum(all_p * q[vg], axis=1)
        else:
            p = all_p[:, 0]
        m = metric_core.binary_metrics(y, p)
        metric_rows.append({"trees": trees, **m}); rows[f"{'joint' if args.joint_class else 'all'}_t{trees}"] = all_p
        if args.joint_class:
            rows[f"pred_t{trees}"] = p
        print(f"t{trees}: s={m['score']:.1f} b={m['brier']:.3e}")
    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    np.savez_compressed(out / "predictions.npz", **rows)


if __name__ == "__main__":
    main()
