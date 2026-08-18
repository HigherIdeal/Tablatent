from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _p in (ROOT, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# A local experimental helper was used by the latest branch but is not present in
# commit 1e0f3ea on GitHub. Keep the builder usable on a clean clone by installing
# a no-op fallback. If the local helper exists, it is used and bundled instead.
PATH_HELPER = SCRIPTS / "run_frozen_domain_path_probe.py"
HAS_LOCAL_PATH_HELPER = PATH_HELPER.is_file()
if not HAS_LOCAL_PATH_HELPER:
    _dummy = types.ModuleType("run_frozen_domain_path_probe")
    _dummy.add_paths = lambda frame, player_col, season_col, target_col: []
    sys.modules["run_frozen_domain_path_probe"] = _dummy

import build_recent_regime_submissions as recent_core
import run_asof_state_engineering as asof_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_frozen_season_anchor_probe as anchor_core
import run_multitask_outcome_boosting as multi_core
import run_offset_residual_boosting as offset_core
import run_regime_feature_prediction_suite as regime_core
from src.canonical_features import add_canonical_derived_features
from src.utils import load_config


AUX_NAMES = ["reverse", "middle", "ball", "strike"]
BASE_TREES = 600
SUCCESS_REPEATS = 8
F_WEIGHT = 2.0
C_GRID = np.arange(-2.0, 2.0001, 0.1, dtype=np.float64)
REQUIREMENTS = "catboost==1.2.10\n"


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((y - p) ** 2))


def score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    ref = float(y.mean() * (1.0 - y.mean()))
    return max(0.0, 1e5 * (1.0 - brier(y, p) / ref)) if ref > 0 else 0.0


def add_regime_continuous(frame: pd.DataFrame, hand_levels: tuple[str, str]) -> None:
    """Reproduce only the continuous regime features used by the rich models.

    This avoids fitting/using any quantile edge from test rows. The two hand levels
    are frozen from official training data and stored in package metadata.
    """
    if len(hand_levels) != 2:
        raise ValueError(f"expected two frozen batter-hand levels, got {hand_levels}")
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    game_type = frame["game_type"].astype("string").fillna("<MISSING>").astype(str).str.upper()
    hand = frame["batter_hand"].astype("string").fillna("<MISSING>").astype(str)
    recent = season.ge(2023).to_numpy()
    old = ~recent
    is_r = game_type.eq("R").to_numpy()
    h1 = hand.eq(hand_levels[0]).to_numpy()
    h2 = hand.eq(hand_levels[1]).to_numpy()
    fast = pd.to_numeric(frame["asof_pitcher_fastball_rate"], errors="coerce").to_numpy(np.float32)
    rrng = pd.to_numeric(frame["eng_ps_recent_range_135"], errors="coerce").to_numpy(np.float32)
    frame[regime_core.RECENT_FLAG] = recent.astype(np.float32)
    for name, mask in {
        "rr_fastball_hand1": is_r & recent & h1,
        "rr_fastball_hand2": is_r & recent & h2,
        "ro_fastball_hand1": is_r & old & h1,
        "ro_fastball_hand2": is_r & old & h2,
    }.items():
        out = np.full(len(frame), np.nan, np.float32)
        out[mask] = fast[mask]
        frame[name] = out
    rr = np.full(len(frame), np.nan, np.float32)
    ro = np.full(len(frame), np.nan, np.float32)
    rr[is_r & recent] = rrng[is_r & recent]
    ro[is_r & old] = rrng[is_r & old]
    frame["rr_recent_range"] = rr
    frame["ro_recent_range"] = ro


def prepare_test_base(test: pd.DataFrame) -> pd.DataFrame:
    out = test.copy()
    add_canonical_derived_features(out)
    asof_core.add_asof_state_features(out)
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["game_type"] = out["game_type"].astype("string").str.strip().str.upper()
    return out


def _path_test_features(history: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work_test = prepare_test_base(test)
    work_test["control_success"] = np.nan
    work_test["_submit_pos"] = np.arange(len(work_test), dtype=np.int64)
    hist = history.copy()
    hist["_submit_pos"] = -1
    combo = pd.concat([hist, work_test], ignore_index=True, sort=False)
    cols = path_core.add_paths(combo, "pitcher_id", "season", "control_success")
    cols = [x for x in cols if not x.endswith("_rate")]
    picked = combo.loc[combo["_submit_pos"].ge(0), ["_submit_pos", *cols]].sort_values("_submit_pos")
    return picked.reset_index(drop=True), cols


def audit_path_independence(history: pd.DataFrame, sample: pd.DataFrame) -> tuple[bool, list[str]]:
    if not HAS_LOCAL_PATH_HELPER:
        print("[path] local run_frozen_domain_path_probe.py missing -> path features disabled")
        return False, []
    if sample.empty:
        print("[path] no sample rows -> path features disabled for rule safety")
        return False, []
    batch, cols = _path_test_features(history, sample)
    if not cols:
        print("[path] helper returned no path features")
        return False, []
    for i in range(min(5, len(sample))):
        single, single_cols = _path_test_features(history, sample.iloc[[i]].copy())
        if single_cols != cols:
            print("[path] feature-list mismatch in single-row audit -> disabled")
            return False, []
        for col in cols:
            a = batch.loc[i, col]
            b = single.loc[0, col]
            if pd.isna(a) and pd.isna(b):
                continue
            if str(a) != str(b):
                try:
                    if np.isfinite(float(a)) and np.isfinite(float(b)) and abs(float(a) - float(b)) <= 1e-12:
                        continue
                except Exception:
                    pass
                print(f"[path] peer-row dependence detected: row={i} col={col} batch={a!r} single={b!r} -> disabled")
                return False, []
    print(f"[path] independent-row audit passed ({len(cols)} features)")
    return True, cols


def engineer_history(
    base: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    hand_levels: tuple[str, str],
    use_paths: bool,
) -> tuple[pd.DataFrame, dict[str, list[str]], set[str]]:
    f = base.copy()
    anchor_core.add_frozen_anchor_features(
        f,
        season_col="season",
        pitcher_col="pitcher_id",
        n_col="asof_pitcher_n",
        count_tolerance=0.05,
    )
    banchor = offset_core.add_batter_anchor(f)
    across = offset_core.add_anchor_cross(f)
    matchup = offset_core.add_frozen_matchup(f)
    count = offset_core.add_frozen_count_profiles(f)
    pressure = offset_core.add_frozen_pressure_profiles(f)
    domain = offset_core.add_frozen_domain_profiles(f)
    auxprof = offset_core.add_frozen_aux_profiles(f, aux, "pressure")
    condprof = offset_core.add_frozen_conditional_profiles(f, aux, False)
    context_core.add_context_interactions(f)
    paths: list[str] = []
    if use_paths:
        paths = path_core.add_paths(f, "pitcher_id", "season", "control_success")
        paths = [x for x in paths if not x.endswith("_rate")]
    add_regime_continuous(f, hand_levels)
    path_cats = {x for x in paths if x.endswith(("last_gt", "current_x_last"))}
    regime_core.EXTRA_CATEGORICAL.update(path_cats)

    base_features = [
        *recent_core.feature_set("recent_raw_game_type"),
        regime_core.RECENT_FLAG,
        *regime_core.FAST_CONT,
        *regime_core.RANGE_CONT,
        *context_core.INTERACTION_COLUMNS,
        *paths,
    ]
    anchor_success = [
        "eng_anchor_available",
        "eng_anchor_gap_n",
        "eng_anchor_success_rate",
        "eng_since_anchor_success_rate",
        "eng_since_anchor_success_minus_long",
    ]
    multi_anchor = [
        x
        for short in ("reverse", "middle", "ball", "strike")
        for x in (
            f"eng_anchor_{short}_rate",
            f"eng_since_anchor_{short}_rate",
            f"eng_since_anchor_{short}_minus_long",
        )
    ]
    rich = [
        *base_features,
        *anchor_success,
        *multi_anchor,
        *banchor,
        *across,
        *matchup,
        *count,
        *pressure,
        *domain,
        *auxprof,
    ]
    hurdle = [*rich, *condprof]
    offset = [*base_features, *anchor_success, *banchor, *across]
    for name, feats in {"rich": rich, "hurdle": hurdle, "offset": offset}.items():
        if len(feats) != len(set(feats)):
            dup = sorted({x for x in feats if feats.count(x) > 1})
            raise RuntimeError(f"duplicate features in {name}: {dup}")
    return f, {"rich": rich, "hurdle": hurdle, "offset": offset, "paths": paths}, path_cats


def model_params(kind: str, *, devices: str, seed: int = 42, depth: int = 8) -> dict:
    p = dict(
        iterations=BASE_TREES,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=20,
        random_strength=0.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        border_count=128,
        random_seed=seed,
        has_time=True,
        one_hot_max_size=10,
        allow_writing_files=False,
        task_type="GPU",
        devices=devices,
        verbose=False,
    )
    if kind == "multi":
        p["loss_function"] = "MultiRMSE"
    elif kind == "reg":
        p["loss_function"] = "RMSE"
    elif kind == "class":
        p["loss_function"] = "Logloss"
    elif kind == "multiclass":
        p["loss_function"] = "MultiClass"
    else:
        raise ValueError(kind)
    return p


def prepared_pool(frame: pd.DataFrame, features: list[str], label=None, weight=None):
    from catboost import Pool

    x, cats = regime_core.prepare_x(frame, features)
    return Pool(x, label=label, weight=weight, cat_features=cats, feature_names=features), cats


def fit_multitask(train, valid, aux_train, features, devices):
    from catboost import CatBoostRegressor

    keep = aux_train[AUX_NAMES].notna().all(axis=1).to_numpy()
    success = train.loc[keep, "control_success"].to_numpy(np.float32)
    av = aux_train.loc[keep, AUX_NAMES].to_numpy(np.float32)
    labels = np.column_stack([*[success] * SUCCESS_REPEATS, av])
    w = np.where(train.loc[keep, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0).astype(np.float32)
    pool, cats = prepared_pool(train.loc[keep], features, labels, w)
    vp, _ = prepared_pool(valid, features)
    model = CatBoostRegressor(**model_params("multi", devices=devices)).fit(pool)
    pred = {t: np.clip(model.predict(vp, ntree_end=t), 0.0, 1.0) for t in (400, 600)}
    return model, pred, cats


def fit_aux_head(train, valid, aux_train, features, head, devices):
    from catboost import CatBoostClassifier

    keep = aux_train[head].notna().to_numpy()
    y = aux_train.loc[keep, head].to_numpy(np.int8)
    pool, cats = prepared_pool(train.loc[keep], features, y)
    vp, _ = prepared_pool(valid, features)
    model = CatBoostClassifier(**model_params("class", devices=devices)).fit(pool)
    pred = {t: model.predict_proba(vp, ntree_end=t)[:, 1] for t in (400, 600)}
    return model, pred, cats


def fit_hurdle(train, valid, aux_train, features, devices):
    from catboost import CatBoostClassifier

    usable = aux_train[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    gate_all = ((aux_train["reverse"] == 0) & (aux_train["middle"] == 0)).to_numpy()
    gate_y = gate_all[usable].astype(np.int8)
    gw = np.where(train.loc[usable, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
    gp, cats = prepared_pool(train.loc[usable], features, gate_y, gw)
    vp, _ = prepared_pool(valid, features)
    gate_model = CatBoostClassifier(**model_params("class", devices=devices)).fit(gp)

    cond = usable & gate_all
    cy = train.loc[cond, "control_success"].to_numpy(np.int8)
    cw = np.where(train.loc[cond, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
    cp, ccats = prepared_pool(train.loc[cond], features, cy, cw)
    cond_model = CatBoostClassifier(**model_params("class", devices=devices)).fit(cp)
    gate_pred = {t: gate_model.predict_proba(vp, ntree_end=t)[:, 1] for t in (200, 400, 600)}
    cond_pred = {t: cond_model.predict_proba(vp, ntree_end=t)[:, 1] for t in (200, 400, 600)}
    return gate_model, cond_model, gate_pred, cond_pred, cats, ccats


def fit_offset(train, valid, features, devices):
    from catboost import CatBoostRegressor

    mean = float(train["control_success"].mean())
    pt = offset_core.prior(train, "recent", mean)
    pv = offset_core.prior(valid, "recent", mean)
    residual = train["control_success"].to_numpy(np.float64) - pt
    pool, cats = prepared_pool(train, features, residual)
    vp, _ = prepared_pool(valid, features)
    model = CatBoostRegressor(**model_params("reg", devices=devices)).fit(pool)
    pred = {t: np.clip(pv + model.predict(vp, ntree_end=t), 0.0, 1.0) for t in (200, 400, 600)}
    return model, pred, mean, cats


def joint_mapping(train: pd.DataFrame, keep: np.ndarray, class_index: np.ndarray, success: np.ndarray, nclasses: int) -> np.ndarray:
    q = np.zeros((2, nclasses), dtype=np.float64)
    gt = train.loc[keep, "game_type"].astype(str).to_numpy()
    season = train.loc[keep, "season"].to_numpy(int)
    for gi, dom in enumerate(("R", "F")):
        dm = gt == dom
        if dom == "F" and np.any(dm & (season >= 2023)):
            dm &= season >= 2023
        fallback = float(success[dm].mean()) if dm.any() else float(success.mean())
        for ci in range(nclasses):
            cm = dm & (class_index == ci)
            q[gi, ci] = float(success[cm].mean()) if cm.any() else fallback
    return q


def fit_joint(train, valid, aux_train, features, devices):
    from catboost import CatBoostClassifier

    keep = aux_train[AUX_NAMES].notna().all(axis=1).to_numpy()
    av = aux_train.loc[keep, AUX_NAMES].to_numpy(np.int8)
    success = train.loc[keep, "control_success"].to_numpy(np.float64)
    codes = av @ (1 << np.arange(len(AUX_NAMES), dtype=np.int16))
    classes = np.unique(codes)
    ci = np.searchsorted(classes, codes)
    w = np.where(train.loc[keep, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
    pool, cats = prepared_pool(train.loc[keep], features, ci, w)
    vp, _ = prepared_pool(valid, features)
    model = CatBoostClassifier(**model_params("multiclass", devices=devices)).fit(pool)
    q = joint_mapping(train, keep, ci, success, len(classes))
    vg = (valid["game_type"].astype(str).to_numpy() == "F").astype(np.int8)
    pred = {}
    for t in (400, 600):
        prob = model.predict_proba(vp, ntree_end=t)
        pred[t] = np.sum(prob * q[vg], axis=1)
    return model, pred, q, classes, cats


def closed_form_blend(y, direct, logic, gt):
    pred = np.asarray(direct, np.float64).copy()
    ws = {}
    for dom in ("R", "F"):
        m = np.asarray(gt).astype(str) == dom
        d = logic[m] - direct[m]
        den = float(np.dot(d, d))
        w = 0.0 if den <= 0 else float(np.clip(np.dot(d, y[m] - direct[m]) / den, 0.0, 1.0))
        pred[m] = direct[m] + w * d
        ws[dom] = w
    return pred, ws


def select_gate_cond(y, gt, direct, reverse_pred, middle_pred, cond_pred):
    best = (float("inf"), None, None)
    for rt in (400, 600):
        for mt in (400, 600):
            r = reverse_pred[rt]
            m = middle_pred[mt]
            for ct in (200, 400, 600):
                cond = cond_pred[ct]
                for c in C_GRID:
                    logic = np.clip(1.0 - r - m + c * r * m, 0.0, 1.0) * cond
                    p, ws = closed_form_blend(y, direct, logic, gt)
                    v = brier(y, p)
                    if v < best[0]:
                        best = (v, {"reverse_tree": rt, "middle_tree": mt, "cond_tree": ct, "c": float(c), "blend": ws}, p)
    return best[1], best[2]


def fit_convex(y: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    k = matrix.shape[1]
    fun = lambda w: float(np.mean((matrix @ w - y) ** 2))
    res = minimize(
        fun,
        np.full(k, 1.0 / k),
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    if not res.success or not np.isfinite(res.x).all():
        raise RuntimeError(f"ensemble optimization failed: {res.message}")
    w = np.clip(np.asarray(res.x, np.float64), 0.0, 1.0)
    w /= w.sum()
    return w


def select_ensemble(y, gt, candidates: dict[str, np.ndarray]):
    names = list(candidates)
    z = np.column_stack([candidates[n] for n in names])
    weights = {}
    pred = np.empty(len(y), np.float64)
    for dom in ("R", "F"):
        m = np.asarray(gt).astype(str) == dom
        w = fit_convex(y[m], z[m])
        weights[dom] = w.tolist()
        pred[m] = z[m] @ w
    return names, weights, pred


def print_candidate_table(y, candidates):
    print("[2024 candidates]")
    for name, p in candidates.items():
        print(f"  {name:<12s} b={brier(y,p):.9f} s={score(y,p):7.1f}")


def copy_python_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*.py"):
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def make_fallback_path_module(path: Path) -> None:
    path.write_text(
        "def add_paths(frame, player_col, season_col, target_col):\n    return []\n",
        encoding="utf-8",
    )


INFERENCE_SCRIPT = r'''from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"
CODE = MODEL / "code"
for p in (CODE, CODE / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_recent_regime_submissions as recent_core
import run_asof_state_engineering as asof_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_frozen_season_anchor_probe as anchor_core
import run_offset_residual_boosting as offset_core
from src.canonical_features import add_canonical_derived_features

AUX_NAMES = ["reverse", "middle", "ball", "strike"]


def find_input_dir() -> Path:
    # The competition page currently contains both spellings in different sections.
    for name in ("open", "data"):
        p = ROOT / name
        if (p / "test.csv").is_file():
            return p
    raise FileNotFoundError("test.csv not found under ./open or ./data")


def add_regime_continuous(frame: pd.DataFrame, hand_levels: list[str]) -> None:
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    gt = frame["game_type"].astype("string").fillna("<MISSING>").astype(str).str.upper()
    hand = frame["batter_hand"].astype("string").fillna("<MISSING>").astype(str)
    recent = season.ge(2023).to_numpy(); old = ~recent; is_r = gt.eq("R").to_numpy()
    h1 = hand.eq(hand_levels[0]).to_numpy(); h2 = hand.eq(hand_levels[1]).to_numpy()
    fast = pd.to_numeric(frame["asof_pitcher_fastball_rate"], errors="coerce").to_numpy(np.float32)
    rrng = pd.to_numeric(frame["eng_ps_recent_range_135"], errors="coerce").to_numpy(np.float32)
    frame["regime_recent"] = recent.astype(np.float32)
    for name, mask in {
        "rr_fastball_hand1": is_r & recent & h1,
        "rr_fastball_hand2": is_r & recent & h2,
        "ro_fastball_hand1": is_r & old & h1,
        "ro_fastball_hand2": is_r & old & h2,
    }.items():
        out = np.full(len(frame), np.nan, np.float32); out[mask] = fast[mask]; frame[name] = out
    rr = np.full(len(frame), np.nan, np.float32); ro = np.full(len(frame), np.nan, np.float32)
    rr[is_r & recent] = rrng[is_r & recent]; ro[is_r & old] = rrng[is_r & old]
    frame["rr_recent_range"] = rr; frame["ro_recent_range"] = ro


def prepare_test(test: pd.DataFrame, meta: dict) -> pd.DataFrame:
    test = test.copy()
    add_canonical_derived_features(test)
    asof_core.add_asof_state_features(test)
    test["season"] = pd.to_numeric(test["season"], errors="raise").astype(int)
    test["game_type"] = test["game_type"].astype("string").str.strip().str.upper()
    test["control_success"] = np.nan
    test["_submit_pos"] = np.arange(len(test), dtype=np.int64)

    history = pd.read_pickle(MODEL / "history.pkl.gz")
    aux_hist = pd.read_pickle(MODEL / "history_aux.pkl.gz").reset_index(drop=True)
    history = history.reset_index(drop=True)
    history["_submit_pos"] = -1
    combo = pd.concat([history, test], ignore_index=True, sort=False)
    aux_test = pd.DataFrame(np.nan, index=np.arange(len(test)), columns=AUX_NAMES, dtype=np.float32)
    aux = pd.concat([aux_hist[AUX_NAMES], aux_test], ignore_index=True)

    anchor_core.add_frozen_anchor_features(combo, season_col="season", pitcher_col="pitcher_id", n_col="asof_pitcher_n", count_tolerance=.05)
    offset_core.add_batter_anchor(combo)
    offset_core.add_anchor_cross(combo)
    offset_core.add_frozen_matchup(combo)
    offset_core.add_frozen_count_profiles(combo)
    offset_core.add_frozen_pressure_profiles(combo)
    offset_core.add_frozen_domain_profiles(combo)
    offset_core.add_frozen_aux_profiles(combo, aux, "pressure")
    offset_core.add_frozen_conditional_profiles(combo, aux, False)
    if meta["use_paths"]:
        path_core.add_paths(combo, "pitcher_id", "season", "control_success")

    out = combo.loc[combo["_submit_pos"].ge(0)].sort_values("_submit_pos").copy()
    if len(out) != len(test):
        raise RuntimeError("engineered test row count mismatch")
    context_core.add_context_interactions(out)
    add_regime_continuous(out, meta["hand_levels"])
    return out


def make_pool(frame: pd.DataFrame, spec: dict) -> Pool:
    features = spec["features"]; cats = set(spec["categorical"])
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"missing inference features: {missing[:20]}")
    x = frame.loc[:, features].copy()
    for col in features:
        if col in cats:
            x[col] = x[col].astype("string").fillna("<MISSING>").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(np.float32).replace([np.inf, -np.inf], np.nan)
    return Pool(x, cat_features=spec["categorical"], feature_names=features)


def load_reg(name: str) -> CatBoostRegressor:
    m = CatBoostRegressor(); m.load_model(str(MODEL / f"{name}.cbm")); return m


def load_cls(name: str) -> CatBoostClassifier:
    m = CatBoostClassifier(); m.load_model(str(MODEL / f"{name}.cbm")); return m


def reg_predict(model, pool, tree):
    try:
        return model.predict(pool, ntree_end=int(tree), task_type="GPU")
    except Exception:
        return model.predict(pool, ntree_end=int(tree), thread_count=-1)


def cls_predict(model, pool, tree):
    try:
        return model.predict_proba(pool, ntree_end=int(tree), task_type="GPU")
    except Exception:
        return model.predict_proba(pool, ntree_end=int(tree), thread_count=-1)


def offset_prior(frame: pd.DataFrame, mean: float) -> np.ndarray:
    n = pd.to_numeric(frame.asof_pitcher_n, errors="coerce").to_numpy(float)
    p = pd.to_numeric(frame.asof_pitcher_success_rate, errors="coerce").to_numpy(float)
    b = pd.to_numeric(frame.asof_batter_success_rate, errors="coerce").to_numpy(float)
    shr = (n * p + 200.0 * mean) / (n + 200.0)
    shr = np.nan_to_num(shr, nan=mean); b = np.nan_to_num(b, nan=mean)
    recent = frame[["asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate"]].mean(axis=1).to_numpy(float)
    recent = np.nan_to_num(recent, nan=mean)
    return .65 * shr + .25 * recent + .10 * b


def main() -> None:
    t0 = time.time()
    meta = json.loads((MODEL / "metadata.json").read_text(encoding="utf-8"))
    inp = find_input_dir()
    test = pd.read_csv(inp / "test.csv", low_memory=False)
    if "row_id" not in test.columns:
        raise ValueError("test.csv missing row_id")
    f = prepare_test(test, meta)

    rich_pool = make_pool(f, meta["specs"]["rich"])
    hurdle_pool = make_pool(f, meta["specs"]["hurdle"])
    offset_pool = make_pool(f, meta["specs"]["offset"])

    multi = load_reg("multi")
    all_multi = np.clip(reg_predict(multi, rich_pool, meta["selection"]["multi_tree"]), 0.0, 1.0)
    direct = all_multi[:, 0]

    rmodel = load_cls("aux_reverse"); mmodel = load_cls("aux_middle")
    gs = meta["selection"]["gate_cond"]
    r = cls_predict(rmodel, rich_pool, gs["reverse_tree"])[:, 1]
    m = cls_predict(mmodel, rich_pool, gs["middle_tree"])[:, 1]

    gate_model = load_cls("hurdle_gate"); cond_model = load_cls("hurdle_cond")
    htree = int(meta["selection"]["hurdle_tree"])
    gate_h = cls_predict(gate_model, hurdle_pool, htree)[:, 1]
    cond_h = cls_predict(cond_model, hurdle_pool, htree)[:, 1]
    hurdle = gate_h * cond_h
    cond_logic = cls_predict(cond_model, hurdle_pool, gs["cond_tree"])[:, 1]
    logic = np.clip(1.0 - r - m + float(gs["c"]) * r * m, 0.0, 1.0) * cond_logic
    gt = f["game_type"].astype(str).to_numpy()
    blend = np.where(gt == "F", float(gs["blend"]["F"]), float(gs["blend"]["R"]))
    gate_cond = direct + blend * (logic - direct)

    off_model = load_reg("offset")
    prior = offset_prior(f, float(meta["offset_mean"]))
    off = np.clip(prior + reg_predict(off_model, offset_pool, meta["selection"]["offset_tree"]), 0.0, 1.0)

    joint_model = load_cls("joint")
    jp = cls_predict(joint_model, rich_pool, meta["selection"]["joint_tree"])
    q = np.asarray(meta["joint_q"], dtype=np.float64)
    gi = (gt == "F").astype(np.int8)
    joint = np.sum(jp * q[gi], axis=1)

    candidates = np.column_stack([gate_cond, direct, off, hurdle, joint])
    expected = meta["candidate_names"]
    if expected != ["gate_cond", "multi", "offset", "hurdle", "joint"]:
        raise RuntimeError(f"unexpected candidate order: {expected}")
    wr = np.asarray(meta["ensemble_weights"]["R"], dtype=np.float64)
    wf = np.asarray(meta["ensemble_weights"]["F"], dtype=np.float64)
    pred = np.empty(len(f), dtype=np.float64)
    mr = gt == "R"; mf = gt == "F"
    pred[mr] = candidates[mr] @ wr
    pred[mf] = candidates[mf] @ wf
    other = ~(mr | mf)
    if other.any():
        pred[other] = candidates[other] @ ((wr + wf) / 2.0)
    pred = np.clip(pred, 0.0, 1.0)
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite prediction")

    sample = inp / "sample_submission.csv"
    if sample.is_file():
        sub = pd.read_csv(sample)
        if len(sub) != len(test):
            raise RuntimeError("sample_submission/test row count mismatch")
        if "row_id" not in sub.columns or not sub["row_id"].astype(str).equals(test["row_id"].astype(str)):
            raise RuntimeError("sample_submission row_id differs from test row order")
        sub = sub[["row_id"]].copy()
    else:
        sub = pd.DataFrame({"row_id": test["row_id"].to_numpy()})
    sub["control_success"] = pred
    out = ROOT / "output"; out.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out / "submission.csv", index=False)
    print(f"rows={len(sub):,} mean={pred.mean():.6f} std={pred.std():.6f} min={pred.min():.6f} max={pred.max():.6f} sec={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
'''


def train_validation_suite(frame, aux, features, devices):
    train = frame.loc[frame.season < 2024].copy()
    valid = frame.loc[frame.season.eq(2024)].copy()
    aux_train = aux.loc[train.index].reset_index(drop=True)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    y = valid.control_success.to_numpy(np.float64)
    gt = valid.game_type.astype(str).to_numpy()

    multi_model, mp, rich_cats = fit_multitask(train, valid, aux_train, features["rich"], devices)
    multi_tree = min((400, 600), key=lambda t: brier(y, mp[t][:, 0]))
    direct = mp[multi_tree][:, 0]
    del multi_model; gc.collect()

    _, rp, _ = fit_aux_head(train, valid, aux_train, features["rich"], "reverse", devices)
    _, mip, _ = fit_aux_head(train, valid, aux_train, features["rich"], "middle", devices)
    gate_model, cond_model, gp, cp, hurdle_cats, _ = fit_hurdle(train, valid, aux_train, features["hurdle"], devices)
    hurdle_tree = min((200, 400, 600), key=lambda t: brier(y, gp[t] * cp[t]))
    hurdle = gp[hurdle_tree] * cp[hurdle_tree]
    del gate_model, cond_model; gc.collect()

    _, op, _, offset_cats = fit_offset(train, valid, features["offset"], devices)
    offset_tree = min((200, 400, 600), key=lambda t: brier(y, op[t]))
    off = op[offset_tree]

    _, jp, _, _, _ = fit_joint(train, valid, aux_train, features["rich"], devices)
    joint_tree = min((400, 600), key=lambda t: brier(y, jp[t]))
    joint = jp[joint_tree]

    gate_spec, gate_cond = select_gate_cond(y, gt, direct, rp, mip, cp)
    candidates = {
        "gate_cond": gate_cond,
        "multi": direct,
        "offset": off,
        "hurdle": hurdle,
        "joint": joint,
    }
    print_candidate_table(y, candidates)
    names, weights, ensemble = select_ensemble(y, gt, candidates)
    print(f"[2024 selected] b={brier(y,ensemble):.9f} s={score(y,ensemble):.1f}")
    print(f"  R={dict(zip(names, np.round(weights['R'],6)))}")
    print(f"  F={dict(zip(names, np.round(weights['F'],6)))}")
    selection = {
        "multi_tree": int(multi_tree),
        "hurdle_tree": int(hurdle_tree),
        "offset_tree": int(offset_tree),
        "joint_tree": int(joint_tree),
        "gate_cond": gate_spec,
    }
    return selection, names, weights, brier(y, ensemble), score(y, ensemble), rich_cats, hurdle_cats, offset_cats


def train_final_suite(frame, aux, features, devices, package_model: Path, selection):
    train = frame.reset_index(drop=True)
    aux_train = aux.reset_index(drop=True)
    empty = train.iloc[:0].copy()

    # Multi-output current model.
    keep = aux_train[AUX_NAMES].notna().all(axis=1).to_numpy()
    success = train.loc[keep, "control_success"].to_numpy(np.float32)
    av = aux_train.loc[keep, AUX_NAMES].to_numpy(np.float32)
    labels = np.column_stack([*[success] * SUCCESS_REPEATS, av])
    w = np.where(train.loc[keep, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
    from catboost import CatBoostClassifier, CatBoostRegressor
    pool, rich_cats = prepared_pool(train.loc[keep], features["rich"], labels, w)
    multi = CatBoostRegressor(**model_params("multi", devices=devices)).fit(pool)
    multi.save_model(str(package_model / "multi.cbm")); del multi, pool; gc.collect()

    # Independent reverse / middle heads used by the learned gate composition.
    for head in ("reverse", "middle"):
        k = aux_train[head].notna().to_numpy()
        p, _ = prepared_pool(train.loc[k], features["rich"], aux_train.loc[k, head].to_numpy(np.int8))
        m = CatBoostClassifier(**model_params("class", devices=devices)).fit(p)
        m.save_model(str(package_model / f"aux_{head}.cbm")); del m, p; gc.collect()

    # Hurdle gate and valid-only conditional head, both with F x2 weighting.
    usable = aux_train[["reverse", "middle"]].notna().all(axis=1).to_numpy()
    gate_all = ((aux_train["reverse"] == 0) & (aux_train["middle"] == 0)).to_numpy()
    gw = np.where(train.loc[usable, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
    gp, hurdle_cats = prepared_pool(train.loc[usable], features["hurdle"], gate_all[usable].astype(np.int8), gw)
    gm = CatBoostClassifier(**model_params("class", devices=devices)).fit(gp)
    gm.save_model(str(package_model / "hurdle_gate.cbm")); del gm, gp; gc.collect()
    cond = usable & gate_all
    cw = np.where(train.loc[cond, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
    cp, _ = prepared_pool(train.loc[cond], features["hurdle"], train.loc[cond, "control_success"].to_numpy(np.int8), cw)
    cm = CatBoostClassifier(**model_params("class", devices=devices)).fit(cp)
    cm.save_model(str(package_model / "hurdle_cond.cbm")); del cm, cp; gc.collect()

    # Residual model keeps the deliberately different prior/loss path for diversity.
    mean = float(train.control_success.mean())
    pt = offset_core.prior(train, "recent", mean)
    residual = train.control_success.to_numpy(np.float64) - pt
    opool, offset_cats = prepared_pool(train, features["offset"], residual)
    om = CatBoostRegressor(**model_params("reg", devices=devices)).fit(opool)
    om.save_model(str(package_model / "offset.cbm")); del om, opool; gc.collect()

    # 4-bit joint auxiliary outcome classifier (usually 12 observed states).
    keepj = aux_train[AUX_NAMES].notna().all(axis=1).to_numpy()
    jav = aux_train.loc[keepj, AUX_NAMES].to_numpy(np.int8)
    jsuccess = train.loc[keepj, "control_success"].to_numpy(np.float64)
    codes = jav @ (1 << np.arange(len(AUX_NAMES), dtype=np.int16))
    classes = np.unique(codes); ci = np.searchsorted(classes, codes)
    jw = np.where(train.loc[keepj, "game_type"].astype(str).to_numpy() == "F", F_WEIGHT, 1.0)
    jpool, _ = prepared_pool(train.loc[keepj], features["rich"], ci, jw)
    jm = CatBoostClassifier(**model_params("multiclass", devices=devices)).fit(jpool)
    jm.save_model(str(package_model / "joint.cbm")); del jm, jpool; gc.collect()
    q = joint_mapping(train, keepj, ci, jsuccess, len(classes))
    return {
        "rich": {"features": features["rich"], "categorical": rich_cats},
        "hurdle": {"features": features["hurdle"], "categorical": hurdle_cats},
        "offset": {"features": features["offset"], "categorical": offset_cats},
    }, mean, q, classes.tolist()


def stage_package(
    package: Path,
    base_history: pd.DataFrame,
    aux_history: pd.DataFrame,
    *,
    metadata: dict,
) -> None:
    model_dir = package / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Keep official train-derived history as a frozen model artifact. Test labels are
    # never reconstructed; test rows are appended with target/aux labels set to NaN.
    base_history.to_pickle(model_dir / "history.pkl.gz", compression="gzip")
    aux_history[AUX_NAMES].to_pickle(model_dir / "history_aux.pkl.gz", compression="gzip")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    code = model_dir / "code"
    copy_python_tree(ROOT / "src", code / "src")
    copy_python_tree(ROOT / "scripts", code / "scripts")
    if not (code / "scripts" / "run_frozen_domain_path_probe.py").is_file():
        make_fallback_path_module(code / "scripts" / "run_frozen_domain_path_probe.py")
    (package / "script.py").write_text(INFERENCE_SCRIPT, encoding="utf-8")
    (package / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")


def smoke_test(package: Path, data_dir: Path, timeout_sec: int) -> float:
    local = package / "open"
    local.mkdir(exist_ok=True)
    shutil.copy2(data_dir / "test.csv", local / "test.csv")
    sample = data_dir / "sample_submission.csv"
    if sample.is_file():
        shutil.copy2(sample, local / "sample_submission.csv")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "script.py"],
        cwd=package,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    sec = time.time() - t0
    print(proc.stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"submission smoke test failed with code {proc.returncode}")
    out = pd.read_csv(package / "output" / "submission.csv")
    p = pd.to_numeric(out["control_success"], errors="raise").to_numpy(float)
    if len(out) == 0 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise RuntimeError("invalid smoke-test submission")
    shutil.rmtree(local, ignore_errors=True)
    shutil.rmtree(package / "output", ignore_errors=True)
    print(f"[smoke] ok rows={len(out):,} sec={sec:.1f}")
    return sec


def write_zip(package: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(package).as_posix())
    with zipfile.ZipFile(output_zip) as zf:
        names = zf.namelist()
        roots = {name.split("/", 1)[0] for name in names}
        if not {"model", "script.py", "requirements.txt"}.issubset(roots):
            raise RuntimeError(f"bad zip roots: {roots}")
        if roots - {"model", "script.py", "requirements.txt"}:
            raise RuntimeError(f"unexpected top-level entries: {sorted(roots)}")
        extracted = sum(info.file_size for info in zf.infolist())
    zipped = output_zip.stat().st_size
    if zipped > 10 * 1024**3:
        raise RuntimeError(f"zip exceeds 10 GiB: {zipped / 1024**3:.2f}")
    if extracted > 32 * 1024**3:
        raise RuntimeError(f"extracted package exceeds 32 GiB: {extracted / 1024**3:.2f}")
    print(f"[zip] {output_zip} compressed={zipped/1024**2:.1f}MiB extracted={extracted/1024**2:.1f}MiB")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a rule-audited code-submission ZIP from the current frozen-profile outcome-decomposition branch. "
            "Only this builder file is added to the development repo; experiment files are not modified."
        )
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--devices", default="0", help="CatBoost logical GPU id; use 0 with CUDA_VISIBLE_DEVICES=2")
    ap.add_argument("--output", default="dist/current_best_submit.zip")
    ap.add_argument("--smoke-data-dir", default="data")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-timeout", type=int, default=540)
    ap.add_argument("--disable-paths", action="store_true", help="Force-disable the local frozen path helper")
    ap.add_argument("--reference-current-best", default="outputs/current_best_2024_predictions.npz")
    args = ap.parse_args()

    config = load_config(ROOT / args.config)
    base, _ = recent_core.prepare_frame(config)
    base["season"] = pd.to_numeric(base["season"], errors="raise").astype(int)
    base["game_type"] = base["game_type"].astype("string").str.strip().str.upper()
    aux = multi_core.auxiliary_targets(base).reset_index(drop=True)
    base = base.reset_index(drop=True)
    hands = sorted(base["batter_hand"].astype("string").dropna().astype(str).unique().tolist())
    if len(hands) != 2:
        raise RuntimeError(f"expected exactly two batter_hand levels, got {hands}")
    hand_levels = (hands[0], hands[1])

    sample_dir = (ROOT / args.smoke_data_dir).resolve()
    sample_path = sample_dir / "test.csv"
    sample = pd.read_csv(sample_path, low_memory=False) if sample_path.is_file() else pd.DataFrame()
    if args.disable_paths:
        use_paths = False
        print("[path] disabled by CLI")
    else:
        use_paths, _ = audit_path_independence(base, sample)

    print(f"[data] rows={len(base):,} aux_complete={aux[AUX_NAMES].notna().all(axis=1).mean():.6f} paths={use_paths}")
    frame, feature_sets, path_cats = engineer_history(
        base,
        aux,
        hand_levels=hand_levels,
        use_paths=use_paths,
    )

    selection, candidate_names, ensemble_weights, val_brier, val_score, _, _, _ = train_validation_suite(
        frame,
        aux,
        feature_sets,
        args.devices,
    )
    ref_path = ROOT / args.reference_current_best
    if ref_path.is_file():
        try:
            ref = np.load(ref_path, allow_pickle=True)
            if "y" in ref and "pred" in ref and len(ref["y"]) == int(frame.season.eq(2024).sum()):
                rb = brier(ref["y"], ref["pred"])
                print(f"[local reference] current_best b={rb:.9f}; deployable builder validation b={val_brier:.9f}; d={val_brier-rb:+.3e}")
        except Exception as exc:
            print(f"[local reference] skipped: {exc}")

    output_zip = (ROOT / args.output).resolve()
    with tempfile.TemporaryDirectory(prefix="aimers_current_best_submit_") as tmp:
        package = Path(tmp)
        model_dir = package / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        specs, offset_mean, joint_q, joint_classes = train_final_suite(
            frame,
            aux,
            feature_sets,
            args.devices,
            model_dir,
            selection,
        )
        metadata = {
            "builder": "scripts/build_current_best_submission.py",
            "source_commit": "1e0f3eab4792880a8ea13335d75138dbc0b7ad10",
            "architecture": "frozen contextual profiles + MultiRMSE current head + independent reverse/middle heads + hurdle conditional + residual offset + joint auxiliary outcome",
            "official_history_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
            "target_season": 2025,
            "independent_test_rows": True,
            "test_auxiliary_reconstruction": False,
            "use_paths": bool(use_paths),
            "path_categorical": sorted(path_cats),
            "hand_levels": list(hand_levels),
            "candidate_names": candidate_names,
            "ensemble_weights": ensemble_weights,
            "selection": selection,
            "validation_2024_brier": float(val_brier),
            "validation_2024_score": float(val_score),
            "offset_mean": float(offset_mean),
            "joint_q": np.asarray(joint_q).tolist(),
            "joint_classes": joint_classes,
            "specs": specs,
            "training": {"trees": BASE_TREES, "depth": 8, "success_repeats": SUCCESS_REPEATS, "f_weight": F_WEIGHT, "catboost": "1.2.10"},
            "rule_guard": {
                "external_data": False,
                "remote_api": False,
                "other_test_rows_in_prediction": False,
                "test_distribution_calibration": False,
                "training_aux_from_official_history_only": True,
                "frozen_profiles_from_official_2019_2024_only": True,
            },
        }
        stage_package(package, base, aux, metadata=metadata)
        # stage_package writes history/metadata after models already exist in model/.
        if not args.skip_smoke:
            if not sample_path.is_file():
                raise FileNotFoundError(f"smoke test.csv not found: {sample_path}")
            smoke_test(package, sample_dir, args.smoke_timeout)
        write_zip(package, output_zip)

    print("[done] submit.zip contract: model/ + script.py + requirements.txt")
    print(f"[done] {output_zip}")


if __name__ == "__main__":
    main()
