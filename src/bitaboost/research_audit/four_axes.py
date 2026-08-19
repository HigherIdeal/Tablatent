from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bitaboost.baseline import _prepare_x
from bitaboost.config import load_config
from bitaboost.ex1.pitcher_season_backward import STATE_NAMES, _adjacent_pairs, _season_profiles
from bitaboost.ex2.hypothesis_backward import ID_FEATURES, _base_features
from bitaboost.features import prepare
from bitaboost.night.common import auc


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else _root() / p


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("research audit config root must be a mapping")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, np.float64)
    pp = np.asarray(p, np.float64)
    return float(np.mean((yy - pp) ** 2))


def _rmse(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    err = (np.asarray(y, np.float64) - np.asarray(p, np.float64)) ** 2
    if w is None:
        return float(np.sqrt(np.mean(err)))
    ww = np.asarray(w, np.float64)
    return float(np.sqrt(np.sum(ww * err) / np.sum(ww)))


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, np.float64)
    bb = np.asarray(b, np.float64)
    keep = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[keep]
    bb = bb[keep]
    if len(aa) < 3 or float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _classification(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    yy = np.asarray(y, np.float64)
    pp = np.clip(np.asarray(p, np.float64), 0.0, 1.0)
    return {
        "rows": int(len(yy)),
        "rate": float(np.mean(yy)),
        "pred_mean": float(np.mean(pp)),
        "residual_mean": float(np.mean(yy - pp)),
        "brier": _brier(yy, pp),
        "auc": auc(yy.astype(np.int8), pp),
    }


def _cat_params(exp: dict[str, Any], *, iterations: int | None = None) -> dict[str, Any]:
    p = dict(exp["row_model"])
    if iterations is not None:
        p["iterations"] = int(iterations)
    p.update(
        {
            "loss_function": "Logloss",
            "task_type": "GPU",
            "devices": "0",
            "bootstrap_type": "Bayesian",
            "allow_writing_files": False,
            "logging_level": "Silent",
        }
    )
    return p


def _fit_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: list[str],
    target_col: str,
    params: dict[str, Any],
) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool

    x, cats = _prepare_x(train, features)
    xv, _ = _prepare_x(test, features)
    y = pd.to_numeric(train[target_col], errors="raise").to_numpy(np.int8)
    model = CatBoostClassifier(**params)
    model.fit(Pool(x, y, cat_features=cats, feature_names=features))
    pred = np.asarray(
        model.predict_proba(Pool(xv, cat_features=cats, feature_names=features))[:, 1],
        dtype=np.float64,
    )
    del model, x, xv
    gc.collect()
    return np.clip(pred, 0.0, 1.0)


def _feature_groups(frame: pd.DataFrame, season_col: str) -> dict[str, list[str]]:
    base = [
        f
        for f in _base_features()
        if f in frame.columns and f not in {"control_success", "row_id", season_col, "season"}
    ]
    pitcher_hist = [f for f in base if f.startswith("asof_pitcher_") or f.startswith("eng_ps_")]
    batter_hist = [f for f in base if f.startswith("asof_batter_")]
    context = [
        f
        for f in base
        if f not in pitcher_hist and f not in batter_hist and f not in ID_FEATURES
    ]
    groups = {
        "context": list(dict.fromkeys(context)),
        "pitcher_history": list(dict.fromkeys([*context, *pitcher_hist])),
        "batter_history": list(dict.fromkeys([*context, *batter_hist])),
        "full_history": list(dict.fromkeys([*context, *pitcher_hist, *batter_hist])),
    }
    for name, cols in groups.items():
        missing = [x for x in cols if x not in frame.columns]
        if missing:
            raise RuntimeError(f"{name} missing columns: {missing[:10]}")
        if len(cols) != len(set(cols)):
            raise RuntimeError(f"duplicate columns in {name}")
    return groups


def _rolling_information_audit(
    frame: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    folds: list[int],
    groups: dict[str, list[str]],
    exp: dict[str, Any],
    out: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    fold_metrics: dict[str, Any] = {}
    params = _cat_params(exp)
    for fold in folds:
        season = pd.to_numeric(frame[season_col], errors="raise").astype(int)
        tr = frame.loc[season < int(fold)].reset_index(drop=True)
        va = frame.loc[season == int(fold)].reset_index(drop=True)
        if tr.empty or va.empty:
            raise RuntimeError(f"rolling fold {fold} is empty")
        print(f"[D rolling fold={fold}] train={len(tr):,} valid={len(va):,}", flush=True)
        block = pd.DataFrame(
            {
                "row_id": va["row_id"].astype(str).to_numpy(),
                "season": np.full(len(va), int(fold), dtype=np.int16),
                "pitcher_id": va["pitcher_id"].astype(str).to_numpy(),
                "y": pd.to_numeric(va[target_col], errors="raise").to_numpy(np.float64),
                "asof_pitcher_n": pd.to_numeric(va["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(np.float64),
                "asof_batter_n": pd.to_numeric(va["asof_batter_n"], errors="coerce").fillna(0).to_numpy(np.float64),
            }
        )
        fold_metrics[str(fold)] = {}
        for name, features in groups.items():
            t0 = time.perf_counter()
            pred = _fit_probability(
                tr,
                va,
                features=features,
                target_col=target_col,
                params=params,
            )
            block[f"pred_{name}"] = pred
            metric = _classification(block["y"].to_numpy(), pred)
            fold_metrics[str(fold)][name] = metric
            print(
                f"  [{name:15s}] brier={metric['brier']:.9f} auc={metric['auc']:.5f} "
                f"sec={time.perf_counter()-t0:.1f}",
                flush=True,
            )
        rows.append(block)
    oof = pd.concat(rows, axis=0, ignore_index=True)
    np.savez_compressed(
        out / "rolling_predictions.npz",
        row_id=oof["row_id"].to_numpy(dtype="U"),
        season=oof["season"].to_numpy(np.int16),
        pitcher_id=oof["pitcher_id"].to_numpy(dtype="U"),
        y=oof["y"].to_numpy(np.float64),
        asof_pitcher_n=oof["asof_pitcher_n"].to_numpy(np.float64),
        asof_batter_n=oof["asof_batter_n"].to_numpy(np.float64),
        **{f"pred_{name}": oof[f"pred_{name}"].to_numpy(np.float64) for name in groups},
    )
    return oof, fold_metrics


def _history_level(x: np.ndarray, cold: int, rich: int) -> np.ndarray:
    xx = np.asarray(x, np.float64)
    return np.where(xx < cold, "cold", np.where(xx >= rich, "rich", "mid")).astype(str)


def _cold_start_summary(
    oof: pd.DataFrame,
    *,
    cold: int,
    rich: int,
    variants: list[str],
) -> dict[str, Any]:
    p = _history_level(oof["asof_pitcher_n"].to_numpy(), cold, rich)
    b = _history_level(oof["asof_batter_n"].to_numpy(), cold, rich)
    cells = np.char.add(np.char.add("P:", p.astype(str)), np.char.add("|B:", b.astype(str)))
    result: dict[str, Any] = {"thresholds": {"cold_lt": cold, "rich_ge": rich}, "cells": {}}
    for cell in sorted(set(cells.tolist())):
        mask = cells == cell
        y = oof.loc[mask, "y"].to_numpy(np.float64)
        item: dict[str, Any] = {"rows": int(mask.sum()), "models": {}}
        for name in variants:
            pred = oof.loc[mask, f"pred_{name}"].to_numpy(np.float64)
            item["models"][name] = _classification(y, pred)
        ranked = sorted(item["models"].items(), key=lambda kv: float(kv[1]["brier"]))
        item["best_model"] = ranked[0][0]
        item["best_brier"] = float(ranked[0][1]["brier"])
        ctx = float(item["models"]["context"]["brier"])
        item["gain_vs_context"] = {
            name: ctx - float(item["models"][name]["brier"]) for name in variants if name != "context"
        }
        result["cells"][cell] = item
    corners = ["P:cold|B:cold", "P:cold|B:rich", "P:rich|B:cold", "P:rich|B:rich"]
    available = [result["cells"][x] for x in corners if x in result["cells"]]
    bests = {x["best_model"] for x in available}
    gains = [float(x["gain_vs_context"].get("full_history", 0.0)) for x in available]
    result["corner_best_models"] = sorted(bests)
    result["corner_full_gain_range"] = float(max(gains) - min(gains)) if gains else 0.0
    if len(bests) >= 2 or result["corner_full_gain_range"] >= 0.00020:
        result["signal"] = "heterogeneous"
    else:
        result["signal"] = "weak"
    return result


def _matched_season_pairs(
    frame: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    min_rows: int,
) -> dict[str, Any]:
    candidate_keys = [
        "pitcher_id",
        "game_type",
        "balls_before",
        "strikes_before",
        "outs_before",
        "batter_hand",
    ]
    keys = [x for x in candidate_keys if x in frame.columns]
    use = frame[[season_col, target_col, *keys]].copy()
    use[target_col] = pd.to_numeric(use[target_col], errors="raise")
    agg = (
        use.groupby([season_col, *keys], dropna=False, sort=False)[target_col]
        .agg(["count", "mean"])
        .reset_index()
    )
    seasons = sorted(pd.to_numeric(frame[season_col], errors="raise").astype(int).unique().tolist())
    out: dict[str, Any] = {"keys": keys, "pairs": {}}
    for old, new in zip(seasons[:-1], seasons[1:]):
        left = agg.loc[agg[season_col].astype(int) == int(old)].drop(columns=[season_col]).rename(
            columns={"count": "n_old", "mean": "mean_old"}
        )
        right = agg.loc[agg[season_col].astype(int) == int(new)].drop(columns=[season_col]).rename(
            columns={"count": "n_new", "mean": "mean_new"}
        )
        joined = left.merge(right, on=keys, how="inner", sort=False)
        joined = joined.loc[(joined.n_old >= min_rows) & (joined.n_new >= min_rows)]
        if joined.empty:
            continue
        w = np.minimum(joined.n_old.to_numpy(np.float64), joined.n_new.to_numpy(np.float64))
        delta = joined.mean_new.to_numpy(np.float64) - joined.mean_old.to_numpy(np.float64)
        raw_old = float(frame.loc[frame[season_col].astype(int) == int(old), target_col].mean())
        raw_new = float(frame.loc[frame[season_col].astype(int) == int(new), target_col].mean())
        out["pairs"][f"{old}->{new}"] = {
            "matched_cells": int(len(joined)),
            "effective_rows": float(w.sum()),
            "matched_weighted_delta": float(np.average(delta, weights=w)),
            "matched_weighted_abs_delta": float(np.average(np.abs(delta), weights=w)),
            "raw_rate_delta": raw_new - raw_old,
        }
    return out


def _pooled_season_increment(
    frame: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    full_features: list[str],
    exp: dict[str, Any],
) -> dict[str, Any]:
    cfg = exp["conditional_shift"]
    modulo = int(cfg.get("holdout_modulo", 5))
    remainder = int(cfg.get("holdout_remainder", 0))
    hashes = pd.util.hash_pandas_object(frame["row_id"].astype(str), index=False).to_numpy(np.uint64)
    hold = (hashes % modulo) == remainder
    fit = ~hold
    train = frame.loc[fit].reset_index(drop=True)
    test = frame.loc[hold].reset_index(drop=True)
    no_season = [x for x in full_features if x != season_col and x != "season"]
    with_season = list(no_season)
    if season_col in frame.columns:
        with_season.append(season_col)
    iterations = int(cfg.get("pooled_iterations", 260))
    params = _cat_params(exp, iterations=iterations)
    print(f"[A pooled] fit={len(train):,} holdout={len(test):,} no-season vs +season", flush=True)
    pred0 = _fit_probability(train, test, features=no_season, target_col=target_col, params=params)
    pred1 = _fit_probability(train, test, features=with_season, target_col=target_col, params=params)
    y = pd.to_numeric(test[target_col], errors="raise").to_numpy(np.float64)
    seasons = pd.to_numeric(test[season_col], errors="raise").astype(int).to_numpy()
    result: dict[str, Any] = {
        "without_season": _classification(y, pred0),
        "with_season": _classification(y, pred1),
        "brier_gain_from_season": _brier(y, pred0) - _brier(y, pred1),
        "by_season": {},
    }
    for season in sorted(set(seasons.tolist())):
        m = seasons == season
        result["by_season"][str(season)] = {
            "rows": int(m.sum()),
            "without_season_brier": _brier(y[m], pred0[m]),
            "with_season_brier": _brier(y[m], pred1[m]),
            "gain": _brier(y[m], pred0[m]) - _brier(y[m], pred1[m]),
        }
    gain = float(result["brier_gain_from_season"])
    result["signal"] = "strong" if gain >= 0.00010 else ("weak" if gain >= 0.00002 else "little")
    return result


def _conditional_shift_summary(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    full_features: list[str],
    exp: dict[str, Any],
) -> dict[str, Any]:
    pooled = _pooled_season_increment(
        frame,
        season_col=season_col,
        target_col=target_col,
        full_features=full_features,
        exp=exp,
    )
    matched = _matched_season_pairs(
        frame,
        season_col=season_col,
        target_col=target_col,
        min_rows=int(exp.get("matched_cell_min_rows", 5)),
    )
    rolling: dict[str, Any] = {}
    for season in sorted(oof["season"].unique().tolist()):
        m = oof["season"].to_numpy() == season
        rolling[str(int(season))] = _classification(
            oof.loc[m, "y"].to_numpy(), oof.loc[m, "pred_full_history"].to_numpy()
        )
    return {"pooled_season_increment": pooled, "matched": matched, "rolling_full_residual": rolling}


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    lam: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    xtr = np.asarray(x_train, np.float64)
    xte = np.asarray(x_test, np.float64)
    ytr = np.asarray(y_train, np.float64)
    med = np.nanmedian(xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    xtr = np.where(np.isfinite(xtr), xtr, med)
    xte = np.where(np.isfinite(xte), xte, med)
    mean = xtr.mean(axis=0)
    std = xtr.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    a = np.column_stack([np.ones(len(xtr)), xtr])
    b = np.column_stack([np.ones(len(xte)), xte])
    if weights is None:
        w = np.ones(len(a), np.float64)
    else:
        w = np.asarray(weights, np.float64)
        w = np.maximum(w, 1e-8)
        w = w / np.mean(w)
    reg = np.eye(a.shape[1], dtype=np.float64) * float(lam)
    reg[0, 0] = 0.0
    lhs = a.T @ (a * w[:, None]) + reg
    rhs = a.T @ (ytr * w)
    beta = np.linalg.solve(lhs, rhs)
    return b @ beta


def _pitcher_residual_table(oof: pd.DataFrame, pred_col: str = "pred_full_history") -> pd.DataFrame:
    tmp = oof[["season", "pitcher_id", "y", pred_col]].copy()
    tmp["residual"] = tmp["y"] - tmp[pred_col]
    grouped = tmp.groupby(["season", "pitcher_id"], sort=True, dropna=False)
    return grouped.agg(rows=("y", "size"), success=("y", "mean"), pred=(pred_col, "mean"), residual=("residual", "mean")).reset_index()


def _latent_state_summary(
    frame: pd.DataFrame,
    aux: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    season_col: str,
    target_col: str,
    min_pitches: int,
    lam: float,
) -> dict[str, Any]:
    profiles = _season_profiles(
        frame,
        aux,
        season_col=season_col,
        pitcher_col="pitcher_id",
        target_col=target_col,
    )
    profiles["pitcher_id"] = profiles["pitcher_id"].astype(str)
    pairs = _adjacent_pairs(profiles, season_col=season_col, pitcher_col="pitcher_id")
    pairs = pairs.loc[
        (pairs["current_pitch_count"] >= min_pitches) & (pairs["past_pitch_count"] >= min_pitches)
    ].reset_index(drop=True)
    correlations: dict[str, Any] = {}
    for name in STATE_NAMES:
        correlations[name] = {
            "overall": _corr(pairs[f"past_{name}"].to_numpy(), pairs[f"current_{name}"].to_numpy()),
            "by_transition": {},
        }
        for season in sorted(pairs["current_season"].astype(int).unique().tolist()):
            m = pairs["current_season"].astype(int).to_numpy() == int(season)
            correlations[name]["by_transition"][f"{season-1}->{season}"] = _corr(
                pairs.loc[m, f"past_{name}"].to_numpy(), pairs.loc[m, f"current_{name}"].to_numpy()
            )

    xcols = [*[f"past_{x}" for x in STATE_NAMES], "past_pitch_count"]
    pairs["past_pitch_count"] = np.log1p(pd.to_numeric(pairs["past_pitch_count"], errors="coerce"))
    next_success: dict[str, Any] = {}
    for test_season in (2022, 2023, 2024):
        tr = pairs.loc[pairs.current_season.astype(int) < test_season]
        te = pairs.loc[pairs.current_season.astype(int) == test_season]
        if len(tr) < 30 or len(te) < 10:
            continue
        pred = _ridge_predict(
            tr[xcols].to_numpy(np.float64),
            tr["current_success"].to_numpy(np.float64),
            te[xcols].to_numpy(np.float64),
            lam=lam,
            weights=tr["current_pitch_count"].to_numpy(np.float64) if "current_pitch_count" in tr else None,
        )
        w = te["current_pitch_count"].to_numpy(np.float64)
        y = te["current_success"].to_numpy(np.float64)
        persistence = te["past_success"].to_numpy(np.float64)
        next_success[str(test_season)] = {
            "rows": int(len(te)),
            "ridge_rmse": _rmse(y, pred, w),
            "persistence_rmse": _rmse(y, persistence, w),
            "gain_vs_persistence": _rmse(y, persistence, w) - _rmse(y, pred, w),
        }

    residuals = _pitcher_residual_table(oof)
    residuals["pitcher_id"] = residuals["pitcher_id"].astype(str)
    rtable = residuals.merge(
        pairs[["current_season", "pitcher_id", *xcols]],
        left_on=["season", "pitcher_id"],
        right_on=["current_season", "pitcher_id"],
        how="inner",
    )
    residual_prediction: dict[str, Any] = {}
    for test_season in (2023, 2024):
        tr = rtable.loc[rtable.season.astype(int) < test_season]
        te = rtable.loc[rtable.season.astype(int) == test_season]
        if len(tr) < 20 or len(te) < 10:
            continue
        pred = _ridge_predict(
            tr[xcols].to_numpy(np.float64),
            tr["residual"].to_numpy(np.float64),
            te[xcols].to_numpy(np.float64),
            lam=lam,
            weights=tr["rows"].to_numpy(np.float64),
        )
        y = te["residual"].to_numpy(np.float64)
        w = te["rows"].to_numpy(np.float64)
        baseline = np.zeros(len(te), np.float64)
        residual_prediction[str(test_season)] = {
            "pitchers": int(len(te)),
            "ridge_rmse": _rmse(y, pred, w),
            "zero_rmse": _rmse(y, baseline, w),
            "gain_vs_zero": _rmse(y, baseline, w) - _rmse(y, pred, w),
        }
    gains = [float(x["gain_vs_zero"]) for x in residual_prediction.values()]
    signal = "promising" if gains and sum(x > 0 for x in gains) >= max(1, len(gains) - 0) else "weak"
    return {
        "pairs": int(len(pairs)),
        "state_correlations": correlations,
        "next_success_rolling": next_success,
        "residual_prediction": residual_prediction,
        "signal": signal,
    }


def _find_trackman_path(cfg: dict[str, Any]) -> Path | None:
    for value in (cfg.get("trackman") or {}).get("candidate_paths", []):
        p = _resolve(value)
        if p.is_file():
            return p
    return None


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in frame.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _mechanical_columns(track: pd.DataFrame, *, max_cols: int, excluded: set[str]) -> list[str]:
    priority = [
        "rel_speed", "release_speed", "velocity", "zone_speed", "spin_rate",
        "horz_break", "horizontal_break", "induced_vert_break", "vert_break",
        "extension", "rel_height", "release_height", "rel_side", "release_side",
    ]
    chosen: list[str] = []
    lower = {str(c).lower(): str(c) for c in track.columns}
    for name in priority:
        if name in lower and lower[name] not in excluded:
            chosen.append(lower[name])
    tokens = ("speed", "velocity", "spin", "break", "release", "rel_", "extension", "height", "side")
    for col in track.columns:
        name = str(col)
        if name in excluded or name in chosen:
            continue
        low = name.lower()
        if any(tok in low for tok in tokens):
            chosen.append(name)
    valid: list[str] = []
    for col in chosen:
        values = pd.to_numeric(track[col], errors="coerce")
        if int(values.notna().sum()) >= 100:
            track[col] = values.astype(np.float32)
            valid.append(col)
        if len(valid) >= max_cols:
            break
    return valid


def _load_safe_2024(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["y"], np.float64), np.asarray(data["pred"], np.float64)


def _trackman_summary(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    season_col: str,
    exp: dict[str, Any],
    baseline_path: Path,
) -> dict[str, Any]:
    path = _find_trackman_path(exp)
    if path is None:
        return {"status": "skipped", "reason": "trackman_history.csv not found in configured paths", "signal": "unknown"}
    print(f"[C Trackman] loading {path}", flush=True)
    track = pd.read_csv(path, low_memory=False)
    pitcher_col = _first_column(track, ["pitcher_id", "pitcher", "pitcherid"])
    season_track = _first_column(track, ["season", "year"])
    if season_track is None:
        date_col = _first_column(track, ["game_date", "date", "pitch_date"])
        if date_col is not None:
            track["__season"] = pd.to_datetime(track[date_col], errors="coerce").dt.year
            season_track = "__season"
    if pitcher_col is None or season_track is None:
        return {
            "status": "skipped",
            "reason": f"could not resolve pitcher/season columns; columns={list(track.columns)[:30]}",
            "signal": "unknown",
        }
    mech = _mechanical_columns(
        track,
        max_cols=int((exp.get("trackman") or {}).get("max_mechanical_columns", 12)),
        excluded={pitcher_col, season_track},
    )
    if len(mech) < 2:
        return {"status": "skipped", "reason": f"too few mechanical columns resolved: {mech}", "signal": "unknown"}
    track[season_track] = pd.to_numeric(track[season_track], errors="coerce")
    track[pitcher_col] = track[pitcher_col].astype(str)
    track = track.loc[track[season_track].notna()].copy()
    track[season_track] = track[season_track].astype(int)
    grouped = track.groupby([season_track, pitcher_col], sort=True, dropna=False)
    agg = grouped[mech].agg(["mean", "std"])
    agg.columns = [f"mech_{stat}_{col}" for col, stat in agg.columns]
    agg = agg.reset_index()
    agg["trackman_n"] = grouped.size().to_numpy()
    min_n = int((exp.get("trackman") or {}).get("min_trackman_pitches", 50))
    agg = agg.loc[agg.trackman_n >= min_n].copy()
    agg["current_season"] = agg[season_track].astype(int) + 1
    agg = agg.rename(columns={pitcher_col: "pitcher_id"})
    agg["pitcher_id"] = agg["pitcher_id"].astype(str)
    std_cols = [x for x in agg.columns if x.startswith("mech_std_")]

    residuals = _pitcher_residual_table(oof)
    residuals["pitcher_id"] = residuals["pitcher_id"].astype(str)
    table = residuals.merge(
        agg[["current_season", "pitcher_id", "trackman_n", *std_cols]],
        left_on=["season", "pitcher_id"],
        right_on=["current_season", "pitcher_id"],
        how="inner",
    )
    ridge_lambda = float((exp.get("trackman") or {}).get("ridge_lambda", 10.0))
    rolling: dict[str, Any] = {}
    xcols = [*std_cols, "trackman_n"]
    table["trackman_n"] = np.log1p(table["trackman_n"].astype(float))
    for test_season in (2023, 2024):
        tr = table.loc[table.season.astype(int) < test_season]
        te = table.loc[table.season.astype(int) == test_season]
        if len(tr) < 20 or len(te) < 10:
            continue
        pred = _ridge_predict(
            tr[xcols].to_numpy(np.float64),
            tr["residual"].to_numpy(np.float64),
            te[xcols].to_numpy(np.float64),
            lam=ridge_lambda,
            weights=tr["rows"].to_numpy(np.float64),
        )
        y = te["residual"].to_numpy(np.float64)
        w = te["rows"].to_numpy(np.float64)
        zero = np.zeros(len(te), np.float64)
        rolling[str(test_season)] = {
            "pitchers": int(len(te)),
            "ridge_rmse": _rmse(y, pred, w),
            "zero_rmse": _rmse(y, zero, w),
            "gain_vs_zero": _rmse(y, zero, w) - _rmse(y, pred, w),
        }

    safe_y, safe_pred = _load_safe_2024(baseline_path)
    va = frame.loc[pd.to_numeric(frame[season_col], errors="raise").astype(int) == 2024].reset_index(drop=True)
    safe_direct: dict[str, Any] = {}
    if len(va) == len(safe_y):
        safe_rows = pd.DataFrame(
            {
                "pitcher_id": va["pitcher_id"].astype(str).to_numpy(),
                "residual": safe_y - safe_pred,
            }
        )
        safe_pitch = safe_rows.groupby("pitcher_id", sort=True).agg(rows=("residual", "size"), residual=("residual", "mean")).reset_index()
        prior24 = agg.loc[agg.current_season.astype(int) == 2024, ["pitcher_id", *std_cols]].copy()
        joined = safe_pitch.merge(prior24, on="pitcher_id", how="inner")
        ranks: list[tuple[str, float]] = []
        for col in std_cols:
            c = _corr(joined[col].to_numpy(), joined["residual"].to_numpy())
            if c is not None:
                ranks.append((col, c))
        ranks.sort(key=lambda kv: abs(kv[1]), reverse=True)
        safe_direct = {
            "pitchers": int(len(joined)),
            "top_abs_correlations": [{"feature": k, "corr": float(v)} for k, v in ranks[:10]],
        }
    gains = [float(x["gain_vs_zero"]) for x in rolling.values()]
    signal = "promising" if gains and sum(x > 0 for x in gains) == len(gains) else "weak"
    return {
        "status": "ok",
        "path": str(path),
        "rows": int(len(track)),
        "pitcher_seasons": int(len(agg)),
        "mechanical_columns": mech,
        "stability_features": std_cols,
        "rolling_residual_prediction": rolling,
        "safe2024_univariate": safe_direct,
        "signal": signal,
    }


def _scorecard(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], d: dict[str, Any]) -> dict[str, Any]:
    return {
        "A_conditional_shift": {
            "signal": a["pooled_season_increment"]["signal"],
            "season_feature_brier_gain": a["pooled_season_increment"]["brier_gain_from_season"],
        },
        "B_latent_pitcher_state": {
            "signal": b.get("signal", "unknown"),
            "residual_gains": {k: v["gain_vs_zero"] for k, v in b.get("residual_prediction", {}).items()},
        },
        "C_trackman_mechanics": {
            "signal": c.get("signal", "unknown"),
            "status": c.get("status", "unknown"),
            "residual_gains": {k: v["gain_vs_zero"] for k, v in c.get("rolling_residual_prediction", {}).items()},
        },
        "D_cold_start": {
            "signal": d.get("signal", "unknown"),
            "corner_best_models": d.get("corner_best_models", []),
            "corner_full_gain_range": d.get("corner_full_gain_range", 0.0),
        },
    }


def _report(result: dict[str, Any]) -> str:
    sc = result["scorecard"]
    lines = [
        "# Four-axis research audit",
        "",
        "> This run is diagnostic. It is designed to decide which research family deserves the next modeling cycle, not to optimize SAFE982 locally.",
        "",
        "## Scorecard",
        "",
        "| axis | question | signal | key diagnostic |",
        "|---|---|---|---|",
        f"| A | Conditional shift: does season add information after X? | **{sc['A_conditional_shift']['signal']}** | season-feature Brier gain `{sc['A_conditional_shift']['season_feature_brier_gain']:+.9f}` |",
        f"| B | Latent pitcher state: does prior state explain future SAFE-like residual? | **{sc['B_latent_pitcher_state']['signal']}** | residual gains `{sc['B_latent_pitcher_state']['residual_gains']}` |",
        f"| C | Trackman mechanics: does prior mechanical stability explain residual? | **{sc['C_trackman_mechanics']['signal']}** | status `{sc['C_trackman_mechanics']['status']}`, residual gains `{sc['C_trackman_mechanics']['residual_gains']}` |",
        f"| D | Cold start: is information value heterogeneous by pitcher/batter history? | **{sc['D_cold_start']['signal']}** | corner best models `{sc['D_cold_start']['corner_best_models']}`, full-gain range `{sc['D_cold_start']['corner_full_gain_range']:.9f}` |",
        "",
        "## A. Conditional shift",
        "",
    ]
    a = result["A_conditional_shift"]
    p = a["pooled_season_increment"]
    lines += [
        f"- pooled deterministic holdout Brier without season: `{p['without_season']['brier']:.9f}`",
        f"- pooled deterministic holdout Brier with season: `{p['with_season']['brier']:.9f}`",
        f"- gain from explicit season: `{p['brier_gain_from_season']:+.9f}`",
        "- rolling full-history residuals:",
    ]
    for season, m in a["rolling_full_residual"].items():
        lines.append(f"  - {season}: Brier `{m['brier']:.9f}`, residual mean `{m['residual_mean']:+.6f}`")
    lines.append("- matched same-pitcher/context transitions:")
    for pair, m in a["matched"]["pairs"].items():
        lines.append(
            f"  - {pair}: cells `{m['matched_cells']:,}`, matched delta `{m['matched_weighted_delta']:+.6f}`, raw delta `{m['raw_rate_delta']:+.6f}`"
        )

    lines += ["", "## B. Latent pitcher state", ""]
    b = result["B_latent_pitcher_state"]
    lines.append(f"- qualifying adjacent pitcher-seasons: `{b['pairs']:,}`")
    for state, item in b["state_correlations"].items():
        lines.append(f"- `{state}` adjacent-season correlation: `{item['overall']}`")
    lines.append("- rolling next-season success prediction:")
    for season, m in b["next_success_rolling"].items():
        lines.append(
            f"  - {season}: state ridge `{m['ridge_rmse']:.6f}`, persistence `{m['persistence_rmse']:.6f}`, gain `{m['gain_vs_persistence']:+.6f}`"
        )
    lines.append("- rolling residual prediction from prior state:")
    for season, m in b["residual_prediction"].items():
        lines.append(
            f"  - {season}: ridge `{m['ridge_rmse']:.6f}`, zero `{m['zero_rmse']:.6f}`, gain `{m['gain_vs_zero']:+.6f}`"
        )

    lines += ["", "## C. Trackman mechanical stability", ""]
    c = result["C_trackman_mechanics"]
    if c.get("status") != "ok":
        lines.append(f"- skipped: {c.get('reason')}")
    else:
        lines.append(f"- source: `{c['path']}`")
        lines.append(f"- mechanical columns: `{', '.join(c['mechanical_columns'])}`")
        for season, m in c["rolling_residual_prediction"].items():
            lines.append(
                f"- {season} residual: mechanics ridge `{m['ridge_rmse']:.6f}`, zero `{m['zero_rmse']:.6f}`, gain `{m['gain_vs_zero']:+.6f}`"
            )
        top = c.get("safe2024_univariate", {}).get("top_abs_correlations", [])
        if top:
            lines.append("- strongest prior-mechanics correlations with SAFE982 2024 pitcher residual:")
            for item in top[:5]:
                lines.append(f"  - `{item['feature']}`: `{item['corr']:+.4f}`")

    lines += ["", "## D. Cold/rich information value", ""]
    d = result["D_cold_start"]
    lines += [
        "| cohort | rows | best model | context Brier | pitcher-history gain | batter-history gain | full-history gain |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for cell, item in d["cells"].items():
        if cell not in {"P:cold|B:cold", "P:cold|B:rich", "P:rich|B:cold", "P:rich|B:rich"}:
            continue
        gains = item["gain_vs_context"]
        lines.append(
            f"| `{cell}` | {item['rows']:,} | `{item['best_model']}` | {item['models']['context']['brier']:.9f} | "
            f"{gains.get('pitcher_history', float('nan')):+.9f} | {gains.get('batter_history', float('nan')):+.9f} | {gains.get('full_history', float('nan')):+.9f} |"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "Do not promote any model from this audit directly. Use the scorecard to select at most one or two axes for the next modeling cycle. A positive diagnostic should also survive the rolling 2022/2023/2024 direction before 2024-only tuning is considered.",
    ]
    return "\n".join(lines) + "\n"


def run(config_path: str | Path) -> dict[str, Any]:
    exp = _load_yaml(config_path)
    base_cfg = load_config(_resolve(exp["baseline_config"]))
    out = _resolve(exp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("[Research Audit] preparing SAFE-compatible frame", flush=True)
    data = prepare(base_cfg)
    frame = data.frame.reset_index(drop=True)
    aux = data.aux.reset_index(drop=True)
    season_col = base_cfg["data"]["season_col"]
    target_col = base_cfg["data"]["target_col"]
    groups = _feature_groups(frame, season_col)
    folds = [int(x) for x in exp.get("rolling_folds", [2022, 2023, 2024])]

    print("\n=== D first: rolling information-value models (also creates residuals for A/B/C) ===", flush=True)
    oof, rolling_metrics = _rolling_information_audit(
        frame,
        season_col=season_col,
        target_col=target_col,
        folds=folds,
        groups=groups,
        exp=exp,
        out=out,
    )
    d = _cold_start_summary(
        oof,
        cold=int(exp.get("cold_threshold", 50)),
        rich=int(exp.get("rich_threshold", 200)),
        variants=list(groups),
    )
    d["rolling_model_metrics"] = rolling_metrics

    print("\n=== A: conditional-shift audit ===", flush=True)
    a = _conditional_shift_summary(
        frame,
        oof,
        season_col=season_col,
        target_col=target_col,
        full_features=groups["full_history"],
        exp=exp,
    )

    print("\n=== B: latent pitcher-state audit ===", flush=True)
    b = _latent_state_summary(
        frame,
        aux,
        oof,
        season_col=season_col,
        target_col=target_col,
        min_pitches=int(exp.get("min_pitcher_season_pitches", 100)),
        lam=float((exp.get("latent_state") or {}).get("ridge_lambda", 5.0)),
    )

    print("\n=== C: Trackman mechanical-stability audit ===", flush=True)
    c = _trackman_summary(
        frame,
        oof,
        season_col=season_col,
        exp=exp,
        baseline_path=_resolve(exp["baseline_predictions"]),
    )

    scorecard = _scorecard(a, b, c, d)
    result = {
        "scorecard": scorecard,
        "A_conditional_shift": a,
        "B_latent_pitcher_state": b,
        "C_trackman_mechanics": c,
        "D_cold_start": d,
        "feature_groups": {k: {"count": len(v), "features": v} for k, v in groups.items()},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(out / "metrics.json", result)
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    print("\n[Four-axis audit complete]", flush=True)
    for key, item in scorecard.items():
        print(f"{key}: signal={item['signal']}", flush=True)
    print(f"elapsed={result['elapsed_seconds']/60.0:.1f} min", flush=True)
    print(f"report={out / 'report.md'}", flush=True)
    return result
