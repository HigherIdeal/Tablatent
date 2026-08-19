from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitaboost.config import load_config
from bitaboost.features import prepare
from bitaboost.research_audit.four_axes import (
    _brier,
    _classification,
    _feature_groups,
    _find_trackman_path,
    _mechanical_columns,
    _ridge_predict,
    _rmse,
    _rolling_information_audit,
    _root,
    _resolve,
    _write_json,
    _load_yaml,
)


def _edges_to_labels(values: np.ndarray, edges: list[int]) -> np.ndarray:
    x = np.asarray(values, np.float64)
    edges = [int(v) for v in edges]
    if not edges or edges[0] != 0:
        raise ValueError("experience_edges must start at 0")
    labels = np.full(len(x), "NA", dtype=object)
    for i in range(len(edges)):
        lo = edges[i]
        if i + 1 < len(edges):
            hi = edges[i + 1]
            if hi == lo + 1:
                label = str(lo)
            else:
                label = f"{lo}-{hi-1}"
            mask = (x >= lo) & (x < hi)
        else:
            label = f"{lo}+"
            mask = x >= lo
        labels[mask] = label
    missing = pd.isna(labels)
    if np.any(missing):
        labels[missing] = "NA"
    return labels.astype(str)


def _attach_oof_context(oof: pd.DataFrame, frame: pd.DataFrame, season_col: str) -> pd.DataFrame:
    cols = ["row_id", "game_type", "asof_pitcher_n", "asof_batter_n"]
    missing = [x for x in cols if x not in frame.columns]
    if missing:
        raise RuntimeError(f"Cycle2 missing context columns: {missing}")
    lookup = frame[[season_col, *cols]].copy()
    lookup["row_id"] = lookup["row_id"].astype(str)
    lookup = lookup.rename(columns={season_col: "_season_frame"})
    out = oof.merge(lookup, on="row_id", how="left", validate="one_to_one", suffixes=("", "_frame"))
    if out["_season_frame"].isna().any():
        raise RuntimeError("Cycle2 OOF row_id merge lost rows")
    if not np.array_equal(
        out["season"].astype(int).to_numpy(),
        out["_season_frame"].astype(int).to_numpy(),
    ):
        raise RuntimeError("Cycle2 OOF/frame season mismatch")
    out["game_type"] = out["game_type"].astype(str)
    out["asof_pitcher_n"] = pd.to_numeric(out["asof_pitcher_n_frame"], errors="coerce").fillna(
        pd.to_numeric(out["asof_pitcher_n"], errors="coerce")
    )
    out["asof_batter_n"] = pd.to_numeric(out["asof_batter_n_frame"], errors="coerce").fillna(
        pd.to_numeric(out["asof_batter_n"], errors="coerce")
    )
    return out.drop(columns=["_season_frame", "asof_pitcher_n_frame", "asof_batter_n_frame"])


def _group_keys(block: pd.DataFrame, variant: str, exp_labels: np.ndarray) -> np.ndarray:
    if variant == "global":
        return np.repeat("__GLOBAL__", len(block)).astype(str)
    if variant == "game_type":
        return block["game_type"].astype(str).to_numpy()
    if variant == "pitcher_experience":
        return exp_labels.astype(str)
    if variant == "game_type_x_pitcher_experience":
        return np.char.add(
            np.char.add(block["game_type"].astype(str).to_numpy().astype(str), "|"),
            exp_labels.astype(str),
        )
    raise ValueError(f"unknown temporal correction variant: {variant}")


def _fit_residual_map(
    source: pd.DataFrame,
    *,
    pred_col: str,
    variant: str,
    exp_edges: list[int],
    shrink_k: float,
    min_group_rows: int,
) -> dict[str, Any]:
    y = source["y"].to_numpy(np.float64)
    p = source[pred_col].to_numpy(np.float64)
    residual = y - p
    global_mean = float(np.mean(residual))
    exp_labels = _edges_to_labels(source["asof_pitcher_n"].to_numpy(), exp_edges)
    keys = _group_keys(source, variant, exp_labels)
    temp = pd.DataFrame({"key": keys, "residual": residual})
    grouped = temp.groupby("key", sort=True).agg(n=("residual", "size"), mean=("residual", "mean"))
    mapping: dict[str, float] = {}
    rows: dict[str, int] = {}
    for key, item in grouped.iterrows():
        n = int(item["n"])
        mean = float(item["mean"])
        if n < int(min_group_rows):
            value = global_mean
        else:
            value = float((n * mean + float(shrink_k) * global_mean) / (n + float(shrink_k)))
        mapping[str(key)] = value
        rows[str(key)] = n
    return {
        "global_mean": global_mean,
        "mapping": mapping,
        "rows": rows,
    }


def _apply_residual_map(
    target: pd.DataFrame,
    fitted: dict[str, Any],
    *,
    pred_col: str,
    variant: str,
    exp_edges: list[int],
    alpha: float,
) -> np.ndarray:
    exp_labels = _edges_to_labels(target["asof_pitcher_n"].to_numpy(), exp_edges)
    keys = _group_keys(target, variant, exp_labels)
    global_mean = float(fitted["global_mean"])
    mapping = fitted["mapping"]
    correction = np.asarray([float(mapping.get(str(k), global_mean)) for k in keys], np.float64)
    pred = target[pred_col].to_numpy(np.float64)
    return np.clip(pred + float(alpha) * correction, 0.0, 1.0)


def _temporal_transfer(oof: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg["temporal_transfer"]
    folds = sorted(int(x) for x in oof["season"].unique())
    pred_col = "pred_full_history"
    results: dict[str, Any] = {"transitions": {}, "candidates": {}}
    variants = [str(x) for x in tc["variants"]]
    alphas = [float(x) for x in tc["correction_alphas"]]
    shrinkages = [float(x) for x in tc["shrinkage_k"]]
    edges = [int(x) for x in tc["experience_edges"]]
    min_rows = int(tc["min_group_rows"])

    for source_season, target_season in zip(folds[:-1], folds[1:]):
        source = oof.loc[oof.season.astype(int) == source_season].reset_index(drop=True)
        target = oof.loc[oof.season.astype(int) == target_season].reset_index(drop=True)
        y = target["y"].to_numpy(np.float64)
        base = target[pred_col].to_numpy(np.float64)
        base_brier = _brier(y, base)
        transition_key = f"{source_season}->{target_season}"
        results["transitions"][transition_key] = {
            "source_residual_mean": float(
                np.mean(source["y"].to_numpy(np.float64) - source[pred_col].to_numpy(np.float64))
            ),
            "target_residual_mean": float(np.mean(y - base)),
            "baseline_brier": base_brier,
            "trials": {},
        }
        for variant in variants:
            k_values = [0.0] if variant == "global" else shrinkages
            for k in k_values:
                fitted = _fit_residual_map(
                    source,
                    pred_col=pred_col,
                    variant=variant,
                    exp_edges=edges,
                    shrink_k=k,
                    min_group_rows=min_rows,
                )
                for alpha in alphas:
                    pred = _apply_residual_map(
                        target,
                        fitted,
                        pred_col=pred_col,
                        variant=variant,
                        exp_edges=edges,
                        alpha=alpha,
                    )
                    name = f"{variant}|k={k:g}|a={alpha:g}"
                    brier = _brier(y, pred)
                    gain = base_brier - brier
                    results["transitions"][transition_key]["trials"][name] = {
                        "brier": brier,
                        "gain": gain,
                    }
                    results["candidates"].setdefault(name, []).append(
                        {"transition": transition_key, "gain": gain, "brier": brier}
                    )

    ranking: list[dict[str, Any]] = []
    for name, vals in results["candidates"].items():
        gains = [float(x["gain"]) for x in vals]
        ranking.append(
            {
                "candidate": name,
                "mean_gain": float(np.mean(gains)),
                "min_gain": float(np.min(gains)),
                "positive_transitions": int(sum(x > 0.0 for x in gains)),
                "transitions": len(gains),
                "gains": {x["transition"]: float(x["gain"]) for x in vals},
            }
        )
    ranking.sort(key=lambda x: (x["positive_transitions"], x["min_gain"], x["mean_gain"]), reverse=True)
    results["ranking"] = ranking
    stable = [
        x
        for x in ranking
        if x["positive_transitions"] == x["transitions"] and x["mean_gain"] > 0.0
    ]
    results["best_stable"] = stable[0] if stable else None
    latest_transition = f"{folds[-2]}->{folds[-1]}" if len(folds) >= 2 else None
    recent_rank = []
    if latest_transition is not None:
        for item in ranking:
            gain = float(item["gains"].get(latest_transition, float("-inf")))
            if np.isfinite(gain):
                recent_rank.append({"candidate": item["candidate"], "gain": gain})
        recent_rank.sort(key=lambda x: x["gain"], reverse=True)
    results["best_recent"] = recent_rank[0] if recent_rank else None
    if stable:
        results["signal"] = "transferable"
    elif results["best_recent"] is not None and float(results["best_recent"]["gain"]) > 0.0:
        results["signal"] = "recent_only"
    else:
        results["signal"] = "unstable"
    return results


def _canonical_id(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    use_numeric = numeric.notna()
    out = s.copy()
    if use_numeric.any():
        rounded = numeric.loc[use_numeric].round()
        integers = (numeric.loc[use_numeric] - rounded).abs() < 1e-9
        idx_int = integers.index[integers]
        idx_float = integers.index[~integers]
        out.loc[idx_int] = rounded.loc[idx_int].astype("Int64").astype("string")
        out.loc[idx_float] = numeric.loc[idx_float].astype(str)
    return out.fillna("<NA>").astype(str)


def _choose_id_columns(
    frame: pd.DataFrame,
    track: pd.DataFrame,
    tc: dict[str, Any],
) -> dict[str, Any]:
    train_candidates = [x for x in tc["train_pitcher_id_candidates"] if x in frame.columns]
    track_candidates = [x for x in tc["trackman_pitcher_id_candidates"] if x in track.columns]
    if not train_candidates or not track_candidates:
        return {
            "status": "failed",
            "reason": f"id candidates missing train={train_candidates} track={track_candidates}",
        }
    scores: list[dict[str, Any]] = []
    for a in train_candidates:
        va = set(_canonical_id(frame[a]).unique().tolist())
        va.discard("<NA>")
        for b in track_candidates:
            vb = set(_canonical_id(track[b]).unique().tolist())
            vb.discard("<NA>")
            overlap = len(va & vb)
            denom = max(1, min(len(va), len(vb)))
            frac = overlap / denom
            scores.append(
                {
                    "train_col": a,
                    "track_col": b,
                    "train_unique": len(va),
                    "track_unique": len(vb),
                    "overlap": overlap,
                    "overlap_fraction_min_unique": float(frac),
                }
            )
    scores.sort(key=lambda x: (x["overlap_fraction_min_unique"], x["overlap"]), reverse=True)
    best = scores[0]
    return {"status": "ok", "best": best, "all": scores}


def _first_present(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _within_pitch_type_dispersion(
    track: pd.DataFrame,
    *,
    season_col: str,
    pitcher_col: str,
    pitch_type_col: str,
    mech: list[str],
    min_pitch_type_pitches: int,
) -> pd.DataFrame:
    keys = [season_col, pitcher_col, pitch_type_col]
    grouped = track.groupby(keys, sort=True, dropna=False)
    counts = grouped.size().rename("_n").reset_index()
    std = grouped[mech].std().reset_index()
    table = counts.merge(std, on=keys, how="inner")
    table = table.loc[table["_n"] >= int(min_pitch_type_pitches)].copy()
    out_rows: list[dict[str, Any]] = []
    for (season, pitcher), block in table.groupby([season_col, pitcher_col], sort=True, dropna=False):
        row: dict[str, Any] = {season_col: season, pitcher_col: pitcher}
        w = block["_n"].to_numpy(np.float64)
        for col in mech:
            values = pd.to_numeric(block[col], errors="coerce").to_numpy(np.float64)
            keep = np.isfinite(values)
            row[f"mech_within_std_{col}"] = (
                float(np.average(values[keep], weights=w[keep])) if keep.any() else np.nan
            )
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _pitcher_residuals(oof: pd.DataFrame, id_col: str = "pitcher_id") -> pd.DataFrame:
    temp = oof[["season", id_col, "y", "pred_full_history"]].copy()
    temp = temp.rename(columns={id_col: "pitcher_id"})
    temp["pitcher_id"] = _canonical_id(temp["pitcher_id"])
    temp["residual"] = temp["y"].to_numpy(np.float64) - temp["pred_full_history"].to_numpy(np.float64)
    return (
        temp.groupby(["season", "pitcher_id"], sort=True)
        .agg(rows=("y", "size"), residual=("residual", "mean"))
        .reset_index()
    )


def _mechanics_trackman(frame: pd.DataFrame, oof: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    tc = cfg["trackman"]
    path = _find_trackman_path(cfg)
    if path is None:
        return {"status": "skipped", "reason": "Trackman file not found", "signal": "unknown"}
    print(f"[C2-C] loading Trackman: {path}", flush=True)
    track = pd.read_csv(path, low_memory=False)
    id_diag = _choose_id_columns(frame, track, tc)
    if id_diag["status"] != "ok":
        return {"status": "skipped", "reason": id_diag["reason"], "id_diagnostic": id_diag, "signal": "unknown"}
    best = id_diag["best"]
    min_overlap = float(tc.get("min_id_overlap_fraction", 0.05))
    if float(best["overlap_fraction_min_unique"]) < min_overlap:
        return {
            "status": "skipped",
            "reason": f"best ID overlap too small: {best}",
            "id_diagnostic": id_diag,
            "signal": "unknown",
        }
    train_id = str(best["train_col"])
    track_id = str(best["track_col"])
    season_track = _first_present(track, [str(x) for x in tc["season_candidates"]])
    if season_track is None:
        return {"status": "skipped", "reason": "Trackman season column missing", "id_diagnostic": id_diag, "signal": "unknown"}
    pitch_type_col = _first_present(track, [str(x) for x in tc["pitch_type_candidates"]])

    track[track_id] = _canonical_id(track[track_id])
    track[season_track] = pd.to_numeric(track[season_track], errors="coerce")
    track = track.loc[track[season_track].notna()].copy()
    track[season_track] = track[season_track].astype(int)

    mech = _mechanical_columns(
        track,
        max_cols=int(tc["max_mechanical_columns"]),
        excluded={track_id, season_track},
    )
    if len(mech) < 2:
        return {
            "status": "skipped",
            "reason": f"too few mechanical columns: {mech}",
            "id_diagnostic": id_diag,
            "signal": "unknown",
        }

    grouped = track.groupby([season_track, track_id], sort=True, dropna=False)
    overall = grouped[mech].agg(["mean", "std"])
    overall.columns = [f"mech_{stat}_{col}" for col, stat in overall.columns]
    overall = overall.reset_index()
    overall["trackman_n"] = grouped.size().to_numpy()
    overall = overall.loc[overall["trackman_n"] >= int(tc["min_trackman_pitches"])].copy()

    if pitch_type_col is not None:
        within = _within_pitch_type_dispersion(
            track,
            season_col=season_track,
            pitcher_col=track_id,
            pitch_type_col=pitch_type_col,
            mech=mech,
            min_pitch_type_pitches=int(tc["min_pitch_type_pitches"]),
        )
        overall = overall.merge(within, on=[season_track, track_id], how="left")

    overall = overall.rename(columns={track_id: "pitcher_id"})
    overall["pitcher_id"] = _canonical_id(overall["pitcher_id"])
    overall["current_season"] = overall[season_track].astype(int) + 1
    overall["log_trackman_n"] = np.log1p(overall["trackman_n"].astype(float))

    oof_for_track = oof.copy()
    residual_id_col = "pitcher_id"
    if train_id != "pitcher_id":
        if "row_id" not in frame.columns:
            return {
                "status": "skipped",
                "reason": f"selected train ID {train_id} requires row_id mapping but row_id is missing",
                "id_diagnostic": id_diag,
                "signal": "unknown",
            }
        id_lookup = frame[["row_id", train_id]].copy()
        id_lookup["row_id"] = id_lookup["row_id"].astype(str)
        oof_for_track = oof_for_track.merge(
            id_lookup,
            on="row_id",
            how="left",
            validate="one_to_one",
            suffixes=("", "_track_id"),
        )
        if oof_for_track[train_id].isna().any():
            return {
                "status": "skipped",
                "reason": f"selected train ID {train_id} could not be mapped to every OOF row",
                "id_diagnostic": id_diag,
                "signal": "unknown",
            }
        residual_id_col = train_id

    residuals = _pitcher_residuals(oof_for_track, residual_id_col)
    table = residuals.merge(
        overall,
        left_on=["season", "pitcher_id"],
        right_on=["current_season", "pitcher_id"],
        how="inner",
    )

    mean_cols = [x for x in overall.columns if x.startswith("mech_mean_")]
    std_cols = [x for x in overall.columns if x.startswith("mech_std_")]
    within_cols = [x for x in overall.columns if x.startswith("mech_within_std_")]
    feature_sets = {
        "overall_std": [*std_cols, "log_trackman_n"],
        "within_pitch_type_std": [*within_cols, "log_trackman_n"] if within_cols else [],
        "mean_plus_std": [*mean_cols, *std_cols, "log_trackman_n"],
        "all_mechanics": [*mean_cols, *std_cols, *within_cols, "log_trackman_n"],
    }
    feature_sets = {k: v for k, v in feature_sets.items() if len(v) >= 2}

    rolling: dict[str, Any] = {}
    ridge_lambda = float(tc["ridge_lambda"])
    for test_season in sorted(int(x) for x in table["season"].unique()):
        if test_season < 2022:
            continue
        tr = table.loc[table.season.astype(int) < test_season]
        te = table.loc[table.season.astype(int) == test_season]
        if len(tr) < 20 or len(te) < 10:
            continue
        y = te["residual"].to_numpy(np.float64)
        w = te["rows"].to_numpy(np.float64)
        zero_rmse = _rmse(y, np.zeros(len(te)), w)
        fold: dict[str, Any] = {
            "pitchers": int(len(te)),
            "zero_rmse": zero_rmse,
            "feature_sets": {},
        }
        for name, cols in feature_sets.items():
            pred = _ridge_predict(
                tr[cols].to_numpy(np.float64),
                tr["residual"].to_numpy(np.float64),
                te[cols].to_numpy(np.float64),
                lam=ridge_lambda,
                weights=tr["rows"].to_numpy(np.float64),
            )
            rmse = _rmse(y, pred, w)
            fold["feature_sets"][name] = {
                "rmse": rmse,
                "gain_vs_zero": zero_rmse - rmse,
            }
        rolling[str(test_season)] = fold

    ranking: list[dict[str, Any]] = []
    for name in feature_sets:
        gains = []
        for fold in rolling.values():
            if name in fold["feature_sets"]:
                gains.append(float(fold["feature_sets"][name]["gain_vs_zero"]))
        if gains:
            ranking.append(
                {
                    "feature_set": name,
                    "mean_gain": float(np.mean(gains)),
                    "min_gain": float(np.min(gains)),
                    "positive_folds": int(sum(x > 0 for x in gains)),
                    "folds": len(gains),
                }
            )
    ranking.sort(key=lambda x: (x["positive_folds"], x["min_gain"], x["mean_gain"]), reverse=True)
    stable = [x for x in ranking if x["positive_folds"] == x["folds"] and x["mean_gain"] > 0]
    return {
        "status": "ok",
        "path": str(path),
        "id_diagnostic": id_diag,
        "selected_train_id": train_id,
        "selected_trackman_id": track_id,
        "pitch_type_col": pitch_type_col,
        "mechanical_columns": mech,
        "pitcher_seasons": int(len(overall)),
        "matched_pitcher_seasons": int(len(table)),
        "rolling": rolling,
        "ranking": ranking,
        "best_stable": stable[0] if stable else None,
        "signal": "promising" if stable else "weak",
    }


def _model_briers(block: pd.DataFrame, variants: list[str]) -> dict[str, float]:
    y = block["y"].to_numpy(np.float64)
    return {name: _brier(y, block[f"pred_{name}"].to_numpy(np.float64)) for name in variants}


def _router_labels(block: pd.DataFrame, axis: str, edges: list[int]) -> np.ndarray:
    p = _edges_to_labels(block["asof_pitcher_n"].to_numpy(), edges)
    b = _edges_to_labels(block["asof_batter_n"].to_numpy(), edges)
    if axis == "pitcher":
        return p
    if axis == "batter":
        return b
    if axis == "cross":
        return np.char.add(np.char.add(p.astype(str), "|"), b.astype(str))
    raise ValueError(axis)


def _fit_router(
    source: pd.DataFrame,
    *,
    variants: list[str],
    axis: str,
    edges: list[int],
    min_rows: int,
) -> dict[str, Any]:
    global_scores = _model_briers(source, variants)
    global_best = min(global_scores, key=global_scores.get)
    labels = _router_labels(source, axis, edges)
    mapping: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        block = source.loc[mask]
        if len(block) < int(min_rows):
            mapping[str(label)] = global_best
            diagnostics[str(label)] = {
                "rows": int(len(block)),
                "selected": global_best,
                "fallback": True,
            }
            continue
        scores = _model_briers(block, variants)
        best = min(scores, key=scores.get)
        mapping[str(label)] = best
        diagnostics[str(label)] = {
            "rows": int(len(block)),
            "selected": best,
            "fallback": False,
            "briers": scores,
        }
    return {
        "global_best": global_best,
        "global_briers": global_scores,
        "mapping": mapping,
        "cohorts": diagnostics,
    }


def _apply_router(
    target: pd.DataFrame,
    router: dict[str, Any],
    *,
    variants: list[str],
    axis: str,
    edges: list[int],
) -> tuple[np.ndarray, dict[str, int]]:
    labels = _router_labels(target, axis, edges)
    global_best = str(router["global_best"])
    mapping = router["mapping"]
    selected = np.asarray([str(mapping.get(str(x), global_best)) for x in labels], dtype=object)
    pred = np.empty(len(target), np.float64)
    counts: dict[str, int] = {}
    for name in variants:
        mask = selected == name
        if mask.any():
            pred[mask] = target.loc[mask, f"pred_{name}"].to_numpy(np.float64)
            counts[name] = int(mask.sum())
    return pred, counts


def _fine_information_table(oof: pd.DataFrame, variants: list[str], edges: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in ("pitcher", "batter", "cross"):
        labels = _router_labels(oof, axis, edges)
        rows: dict[str, Any] = {}
        for label in sorted(set(labels.tolist())):
            block = oof.loc[labels == label]
            scores = _model_briers(block, variants)
            context = float(scores["context"])
            best = min(scores, key=scores.get)
            rows[str(label)] = {
                "rows": int(len(block)),
                "best_model": best,
                "briers": scores,
                "gains_vs_context": {
                    name: context - float(score) for name, score in scores.items() if name != "context"
                },
            }
        result[axis] = rows
    return result


def _causal_reliability(oof: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    rc = cfg["reliability"]
    variants = [str(x) for x in rc["variants"]]
    edges = [int(x) for x in rc["experience_edges"]]
    min_rows = int(rc["min_source_rows"])
    axes = [str(x) for x in rc["router_axes"]]
    folds = sorted(int(x) for x in oof.season.unique())
    transitions: dict[str, Any] = {}
    summary: dict[str, list[float]] = {axis: [] for axis in axes}

    for source_season, target_season in zip(folds[:-1], folds[1:]):
        source = oof.loc[oof.season.astype(int) == source_season].reset_index(drop=True)
        target = oof.loc[oof.season.astype(int) == target_season].reset_index(drop=True)
        y = target["y"].to_numpy(np.float64)
        full = target["pred_full_history"].to_numpy(np.float64)
        full_brier = _brier(y, full)
        key = f"{source_season}->{target_season}"
        transitions[key] = {"full_history_brier": full_brier, "routers": {}}
        for axis in axes:
            router = _fit_router(
                source,
                variants=variants,
                axis=axis,
                edges=edges,
                min_rows=min_rows,
            )
            pred, counts = _apply_router(
                target,
                router,
                variants=variants,
                axis=axis,
                edges=edges,
            )
            brier = _brier(y, pred)
            gain = full_brier - brier
            summary[axis].append(gain)
            transitions[key]["routers"][axis] = {
                "brier": brier,
                "gain_vs_full_history": gain,
                "selected_counts": counts,
                "source_global_best": router["global_best"],
                "source_cohorts": router["cohorts"],
            }

    axes_summary: dict[str, Any] = {}
    for axis, gains in summary.items():
        axes_summary[axis] = {
            "mean_gain": float(np.mean(gains)) if gains else 0.0,
            "min_gain": float(np.min(gains)) if gains else 0.0,
            "positive_transitions": int(sum(x > 0 for x in gains)),
            "transitions": len(gains),
            "gains": gains,
        }
    stable = [
        (axis, item)
        for axis, item in axes_summary.items()
        if item["transitions"] > 0
        and item["positive_transitions"] == item["transitions"]
        and item["mean_gain"] > 0
    ]
    stable.sort(key=lambda kv: (kv[1]["min_gain"], kv[1]["mean_gain"]), reverse=True)
    return {
        "fine_information": _fine_information_table(oof, variants, edges),
        "transitions": transitions,
        "axes_summary": axes_summary,
        "best_stable": {"axis": stable[0][0], **stable[0][1]} if stable else None,
        "signal": "causal_router" if stable else "full_history_default",
    }


def _scorecard(a: dict[str, Any], c: dict[str, Any], d: dict[str, Any]) -> dict[str, Any]:
    return {
        "A_temporal_transfer": {
            "signal": a["signal"],
            "best_stable": a.get("best_stable"),
            "best_recent": a.get("best_recent"),
        },
        "C_trackman_mechanics": {
            "signal": c.get("signal", "unknown"),
            "status": c.get("status", "unknown"),
            "best_stable": c.get("best_stable"),
            "id_best": (c.get("id_diagnostic") or {}).get("best"),
        },
        "D_reliability": {
            "signal": d["signal"],
            "best_stable": d.get("best_stable"),
        },
    }


def _report(result: dict[str, Any]) -> str:
    sc = result["scorecard"]
    lines = [
        "# Cycle 2: temporal transfer, Trackman mechanics, and reliability",
        "",
        "> Diagnostic only. Nothing in this report is a deployment recipe until it survives causal historical transfer.",
        "",
        "## Scorecard",
        "",
        "| axis | signal | best stable diagnostic |",
        "|---|---|---|",
        f"| A temporal transfer | **{sc['A_temporal_transfer']['signal']}** | stable=`{sc['A_temporal_transfer']['best_stable']}`, recent=`{sc['A_temporal_transfer']['best_recent']}` |",
        f"| C Trackman mechanics | **{sc['C_trackman_mechanics']['signal']}** ({sc['C_trackman_mechanics']['status']}) | `{sc['C_trackman_mechanics']['best_stable']}` |",
        f"| D reliability/router | **{sc['D_reliability']['signal']}** | `{sc['D_reliability']['best_stable']}` |",
        "",
        "## A. Strict temporal residual transfer",
        "",
    ]
    for key, item in result["A"]["transitions"].items():
        lines.append(
            f"- {key}: source residual `{item['source_residual_mean']:+.6f}`, "
            f"target residual `{item['target_residual_mean']:+.6f}`, "
            f"baseline Brier `{item['baseline_brier']:.9f}`"
        )
    lines += ["", "Top temporal candidates:"]
    for item in result["A"]["ranking"][:10]:
        lines.append(
            f"- `{item['candidate']}` mean gain `{item['mean_gain']:+.9f}`, "
            f"min gain `{item['min_gain']:+.9f}`, "
            f"positive `{item['positive_transitions']}/{item['transitions']}`, "
            f"gains `{item['gains']}`"
        )

    lines += ["", "## C. Trackman mechanics", ""]
    c = result["C"]
    lines.append(f"- status: `{c.get('status')}`")
    if c.get("status") == "ok":
        lines.append(f"- selected ID mapping: `{c['selected_train_id']} -> {c['selected_trackman_id']}`")
        lines.append(f"- ID overlap: `{c['id_diagnostic']['best']}`")
        lines.append(f"- pitch type column: `{c.get('pitch_type_col')}`")
        lines.append(f"- mechanical columns: `{c.get('mechanical_columns')}`")
        lines.append(
            f"- matched prior-mechanics/current-residual pitcher-seasons: `{c.get('matched_pitcher_seasons')}`"
        )
        for season, fold in c.get("rolling", {}).items():
            lines.append(f"- {season}: pitchers `{fold['pitchers']}`, zero RMSE `{fold['zero_rmse']:.6f}`")
            for name, metric in fold["feature_sets"].items():
                lines.append(
                    f"  - {name}: RMSE `{metric['rmse']:.6f}`, gain `{metric['gain_vs_zero']:+.6f}`"
                )
    else:
        lines.append(f"- reason: `{c.get('reason')}`")
        lines.append(f"- ID diagnostic: `{c.get('id_diagnostic')}`")

    lines += ["", "## D. Fine reliability and previous-fold routing", ""]
    for key, item in result["D"]["transitions"].items():
        lines.append(f"- {key}: full-history Brier `{item['full_history_brier']:.9f}`")
        for axis, metric in item["routers"].items():
            lines.append(
                f"  - {axis}: Brier `{metric['brier']:.9f}`, "
                f"gain vs full `{metric['gain_vs_full_history']:+.9f}`, "
                f"source global best `{metric['source_global_best']}`, "
                f"selected `{metric['selected_counts']}`"
            )
    lines += ["", "Router summary:"]
    for axis, item in result["D"]["axes_summary"].items():
        lines.append(
            f"- {axis}: mean gain `{item['mean_gain']:+.9f}`, "
            f"min gain `{item['min_gain']:+.9f}`, "
            f"positive `{item['positive_transitions']}/{item['transitions']}`"
        )

    lines += [
        "",
        "## Decision rule",
        "",
        "- A survives only if one fixed correction family has positive gain on every historical next-season transfer.",
        "- C survives only if the same mechanical feature family improves residual prediction on every eligible rolling fold.",
        "- D survives as a specialist/router idea only if a previous-fold-selected router beats full_history on every historical transfer. Otherwise keep full_history as the default and treat cohort differences as reliability diagnostics only.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: str | Path) -> dict[str, Any]:
    exp = _load_yaml(config_path)
    cfg = load_config(_resolve(exp["baseline_config"]))
    out = _resolve(exp["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("[Cycle2] preparing SAFE-compatible frame", flush=True)
    data = prepare(cfg)
    frame = data.frame.reset_index(drop=True)
    season_col = cfg["data"]["season_col"]
    target_col = cfg["data"]["target_col"]
    groups = _feature_groups(frame, season_col)

    print("\n=== Rolling OOF foundation for A/C/D ===", flush=True)
    oof, fold_metrics = _rolling_information_audit(
        frame,
        season_col=season_col,
        target_col=target_col,
        folds=[int(x) for x in exp["rolling_folds"]],
        groups=groups,
        exp=exp,
        out=out,
    )
    oof = _attach_oof_context(oof, frame, season_col)

    print("\n=== A: strict prior-season residual transfer ===", flush=True)
    a = _temporal_transfer(oof, exp)

    print("\n=== C: repaired Trackman mechanics audit ===", flush=True)
    c = _mechanics_trackman(frame, oof, exp)

    print("\n=== D: fine reliability + causal previous-fold routing ===", flush=True)
    d = _causal_reliability(oof, exp)

    result = {
        "fold_metrics": fold_metrics,
        "A": a,
        "C": c,
        "D": d,
        "scorecard": _scorecard(a, c, d),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(out / "metrics.json", result)
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    print("\n[Cycle2 complete]", flush=True)
    for name, item in result["scorecard"].items():
        print(f"{name}: signal={item['signal']} best={item.get('best_stable')}", flush=True)
    print(f"elapsed={result['elapsed_seconds']/60.0:.1f} min", flush=True)
    print(f"report={out / 'report.md'}", flush=True)
    gc.collect()
    return result
