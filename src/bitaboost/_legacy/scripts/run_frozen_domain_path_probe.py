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
import run_game_type_temporal_regime_ablation as model_core
import run_regime_feature_prediction_suite as regime_core
from src.utils import load_config


def add_paths(frame: pd.DataFrame, entity: str, season_col: str, target: str) -> list[str]:
    prefix = "p" if entity == "pitcher_id" else "b"
    outputs = [
        f"path_{prefix}_r_log", f"path_{prefix}_f_log", f"path_{prefix}_f_share",
        f"path_{prefix}_last_gt", f"path_{prefix}_current_x_last",
        f"path_{prefix}_ever_both", f"path_{prefix}_seasons_seen",
        f"path_{prefix}_r_rate", f"path_{prefix}_f_rate",
        f"path_{prefix}_last_r_rate", f"path_{prefix}_last_f_rate",
    ]
    for column in outputs:
        frame[column] = np.nan if not column.endswith(("last_gt", "current_x_last")) else "NEW"

    history: pd.DataFrame | None = None
    for year in sorted(frame[season_col].unique()):
        idx = frame.index[frame[season_col].eq(year)]
        current = frame.loc[idx, entity]
        if history is not None:
            r = current.map(history["R"]).fillna(0).to_numpy(float)
            f = current.map(history["F"]).fillna(0).to_numpy(float)
            last = current.map(history["last_gt"]).fillna("NEW").astype(str)
            seen = current.map(history["seasons_seen"]).fillna(0).to_numpy(float)
            r_rate = current.map(history["R_y"] / history["R"].replace(0, np.nan))
            f_rate = current.map(history["F_y"] / history["F"].replace(0, np.nan))
            frame.loc[idx, f"path_{prefix}_r_log"] = np.log1p(r).astype(np.float32)
            frame.loc[idx, f"path_{prefix}_f_log"] = np.log1p(f).astype(np.float32)
            frame.loc[idx, f"path_{prefix}_f_share"] = (f / np.maximum(r + f, 1)).astype(np.float32)
            frame.loc[idx, f"path_{prefix}_last_gt"] = last.to_numpy()
            frame.loc[idx, f"path_{prefix}_current_x_last"] = (
                frame.loc[idx, "game_type"].astype(str) + "<-" + last.to_numpy()
            )
            frame.loc[idx, f"path_{prefix}_ever_both"] = ((r > 0) & (f > 0)).astype(np.float32)
            frame.loc[idx, f"path_{prefix}_seasons_seen"] = seen.astype(np.float32)
            frame.loc[idx, f"path_{prefix}_r_rate"] = r_rate.to_numpy(np.float32)
            frame.loc[idx, f"path_{prefix}_f_rate"] = f_rate.to_numpy(np.float32)
            frame.loc[idx, f"path_{prefix}_last_r_rate"] = current.map(history["last_r_rate"]).to_numpy(np.float32)
            frame.loc[idx, f"path_{prefix}_last_f_rate"] = current.map(history["last_f_rate"]).to_numpy(np.float32)
        else:
            for c in (f"path_{prefix}_r_log", f"path_{prefix}_f_log", f"path_{prefix}_f_share", f"path_{prefix}_ever_both", f"path_{prefix}_seasons_seen"):
                frame.loc[idx, c] = np.float32(0)

        part = frame.loc[idx, [entity, "game_type", target]]
        counts = part.groupby([entity, "game_type"], observed=True).size().unstack(fill_value=0)
        sums = part.groupby([entity, "game_type"], observed=True)[target].sum().unstack(fill_value=0)
        for gt in ("R", "F"):
            if gt not in counts:
                counts[gt] = 0
            if gt not in sums:
                sums[gt] = 0
        dominant = counts[["R", "F"]].idxmax(axis=1).rename("last_gt")
        update = counts[["R", "F"]].join(dominant)
        update["R_y"] = sums["R"]
        update["F_y"] = sums["F"]
        update["last_r_rate"] = sums["R"] / counts["R"].replace(0, np.nan)
        update["last_f_rate"] = sums["F"] / counts["F"].replace(0, np.nan)
        update["seasons_seen"] = 1
        if history is None:
            history = update
        else:
            combined = history[["R", "F", "R_y", "F_y", "seasons_seen"]].add(update[["R", "F", "R_y", "F_y", "seasons_seen"]], fill_value=0)
            combined["last_gt"] = update["last_gt"].combine_first(history["last_gt"])
            combined["last_r_rate"] = update["last_r_rate"].combine_first(history["last_r_rate"])
            combined["last_f_rate"] = update["last_f_rate"].combine_first(history["last_f_rate"])
            history = combined
    return outputs


def add_team_paths(frame: pd.DataFrame, entity: str, team: str, season_col: str) -> list[str]:
    prefix = "p" if entity == "pitcher_id" else "b"
    outputs = [f"path_{prefix}_last_team", f"path_{prefix}_team_transition", f"path_{prefix}_team_switch", f"path_{prefix}_teams_seen"]
    frame[outputs[0]] = "NEW"; frame[outputs[1]] = "NEW"
    frame[outputs[2]] = np.float32(0); frame[outputs[3]] = np.float32(0)
    history: dict[object, set[str]] = {}
    last: dict[object, str] = {}
    for year in sorted(frame[season_col].unique()):
        idx = frame.index[frame[season_col].eq(year)]
        ids = frame.loc[idx, entity]
        previous = ids.map(last).fillna("NEW").astype(str)
        current = frame.loc[idx, team].astype("string").fillna("<MISSING>").astype(str)
        frame.loc[idx, outputs[0]] = previous.to_numpy()
        frame.loc[idx, outputs[1]] = (current + "<-" + previous.to_numpy()).to_numpy()
        frame.loc[idx, outputs[2]] = ((previous != "NEW") & (current != previous)).astype(np.float32).to_numpy()
        frame.loc[idx, outputs[3]] = ids.map(lambda x: len(history.get(x, set()))).astype(np.float32).to_numpy()
        part = frame.loc[idx, [entity, team]].copy()
        part[team] = part[team].astype("string").fillna("<MISSING>").astype(str)
        dominant = part.groupby([entity, team], observed=True).size().unstack(fill_value=0).idxmax(axis=1)
        for player, value in dominant.items():
            last[player] = str(value)
        for player, values in part.groupby(entity, observed=True)[team]:
            history.setdefault(player, set()).update(values.astype(str).unique())
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2024)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--output-dir", default="outputs/frozen_domain_path_probe")
    args = ap.parse_args()

    config = load_config(ROOT / "configs/default.yaml")
    target, season = config["data"]["target_col"], config["data"]["season_col"]
    frame, _ = recent_core.prepare_frame(config)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame["game_type"] = frame["game_type"].astype("string").str.upper()
    context_core.add_context_interactions(frame)
    path_features = add_paths(frame, "pitcher_id", season, target) + add_paths(frame, "batter_id", season, target)
    team_features = add_team_paths(frame, "pitcher_id", "pitcher_team_id", season) + add_team_paths(frame, "batter_id", "batter_team_id", season)
    state_features = [x for x in path_features if not x.endswith("_rate")]
    r_rate_features = [x for x in path_features if x.endswith(("_r_rate", "_last_r_rate"))]
    f_rate_features = [x for x in path_features if x.endswith(("_f_rate", "_last_f_rate"))]
    train, valid = frame[frame[season] < args.fold].copy(), frame[frame[season] == args.fold].copy()
    regime_core.add_regime_features(train, valid, season_col=season, recent_start=2023)
    base = [
        *recent_core.feature_set("recent_raw_game_type"), regime_core.RECENT_FLAG,
        *regime_core.FAST_CONT, *regime_core.RANGE_CONT, *context_core.INTERACTION_COLUMNS,
    ]
    params = model_core.build_params(config=config, iterations=args.iterations, task_type="GPU", devices=args.devices, gpu_ram_part=.95, pinned_memory_size="4GB")
    y = valid[target].to_numpy(float)
    predictions = {}
    variants = [
        ("BASE", base, set(context_core.INTERACTION_COLUMNS)),
        ("STATE", base + state_features),
        ("STATE_TEAM", base + state_features + team_features),
        ("STATE_R", base + state_features + r_rate_features),
        ("STATE_F", base + state_features + f_rate_features),
        ("STATE_RF", base + path_features),
    ]
    for item in variants:
        name, features = item[:2]
        cats = set(context_core.INTERACTION_COLUMNS) | {x for x in features if x.endswith(("last_gt", "current_x_last", "last_team", "team_transition"))}
        p = model_core.fit_predict(train=train, valid=valid, target_col=target, features=features, extra_categorical=cats, params=params)
        predictions[name] = p
        m = model_core.binary_metrics(y, p)
        print(f"{name}: s={m['score']:.1f} b={m['brier']:.3e}")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"fold_{args.fold}.npz", y=y, **predictions)


if __name__ == "__main__":
    main()
