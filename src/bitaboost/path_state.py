from __future__ import annotations

import numpy as np
import pandas as pd


def add_paths(frame: pd.DataFrame, entity: str, season_col: str, target: str | None = None) -> list[str]:
    """Recovered frozen path-state features used by the SAFE models.

    Every row in season s reads only aggregate state from seasons < s. Historical
    target-rate path outputs are intentionally omitted because the SAFE recipe
    filtered every *_rate path feature before model training.
    """
    del target
    prefix = "p" if entity == "pitcher_id" else "b"
    numeric = [f"path_{prefix}_r_log", f"path_{prefix}_f_log", f"path_{prefix}_f_share", f"path_{prefix}_ever_both", f"path_{prefix}_seasons_seen"]
    categorical = [f"path_{prefix}_last_gt", f"path_{prefix}_current_x_last"]
    outputs = [numeric[0], numeric[1], numeric[2], categorical[0], categorical[1], numeric[3], numeric[4]]
    missing = [c for c in numeric if c not in frame]
    if missing: frame[missing] = np.nan
    for c in categorical:
        if c not in frame: frame[c] = "NEW"
    history = None
    season_values = pd.to_numeric(frame[season_col], errors="raise").astype(int)
    for year in sorted(season_values.unique().tolist()):
        idx = frame.index[season_values.eq(year)]
        current = frame.loc[idx, entity]
        if history is None:
            frame.loc[idx, numeric] = np.float32(0.0)
        else:
            r = current.map(history["R"]).fillna(0).to_numpy(float)
            f = current.map(history["F"]).fillna(0).to_numpy(float)
            last = current.map(history["last_gt"]).fillna("NEW").astype(str)
            seen = current.map(history["seasons_seen"]).fillna(0).to_numpy(float)
            vals = np.column_stack([np.log1p(r), np.log1p(f), f/np.maximum(r+f,1), ((r>0)&(f>0)).astype(float), seen]).astype(np.float32)
            frame.loc[idx, numeric] = vals
            frame.loc[idx, categorical[0]] = last.to_numpy()
            frame.loc[idx, categorical[1]] = frame.loc[idx,"game_type"].astype(str).to_numpy() + "<-" + last.to_numpy()
        part = frame.loc[idx, [entity, "game_type"]]
        counts = part.groupby([entity,"game_type"], observed=True).size().unstack(fill_value=0)
        for gt in ("R","F"):
            if gt not in counts: counts[gt] = 0
        dominant = counts[["R","F"]].idxmax(axis=1).rename("last_gt")
        update = counts[["R","F"]].join(dominant); update["seasons_seen"] = 1
        if history is None:
            history = update
        else:
            combined = history[["R","F","seasons_seen"]].add(update[["R","F","seasons_seen"]], fill_value=0)
            combined["last_gt"] = update["last_gt"].combine_first(history["last_gt"])
            history = combined
    return outputs
