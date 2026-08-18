from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_recent_regime_submissions as recent_core
import run_context_interaction_screen as context_core
import run_frozen_domain_path_probe as path_core
import run_regime_feature_prediction_suite as regime_core
from src.canonical_features import CANONICAL_CATEGORICAL
from src.utils import load_config


class Model(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, min(16, max(4, int(np.ceil(n ** .25) * 2)))) for n in cardinalities])
        self.wide_cat = nn.ModuleList([nn.Embedding(n, 1) for n in cardinalities])
        width = n_num + sum(x.embedding_dim for x in self.emb)
        self.deep = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Dropout(.2), nn.Linear(256, 128), nn.SiLU(), nn.Dropout(.15), nn.Linear(128, 1))
        self.wide_num = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.wide_num.weight); nn.init.zeros_(self.wide_num.bias)
        for layer in self.wide_cat:
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.deep[-1].weight); nn.init.zeros_(self.deep[-1].bias)

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        embeddings = [layer(cat[:, i]) for i, layer in enumerate(self.emb)]
        wide = self.wide_num(num) + sum(layer(cat[:, i]) for i, layer in enumerate(self.wide_cat))
        return (wide + self.deep(torch.cat([num, *embeddings], dim=1)) + self.bias).squeeze(1)


def batches(n: int, size: int, shuffle: bool, device: torch.device, arrays: tuple[np.ndarray, ...]):
    order = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, size):
        idx = order[start:start + size]
        yield tuple(torch.from_numpy(x[idx]).to(device, non_blocking=True) for x in arrays)


def main() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    target, season = config["data"]["target_col"], config["data"]["season_col"]
    frame, _ = recent_core.prepare_frame(config)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame.game_type = frame.game_type.astype("string").str.upper()
    context_core.add_context_interactions(frame)
    paths = path_core.add_paths(frame, "pitcher_id", season, target) + path_core.add_paths(frame, "batter_id", season, target)
    paths = [x for x in paths if not x.endswith("_rate")]
    train, valid = frame[frame[season] < 2024].copy(), frame[frame[season] == 2024].copy()
    regime_core.add_regime_features(train, valid, season_col=season, recent_start=2023)
    base = [*recent_core.feature_set("recent_raw_game_type"), regime_core.RECENT_FLAG, *regime_core.FAST_CONT, *regime_core.RANGE_CONT]
    cat_cols = list(dict.fromkeys([*CANONICAL_CATEGORICAL, *context_core.INTERACTION_COLUMNS, "pitcher_id", "batter_id", *[x for x in paths if x.endswith(("last_gt", "current_x_last"))]]))
    num_cols = [x for x in [*base, *paths] if x not in cat_cols]

    train_cat, valid_cat, cardinalities = [], [], []
    for column in cat_cols:
        values = train[column].astype("string").fillna("<MISSING>")
        categories = pd.Index(values.unique())
        lookup = pd.Series(np.arange(1, len(categories) + 1), index=categories)
        train_cat.append(values.map(lookup).fillna(0).to_numpy(np.int64))
        valid_cat.append(valid[column].astype("string").fillna("<MISSING>").map(lookup).fillna(0).to_numpy(np.int64))
        cardinalities.append(len(categories) + 1)
    train_cat = np.column_stack(train_cat); valid_cat = np.column_stack(valid_cat)
    train_num = train[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    valid_num = valid[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    mean, std = np.nanmean(train_num, axis=0), np.nanstd(train_num, axis=0)
    std[std < 1e-6] = 1
    train_num = np.nan_to_num((train_num - mean) / std).astype(np.float32)
    valid_num = np.nan_to_num((valid_num - mean) / std).astype(np.float32)
    y_train = train[target].to_numpy(np.float32); y_valid = valid[target].to_numpy(np.float32)

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    model = Model(cardinalities, len(num_cols)).to(device)
    model.bias.data.fill_(float(np.log(y_train.mean() / (1.0 - y_train.mean()))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    best = (float("inf"), None)
    for epoch in range(1, 6):
        model.train(); total = 0.0
        for num, cat, y in batches(len(train), 8192, True, device, (train_num, train_cat, y_train)):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probability = torch.sigmoid(model(num, cat))
                loss = torch.mean((probability - y) ** 2)
            loss.backward(); optimizer.step(); total += float(loss) * len(y)
        model.eval(); predictions = []
        with torch.no_grad():
            for num, cat, _ in batches(len(valid), 16384, False, device, (valid_num, valid_cat, y_valid)):
                with torch.autocast("cuda", dtype=torch.bfloat16): predictions.append(torch.sigmoid(model(num, cat)).float().cpu().numpy())
        p = np.concatenate(predictions); brier = float(np.mean((p - y_valid) ** 2)); ref = float(y_valid.mean() * (1 - y_valid.mean())); score = 1e5 * (1 - brier / ref)
        print(f"e{epoch}: l={total/len(train):.2e} s={score:.1f} b={brier:.3e}")
        if brier < best[0]: best = (brier, p.copy())
    out = ROOT / "outputs/wide_deep_brier"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", y=y_valid, probability=best[1])


if __name__ == "__main__":
    main()
