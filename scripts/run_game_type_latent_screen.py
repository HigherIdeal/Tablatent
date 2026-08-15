from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_asof_state_engineering as asof_core
import run_context_interaction_screen as context_core
from src.canonical_features import (
    CANONICAL_FEATURES,
    CANONICAL_SOURCE_COLUMNS,
    PITCHER_TEAM_WIN_EXPECTANCY,
    add_canonical_derived_features,
    validate_canonical_schema,
)
from src.data import load_frame
from src.evaluation_metrics import probability_metrics
from src.utils import load_config, save_json, seed_everything


GAME_TYPE_COLUMN = "game_type"
POSITIVE_GAME_TYPE = "F"
NEGATIVE_GAME_TYPE = "R"
SOFT_PROXY_COLUMN = "game_context_prob_f"
LATENT_PREFIX = "game_context_latent_"

# Deliberately narrow: these describe the pre-pitch game state, not player identity,
# long-run player quality, target-derived asof_* history, or the season itself.
CONTEXT_CATEGORICAL = [
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "base_state",
]
CONTEXT_NUMERIC = [
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_total_before",
    "score_diff_home",
    PITCHER_TEAM_WIN_EXPECTANCY,
    "li",
]

VARIANTS = [
    "raw_game_type",
    "drop_game_type",
    "soft_proxy",
    "latent_proxy",
    "raw_plus_latent",
]


def parse_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("at least one fold is required")
    return result


def parse_strings(value: str) -> list[str]:
    result = [x.strip() for x in value.split(",") if x.strip()]
    if not result:
        raise ValueError("at least one variant is required")
    return result


def latent_columns(latent_dim: int) -> list[str]:
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    return [f"{LATENT_PREFIX}{index:02d}" for index in range(latent_dim)]


def feature_set(variant: str, latent_dim: int) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    base = list(CANONICAL_FEATURES) + list(asof_core.SUCCESS_STATE)
    without_game_type = [column for column in base if column != GAME_TYPE_COLUMN]
    latents = latent_columns(latent_dim)

    if variant == "raw_game_type":
        result = base
    elif variant == "drop_game_type":
        result = without_game_type
    elif variant == "soft_proxy":
        result = without_game_type + [SOFT_PROXY_COLUMN]
    elif variant == "latent_proxy":
        result = without_game_type + latents
    elif variant == "raw_plus_latent":
        result = base + latents
    else:  # pragma: no cover - guarded above
        raise AssertionError(variant)

    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate features in variant {variant}")
    return result


def _tokens(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<MISSING>").astype(str)


def encode_game_type(series: pd.Series) -> np.ndarray:
    tokens = _tokens(series)
    values = set(tokens.unique())
    allowed = {POSITIVE_GAME_TYPE, NEGATIVE_GAME_TYPE}
    unexpected = sorted(values - allowed)
    if unexpected:
        raise ValueError(f"Unexpected game_type values: {unexpected}")
    return tokens.eq(POSITIVE_GAME_TYPE).to_numpy(np.float32)


@dataclass
class GameContextPreprocessor:
    categories: dict[str, list[str]]
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "GameContextPreprocessor":
        missing = sorted((set(CONTEXT_CATEGORICAL) | set(CONTEXT_NUMERIC)) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing game-context columns: {missing}")

        categories: dict[str, list[str]] = {}
        for column in CONTEXT_CATEGORICAL:
            categories[column] = pd.unique(_tokens(frame[column])).tolist()

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for column in CONTEXT_NUMERIC:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64, copy=True)
            values[~np.isfinite(values)] = np.nan
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = float(finite.mean())
                std = float(finite.std())
                if not np.isfinite(std) or std < 1e-6:
                    std = 1.0
            means[column] = mean
            stds[column] = std

        return cls(categories=categories, means=means, stds=stds)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = sorted((set(CONTEXT_CATEGORICAL) | set(CONTEXT_NUMERIC)) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing game-context columns: {missing}")

        categorical_parts: list[np.ndarray] = []
        for column in CONTEXT_CATEGORICAL:
            tokens = _tokens(frame[column])
            categories = self.categories[column]
            # code -1 is validation/test-only category; +1 reserves 0 for unknown.
            codes = pd.Categorical(tokens, categories=categories).codes.astype(np.int64) + 1
            categorical_parts.append(codes)
        categorical = np.column_stack(categorical_parts).astype(np.int64, copy=False)

        numeric_parts: list[np.ndarray] = []
        for column in CONTEXT_NUMERIC:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64, copy=True)
            values[~np.isfinite(values)] = np.nan
            values = np.where(np.isfinite(values), values, self.means[column])
            values = (values - self.means[column]) / self.stds[column]
            numeric_parts.append(values.astype(np.float32, copy=False))
        numeric = np.column_stack(numeric_parts).astype(np.float32, copy=False)
        return categorical, numeric

    def category_cardinalities(self) -> list[int]:
        # +1 for unknown category at index 0.
        return [len(self.categories[column]) + 1 for column in CONTEXT_CATEGORICAL]


class GameTypeBottleneck(nn.Module):
    def __init__(
        self,
        category_cardinalities: list[int],
        numeric_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        embedding_dims = [min(8, max(2, (cardinality + 1) // 2)) for cardinality in category_cardinalities]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dim, padding_idx=0)
                for cardinality, embedding_dim in zip(category_cardinalities, embedding_dims)
            ]
        )
        input_dim = int(sum(embedding_dims) + numeric_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Linear(latent_dim, 1)

    def encode(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
        joined = torch.cat(embedded + [numeric], dim=1)
        return self.encoder(joined)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(categorical, numeric)
        logit = self.head(latent).squeeze(1)
        return logit, latent


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--encoder-device cuda requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def _loader(
    categorical: np.ndarray,
    numeric: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    tensors: list[torch.Tensor] = [
        torch.from_numpy(np.asarray(categorical, dtype=np.int64)),
        torch.from_numpy(np.asarray(numeric, dtype=np.float32)),
    ]
    if labels is not None:
        tensors.append(torch.from_numpy(np.asarray(labels, dtype=np.float32)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def _build_encoder(
    preprocessor: GameContextPreprocessor,
    latent_dim: int,
    hidden_dim: int,
    dropout: float,
) -> GameTypeBottleneck:
    return GameTypeBottleneck(
        category_cardinalities=preprocessor.category_cardinalities(),
        numeric_dim=len(CONTEXT_NUMERIC),
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def _run_encoder_epoch(
    model: GameTypeBottleneck,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    seen = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for categorical, numeric, labels in loader:
            categorical = categorical.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(categorical, numeric)
            loss = loss_fn(logits, labels)
            if training:
                (loss / max(len(labels), 1)).backward()
                optimizer.step()
            total_loss += float(loss.item())
            seen += len(labels)
    return total_loss / max(seen, 1)


def select_encoder_epoch(
    fit_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    latent_dim: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[int, list[dict[str, float]]]:
    preprocessor = GameContextPreprocessor.fit(fit_frame)
    fit_cat, fit_num = preprocessor.transform(fit_frame)
    valid_cat, valid_num = preprocessor.transform(valid_frame)
    fit_y = encode_game_type(fit_frame[GAME_TYPE_COLUMN])
    valid_y = encode_game_type(valid_frame[GAME_TYPE_COLUMN])

    model = _build_encoder(preprocessor, latent_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fit_loader = _loader(fit_cat, fit_num, fit_y, batch_size, True, num_workers)
    valid_loader = _loader(valid_cat, valid_num, valid_y, batch_size, False, num_workers)

    best_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        train_bce = _run_encoder_epoch(model, fit_loader, device, optimizer)
        valid_bce = _run_encoder_epoch(model, valid_loader, device, None)
        history.append({"epoch": float(epoch), "train_bce": train_bce, "valid_bce": valid_bce})
        print(
            f"    [encoder select] epoch={epoch:02d} train_bce={train_bce:.6f} "
            f"valid_bce={valid_bce:.6f}",
            flush=True,
        )
        if valid_bce < best_loss - 1e-6:
            best_loss = valid_bce
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("encoder selection failed to produce a best state")

    del model, optimizer, fit_loader, valid_loader, fit_cat, fit_num, valid_cat, valid_num, fit_y, valid_y
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history


def fit_encoder_fixed_epochs(
    fit_frame: pd.DataFrame,
    epochs: int,
    latent_dim: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[GameContextPreprocessor, GameTypeBottleneck, list[dict[str, float]]]:
    preprocessor = GameContextPreprocessor.fit(fit_frame)
    fit_cat, fit_num = preprocessor.transform(fit_frame)
    fit_y = encode_game_type(fit_frame[GAME_TYPE_COLUMN])
    model = _build_encoder(preprocessor, latent_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fit_loader = _loader(fit_cat, fit_num, fit_y, batch_size, True, num_workers)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_bce = _run_encoder_epoch(model, fit_loader, device, optimizer)
        history.append({"epoch": float(epoch), "train_bce": train_bce})
        print(f"    [encoder refit ] epoch={epoch:02d} train_bce={train_bce:.6f}", flush=True)

    del optimizer, fit_loader, fit_cat, fit_num, fit_y
    gc.collect()
    return preprocessor, model, history


def predict_encoder(
    preprocessor: GameContextPreprocessor,
    model: GameTypeBottleneck,
    frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    categorical, numeric = preprocessor.transform(frame)
    loader = _loader(categorical, numeric, None, batch_size, False, num_workers)
    probabilities: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for categorical_batch, numeric_batch in loader:
            categorical_batch = categorical_batch.to(device, non_blocking=True)
            numeric_batch = numeric_batch.to(device, non_blocking=True)
            logits, latent = model(categorical_batch, numeric_batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False))
            latents.append(latent.cpu().numpy().astype(np.float32, copy=False))
    del loader, categorical, numeric
    return np.concatenate(probabilities), np.concatenate(latents, axis=0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return float("nan")
    a = a[finite]
    b = b[finite]
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def game_type_probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logloss = float(-np.mean(y_true * np.log(probability) + (1.0 - y_true) * np.log(1.0 - probability)))
    brier = float(np.mean((probability - y_true) ** 2))
    accuracy = float(np.mean((probability >= 0.5) == (y_true >= 0.5)))
    auc = float(roc_auc_score(y_true, probability)) if np.unique(y_true).size == 2 else float("nan")
    return {
        "game_type_logloss": logloss,
        "game_type_brier": brier,
        "game_type_auc": auc,
        "game_type_accuracy": accuracy,
        "actual_f_rate": float(y_true.mean()),
        "predicted_f_rate": float(probability.mean()),
        "predicted_f_std": float(probability.std()),
    }


def season_game_type_summary(frame: pd.DataFrame, season_col: str, target_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year, group in frame.groupby(season_col, sort=True):
        game_type = _tokens(group[GAME_TYPE_COLUMN])
        unexpected = sorted(set(game_type.unique()) - {POSITIVE_GAME_TYPE, NEGATIVE_GAME_TYPE})
        if unexpected:
            raise ValueError(f"Season {year}: unexpected game_type values {unexpected}")
        target = pd.to_numeric(group[target_col], errors="raise").to_numpy(np.float64)
        is_f = game_type.eq(POSITIVE_GAME_TYPE).to_numpy()
        f_rate = float(target[is_f].mean()) if is_f.any() else float("nan")
        r_rate = float(target[~is_f].mean()) if (~is_f).any() else float("nan")
        rows.append(
            {
                "season": int(year),
                "rows": int(len(group)),
                "f_rows": int(is_f.sum()),
                "r_rows": int((~is_f).sum()),
                "f_share": float(is_f.mean()),
                "target_rate": float(target.mean()),
                "f_target_rate": f_rate,
                "r_target_rate": r_rate,
                "f_minus_r_target_gap": f_rate - r_rate,
            }
        )
    return pd.DataFrame(rows)


def proxy_bin_rows(
    validation_year: int,
    probability_f: np.ndarray,
    actual_f: np.ndarray,
    control_target: np.ndarray,
) -> list[dict[str, float | int | str]]:
    probability_f = np.clip(np.asarray(probability_f, dtype=np.float64), 0.0, 1.0)
    actual_f = np.asarray(actual_f, dtype=np.float64)
    control_target = np.asarray(control_target, dtype=np.float64)
    bin_index = np.minimum((probability_f * 10.0).astype(np.int64), 9)
    rows: list[dict[str, float | int | str]] = []
    for index in range(10):
        mask = bin_index == index
        if not mask.any():
            continue
        rows.append(
            {
                "validation_year": int(validation_year),
                "probability_bin": f"[{index / 10:.1f},{(index + 1) / 10:.1f})",
                "rows": int(mask.sum()),
                "probability_f_mean": float(probability_f[mask].mean()),
                "actual_f_rate": float(actual_f[mask].mean()),
                "control_success_rate": float(control_target[mask].mean()),
            }
        )
    return rows


def fit_catboost_variant(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict,
    iterations: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> dict[str, float]:
    from catboost import CatBoostClassifier, Pool

    x_train, categorical = context_core.prepare_x(train, features)
    x_valid, _ = context_core.prepare_x(valid, features)
    y_train = pd.to_numeric(train[target], errors="raise").to_numpy(np.float32)
    y_valid = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
    params = context_core.catboost_params(config, iterations, task_type, devices, verbose)
    train_pool = Pool(x_train, label=y_train, cat_features=categorical, feature_names=features)
    valid_pool = Pool(x_valid, cat_features=categorical, feature_names=features)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=verbose)
    prediction = model.predict_proba(valid_pool)[:, 1].astype(np.float64)
    metric = probability_metrics(y_valid, prediction)

    del model, train_pool, valid_pool, x_train, x_valid, y_train, y_valid, prediction
    gc.collect()
    return metric


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a season-blind neural bottleneck that predicts game_type from pure pre-pitch game context, "
            "then test its soft probability / latent representation as replacements for raw game_type in CatBoost."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2022,2023,2024")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--encoder-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-weight-decay", type=float, default=1e-4)
    parser.add_argument("--encoder-max-epochs", type=int, default=15)
    parser.add_argument("--encoder-patience", type=int, default=3)
    parser.add_argument("--encoder-batch-size", type=int, default=8192)
    parser.add_argument("--encoder-workers", type=int, default=0)
    parser.add_argument("--encoder-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if args.encoder_max_epochs <= 0 or args.encoder_patience <= 0:
        raise ValueError("encoder epochs/patience must be positive")
    if args.encoder_batch_size <= 0:
        raise ValueError("encoder batch size must be positive")

    config = load_config(ROOT / args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    target = config["data"]["target_col"]
    season = config["data"]["season_col"]
    row_id = config["data"].get("row_id_col", "row_id")
    folds = parse_ints(args.folds)
    variants = VARIANTS if args.variants == "all" else parse_strings(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    device = _resolve_device(args.encoder_device)
    frame = load_frame(config).copy()
    raw_canonical = [feature for feature in CANONICAL_FEATURES if feature != PITCHER_TEAM_WIN_EXPECTANCY]
    required = set(
        raw_canonical
        + CANONICAL_SOURCE_COLUMNS
        + list(CONTEXT_CATEGORICAL)
        + [column for column in CONTEXT_NUMERIC if column != PITCHER_TEAM_WIN_EXPECTANCY]
        + [
            target,
            season,
            row_id,
            GAME_TYPE_COLUMN,
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate",
            "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_success_rate",
            "asof_batter_middle_rate",
        ]
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    invariant_check = validate_canonical_schema(frame)
    add_canonical_derived_features(frame)
    asof_core.add_asof_state_features(frame)
    frame[season] = pd.to_numeric(frame[season], errors="raise").astype(int)
    frame = frame.sort_values([season, "game_month", row_id]).reset_index(drop=True)
    encode_game_type(frame[GAME_TYPE_COLUMN])  # strict F/R schema check

    latents = latent_columns(args.latent_dim)
    frame[SOFT_PROXY_COLUMN] = np.nan
    for column in latents:
        frame[column] = np.nan

    output_dir = Path(config["paths"]["output_dir"]) / "game_type_latent_screen"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = {variant: feature_set(variant, args.latent_dim) for variant in variants}
    (output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    season_summary = season_game_type_summary(frame, season, target)
    season_summary.to_csv(output_dir / "season_game_type_summary.csv", index=False)

    print(
        f"[Game-Type Latent Screen] folds={folds}, variants={variants}, latent_dim={args.latent_dim}, "
        f"encoder_device={device}, catboost_iterations={args.iterations}"
    )
    print("[Game-Type Latent Screen] encoder target=game_type only; control_success is NEVER used by the encoder.")
    print("[Game-Type Latent Screen] encoder inputs exclude season, game_type, IDs, team IDs, hands, and all asof_* history.")
    print(f"[Game-Type Latent Screen] context categorical={CONTEXT_CATEGORICAL}")
    print(f"[Game-Type Latent Screen] context numeric={CONTEXT_NUMERIC}")
    print("\n[Observed season x game_type target relation]")
    print(
        season_summary.to_string(
            index=False,
            formatters={
                "f_share": "{:.5f}".format,
                "target_rate": "{:.5f}".format,
                "f_target_rate": "{:.5f}".format,
                "r_target_rate": "{:.5f}".format,
                "f_minus_r_target_gap": "{:+.5f}".format,
            },
        )
    )

    fold_rows: list[dict[str, float | int | str]] = []
    encoder_rows: list[dict[str, float | int]] = []
    selection_history_rows: list[dict[str, float | int | str]] = []
    proxy_rows: list[dict[str, float | int | str]] = []
    latent_corr_rows: list[dict[str, float | int | str]] = []

    for validation_year in folds:
        train_mask = frame[season] < validation_year
        valid_mask = frame[season] == validation_year
        train = frame.loc[train_mask]
        valid = frame.loc[valid_mask]
        if train.empty or valid.empty:
            raise ValueError(f"Fold {validation_year}: empty train or validation")

        train_seasons = sorted(train[season].unique().tolist())
        if len(train_seasons) < 2:
            raise ValueError(f"Fold {validation_year}: need at least two prior seasons for encoder epoch selection")
        encoder_validation_year = int(train_seasons[-1])
        encoder_fit = train.loc[train[season] < encoder_validation_year]
        encoder_valid = train.loc[train[season] == encoder_validation_year]
        if encoder_fit.empty or encoder_valid.empty:
            raise ValueError(f"Fold {validation_year}: invalid encoder temporal split")

        print(
            f"\n[Fold {validation_year}] downstream_train={len(train):,}, downstream_valid={len(valid):,}; "
            f"encoder_select={encoder_fit[season].min()}..{encoder_validation_year - 1} -> {encoder_validation_year}",
            flush=True,
        )

        seed_everything(seed + validation_year)
        best_epoch, selection_history = select_encoder_epoch(
            encoder_fit,
            encoder_valid,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            learning_rate=args.encoder_lr,
            weight_decay=args.encoder_weight_decay,
            batch_size=args.encoder_batch_size,
            num_workers=args.encoder_workers,
            max_epochs=args.encoder_max_epochs,
            patience=args.encoder_patience,
            device=device,
        )
        for row in selection_history:
            selection_history_rows.append(
                {
                    "validation_year": int(validation_year),
                    "stage": "select",
                    "encoder_validation_year": encoder_validation_year,
                    **row,
                }
            )
        print(f"    selected_encoder_epochs={best_epoch}")

        seed_everything(seed + validation_year)
        preprocessor, encoder, refit_history = fit_encoder_fixed_epochs(
            train,
            epochs=best_epoch,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            learning_rate=args.encoder_lr,
            weight_decay=args.encoder_weight_decay,
            batch_size=args.encoder_batch_size,
            num_workers=args.encoder_workers,
            device=device,
        )
        for row in refit_history:
            selection_history_rows.append(
                {
                    "validation_year": int(validation_year),
                    "stage": "refit",
                    "encoder_validation_year": encoder_validation_year,
                    **row,
                }
            )

        train_prob_f, train_latent = predict_encoder(
            preprocessor, encoder, train, args.encoder_batch_size, args.encoder_workers, device
        )
        valid_prob_f, valid_latent = predict_encoder(
            preprocessor, encoder, valid, args.encoder_batch_size, args.encoder_workers, device
        )

        train_indices = train.index.to_numpy()
        valid_indices = valid.index.to_numpy()
        frame.loc[train_indices, SOFT_PROXY_COLUMN] = train_prob_f
        frame.loc[valid_indices, SOFT_PROXY_COLUMN] = valid_prob_f
        frame.loc[train_indices, latents] = train_latent
        frame.loc[valid_indices, latents] = valid_latent

        actual_f = encode_game_type(valid[GAME_TYPE_COLUMN])
        control_target = pd.to_numeric(valid[target], errors="raise").to_numpy(np.float64)
        encoder_metric = game_type_probability_metrics(actual_f, valid_prob_f)
        encoder_metric.update(
            {
                "validation_year": int(validation_year),
                "selected_epochs": int(best_epoch),
                "control_corr_prob_f": _safe_corr(valid_prob_f, control_target),
                "control_corr_game_type_residual": _safe_corr(actual_f - valid_prob_f, control_target),
            }
        )
        encoder_rows.append(encoder_metric)
        proxy_rows.extend(proxy_bin_rows(validation_year, valid_prob_f, actual_f, control_target))
        for index, column in enumerate(latents):
            latent_corr_rows.append(
                {
                    "validation_year": int(validation_year),
                    "latent": column,
                    "corr_game_type": _safe_corr(valid_latent[:, index], actual_f),
                    "corr_control_success": _safe_corr(valid_latent[:, index], control_target),
                }
            )

        print(
            f"    encoder outer: F_auc={encoder_metric['game_type_auc']:.5f} "
            f"F_brier={encoder_metric['game_type_brier']:.6f} "
            f"actual_F={encoder_metric['actual_f_rate']:.5f} "
            f"pred_F={encoder_metric['predicted_f_rate']:.5f} "
            f"corr(probF, control)={encoder_metric['control_corr_prob_f']:+.5f}",
            flush=True,
        )

        del encoder, preprocessor, train_prob_f, valid_prob_f, train_latent, valid_latent, actual_f, control_target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Re-select views after assigning proxy/latent columns into the shared frame.
        train_augmented = frame.loc[train_mask]
        valid_augmented = frame.loc[valid_mask]
        for variant in variants:
            features = feature_sets[variant]
            metric = fit_catboost_variant(
                train_augmented,
                valid_augmented,
                target=target,
                features=features,
                config=config,
                iterations=args.iterations,
                task_type=args.task_type,
                devices=args.devices,
                verbose=args.verbose,
            )
            fold_rows.append(
                {
                    "validation_year": int(validation_year),
                    "variant": variant,
                    "train_rows": int(len(train_augmented)),
                    "valid_rows": int(len(valid_augmented)),
                    "feature_count": int(len(features)),
                    "selected_encoder_epochs": int(best_epoch),
                    **metric,
                }
            )
            print(
                f"    [{variant:<18s}] brier={metric['brier']:.8f} "
                f"raw_score={metric['raw_score']:+.2f} auc={metric['auc']:.5f} "
                f"p_std={metric['prediction_std']:.5f}",
                flush=True,
            )

        gc.collect()

    results = pd.DataFrame(fold_rows)
    raw_reference = (
        results.loc[results["variant"].eq("raw_game_type"), ["validation_year", "brier"]]
        .rename(columns={"brier": "raw_game_type_brier"})
    )
    if not raw_reference.empty:
        results = results.merge(raw_reference, on="validation_year", how="left")
        results["delta_brier_vs_raw_game_type"] = results["brier"] - results["raw_game_type_brier"]
    else:
        results["raw_game_type_brier"] = np.nan
        results["delta_brier_vs_raw_game_type"] = np.nan
    results.to_csv(output_dir / "fold_results.csv", index=False)

    summary = (
        results.groupby("variant", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            feature_count=("feature_count", "first"),
            mean_brier=("brier", "mean"),
            worst_brier=("brier", "max"),
            mean_delta_vs_raw=("delta_brier_vs_raw_game_type", "mean"),
            worst_delta_vs_raw=("delta_brier_vs_raw_game_type", "max"),
            best_delta_vs_raw=("delta_brier_vs_raw_game_type", "min"),
            mean_raw_score=("raw_score", "mean"),
            worst_raw_score=("raw_score", "min"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["mean_delta_vs_raw", "worst_delta_vs_raw"], na_position="last")
        .reset_index(drop=True)
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(encoder_rows).to_csv(output_dir / "encoder_results.csv", index=False)
    pd.DataFrame(selection_history_rows).to_csv(output_dir / "encoder_training_history.csv", index=False)
    pd.DataFrame(proxy_rows).to_csv(output_dir / "proxy_bins.csv", index=False)
    pd.DataFrame(latent_corr_rows).to_csv(output_dir / "latent_target_correlations.csv", index=False)

    save_json(
        {
            "folds": folds,
            "variants": variants,
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "encoder_lr": float(args.encoder_lr),
            "encoder_weight_decay": float(args.encoder_weight_decay),
            "encoder_max_epochs": int(args.encoder_max_epochs),
            "encoder_patience": int(args.encoder_patience),
            "encoder_batch_size": int(args.encoder_batch_size),
            "encoder_device": str(device),
            "catboost_iterations": int(args.iterations),
            "catboost_task_type": args.task_type,
            "context_categorical": CONTEXT_CATEGORICAL,
            "context_numeric": CONTEXT_NUMERIC,
            "encoder_excludes": [
                "season",
                "control_success",
                "game_type as input",
                "pitcher_id",
                "batter_id",
                "pitcher_team_id",
                "batter_team_id",
                "pitcher_hand",
                "batter_hand",
                "all asof_* history",
            ],
            "encoder_target": GAME_TYPE_COLUMN,
            "positive_game_type": POSITIVE_GAME_TYPE,
            "canonical_invariants": invariant_check,
            "sampling": "none",
            "protocol": (
                "For outer year Y, select neural encoder epoch count using only prior seasons: "
                "train < Y-1 and validate Y-1 on game_type; then refit the encoder on all <Y for that epoch count. "
                "The encoder never sees control_success. CatBoost is fixed-complexity and evaluates Y."
            ),
        },
        output_dir / "run_config.json",
    )

    print("\n[Game-Type Latent Summary: negative delta beats raw game_type]")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mean_brier": "{:.8f}".format,
                "worst_brier": "{:.8f}".format,
                "mean_delta_vs_raw": "{:+.8f}".format,
                "worst_delta_vs_raw": "{:+.8f}".format,
                "best_delta_vs_raw": "{:+.8f}".format,
                "mean_raw_score": "{:+.2f}".format,
                "worst_raw_score": "{:+.2f}".format,
                "mean_auc": "{:.5f}".format,
            },
        )
    )
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
