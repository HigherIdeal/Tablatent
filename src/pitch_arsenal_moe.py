"""Shared data and model components for the Physics-Arsenal MoE experiment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
PHYSICAL_TARGETS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]

CONTEXT_CATEGORICAL = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

CONTEXT_NUMERIC = [
    "season",
    "run_total_before",
    "score_diff_home",
    "pitcher_team_win_expectancy",
    "li",
    "log1p_asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "log1p_asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "log1p_asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "eng_ps_prev1_minus_long",
    "eng_ps_prev3_minus_long",
    "eng_ps_prev5_minus_long",
    "eng_ps_prev1_minus_prev3",
    "eng_ps_prev3_minus_prev5",
    "eng_ps_prev1_minus_prev5",
    "eng_ps_recent_mean_135",
    "eng_ps_recent_mean_minus_long",
    "eng_ps_recent_range_135",
]

RAW_CONTEXT_REQUIRED = sorted(
    {
        *CONTEXT_CATEGORICAL,
        "season",
        "run_total_before",
        "score_diff_home",
        "home_win_expectancy",
        "away_win_expectancy",
        "li",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_pitchmix_n",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
        "row_id",
    }
)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(np.float32)


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    top_bottom = frame["top_bottom"].astype("string").str.strip()
    home = numeric(frame["home_win_expectancy"])
    away = numeric(frame["away_win_expectancy"])
    frame["pitcher_team_win_expectancy"] = np.where(
        top_bottom.eq("T"), home, away
    ).astype(np.float32)

    long_rate = numeric(frame["asof_pitcher_success_rate"])
    prev1 = numeric(frame["asof_pitcher_prev1_game_success_rate"])
    prev3 = numeric(frame["asof_pitcher_prev3_game_success_rate"])
    prev5 = numeric(frame["asof_pitcher_prev5_game_success_rate"])
    frame["eng_ps_prev1_minus_long"] = prev1 - long_rate
    frame["eng_ps_prev3_minus_long"] = prev3 - long_rate
    frame["eng_ps_prev5_minus_long"] = prev5 - long_rate
    frame["eng_ps_prev1_minus_prev3"] = prev1 - prev3
    frame["eng_ps_prev3_minus_prev5"] = prev3 - prev5
    frame["eng_ps_prev1_minus_prev5"] = prev1 - prev5
    recent = pd.concat([prev1, prev3, prev5], axis=1)
    frame["eng_ps_recent_mean_135"] = recent.mean(axis=1, skipna=False)
    frame["eng_ps_recent_mean_minus_long"] = (
        frame["eng_ps_recent_mean_135"] - long_rate
    )
    frame["eng_ps_recent_range_135"] = (
        recent.max(axis=1, skipna=False) - recent.min(axis=1, skipna=False)
    )

    for source, target in (
        ("asof_pitcher_n", "log1p_asof_pitcher_n"),
        ("asof_batter_n", "log1p_asof_batter_n"),
        ("asof_pitcher_pitchmix_n", "log1p_asof_pitcher_pitchmix_n"),
    ):
        values = numeric(frame[source]).clip(lower=0)
        frame[target] = np.log1p(values).astype(np.float32)
    return frame


def normalize_category_value(value: Any) -> str:
    if pd.isna(value):
        return "<MISSING>"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "<MISSING>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def fit_category_maps(
    frame: pd.DataFrame, train_indices: np.ndarray
) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    train = frame.iloc[train_indices]
    for column in CONTEXT_CATEGORICAL:
        values = sorted(
            {
                normalize_category_value(value)
                for value in train[column].to_numpy()
            }
        )
        maps[column] = {value: index + 2 for index, value in enumerate(values)}
    return maps


def encode_categories(
    frame: pd.DataFrame, category_maps: dict[str, dict[str, int]]
) -> np.ndarray:
    encoded = np.zeros(
        (len(frame), len(CONTEXT_CATEGORICAL)), dtype=np.int64
    )
    for column_index, column in enumerate(CONTEXT_CATEGORICAL):
        mapping = category_maps[column]
        values = frame[column].map(normalize_category_value)
        encoded[:, column_index] = (
            values.map(mapping).fillna(1).astype(np.int64).to_numpy()
        )
    return encoded


def category_cardinalities(
    category_maps: dict[str, dict[str, int]]
) -> list[int]:
    return [len(category_maps[column]) + 2 for column in CONTEXT_CATEGORICAL]


def raw_context_numeric(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [numeric(frame[column]).to_numpy() for column in CONTEXT_NUMERIC]
    ).astype(np.float32, copy=False)


def fit_scaler(values: np.ndarray, train_indices: np.ndarray) -> dict[str, list[float]]:
    selected = values[train_indices]
    with np.errstate(invalid="ignore"):
        center = np.nanmean(selected, axis=0)
        scale = np.nanstd(selected, axis=0)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(
        np.float32
    )
    return {"center": center.tolist(), "scale": scale.tolist()}


def apply_scaler(values: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    center = np.asarray(scaler["center"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    transformed = (values - center) / scale
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def regime_indices(frame: pd.DataFrame) -> np.ndarray:
    values = frame["game_type"].astype("string").str.strip()
    unknown = sorted(set(values.dropna().unique()) - {"F", "R"})
    if unknown:
        raise ValueError(f"Unexpected game_type values: {unknown}")
    return values.map({"F": 0, "R": 1}).fillna(0).astype(np.int64).to_numpy()


def recency_weights(seasons: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    maximum = int(np.max(seasons[train_indices]))
    age = maximum - seasons
    weights = np.select(
        [age <= 0, age == 1, age == 2],
        [1.0, 0.70, 0.35],
        default=0.15,
    )
    return weights.astype(np.float32)


def fit_regime_priors(
    frame: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    sample_weights: np.ndarray,
) -> dict[str, float]:
    priors: dict[str, float] = {}
    game_types = frame["game_type"].astype("string").str.strip().to_numpy()
    weighted_global = float(
        np.average(target[train_indices], weights=sample_weights[train_indices])
    )
    priors["__global__"] = float(np.clip(weighted_global, 1e-4, 1 - 1e-4))
    for game_type in ("F", "R"):
        selected = train_indices[game_types[train_indices] == game_type]
        if len(selected):
            value = float(
                np.average(target[selected], weights=sample_weights[selected])
            )
        else:
            value = weighted_global
        priors[game_type] = float(np.clip(value, 1e-4, 1 - 1e-4))
    return priors


def compute_prior_logits(
    frame: pd.DataFrame,
    regime_priors: dict[str, float],
    prior_strength: float,
) -> np.ndarray:
    game_type = frame["game_type"].astype("string").str.strip()
    fallback = game_type.map(regime_priors).fillna(regime_priors["__global__"])
    pitcher_rate = numeric(frame["asof_pitcher_success_rate"])
    pitcher_n = numeric(frame["asof_pitcher_n"]).clip(lower=0)
    reliability = pitcher_n / (pitcher_n + float(prior_strength))
    probability = reliability * pitcher_rate + (1.0 - reliability) * fallback
    probability = probability.fillna(fallback).clip(1e-5, 1 - 1e-5)
    values = probability.to_numpy(np.float32)
    return (np.log(values) - np.log1p(-values)).astype(np.float32)


@dataclass
class ArsenalTables:
    player: pd.DataFrame
    team_hand: pd.DataFrame
    league_hand: pd.DataFrame
    league: pd.DataFrame
    feature_columns: list[str]


def _indexed_profile(
    frame: pd.DataFrame, keys: list[str], features: list[str]
) -> pd.DataFrame:
    normalized = frame.copy()
    for column in keys:
        if column != "pitch_group_id":
            normalized[column] = pd.to_numeric(
                normalized[column], errors="coerce"
            ).astype("Int64")
    normalized["pitch_group_id"] = pd.to_numeric(
        normalized["pitch_group_id"], errors="raise"
    ).astype("int8")
    if normalized.duplicated(keys).any():
        raise ValueError(f"Duplicate arsenal profile keys: {keys}")
    return normalized.set_index(keys)[features].sort_index()


def load_arsenal_tables(directory: Path) -> ArsenalTables:
    manifest = json.loads(
        (directory / "arsenal_feature_manifest.json").read_text(encoding="utf-8")
    )
    features = [str(value) for value in manifest["arsenal_feature_columns"]]
    player_raw = pd.read_parquet(directory / "pitcher_arsenal_by_season.parquet")
    team_raw = pd.read_parquet(directory / "team_hand_arsenal_by_season.parquet")
    hand_raw = pd.read_parquet(directory / "league_hand_arsenal_by_season.parquet")
    league_raw = pd.read_parquet(directory / "league_arsenal_by_season.parquet")
    return ArsenalTables(
        player=_indexed_profile(
            player_raw,
            ["pitcher_id", "feature_season", "pitch_group_id"],
            features,
        ),
        team_hand=_indexed_profile(
            team_raw,
            [
                "pitcher_team_id",
                "pitcher_hand",
                "feature_season",
                "pitch_group_id",
            ],
            features,
        ),
        league_hand=_indexed_profile(
            hand_raw,
            ["pitcher_hand", "feature_season", "pitch_group_id"],
            features,
        ),
        league=_indexed_profile(
            league_raw,
            ["feature_season", "pitch_group_id"],
            features,
        ),
        feature_columns=features,
    )


def _nullable_integer_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype("Int64").to_numpy()


def _lookup_profile(
    table: pd.DataFrame, arrays: list[np.ndarray], names: list[str]
) -> np.ndarray:
    if len(names) == 1:
        index = pd.Index(arrays[0], name=names[0])
    else:
        index = pd.MultiIndex.from_arrays(arrays, names=names)
    return table.reindex(index).to_numpy(dtype=np.float32, na_value=np.nan)


def resolve_arsenal(
    frame: pd.DataFrame, tables: ArsenalTables
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(frame)
    n_features = len(tables.feature_columns)
    result = np.full(
        (n_rows, len(PITCH_GROUPS), n_features), np.nan, dtype=np.float32
    )
    source = np.full((n_rows, len(PITCH_GROUPS)), 4, dtype=np.int64)
    pitcher_ids = _nullable_integer_values(frame["pitcher_id"])
    team_ids = _nullable_integer_values(frame["pitcher_team_id"])
    hands = _nullable_integer_values(frame["pitcher_hand"])
    seasons = _nullable_integer_values(frame["season"])

    for group_id in range(len(PITCH_GROUPS)):
        groups = np.full(n_rows, group_id, dtype=np.int8)
        resolved = _lookup_profile(
            tables.player,
            [pitcher_ids, seasons, groups],
            ["pitcher_id", "feature_season", "pitch_group_id"],
        )
        available = np.isfinite(resolved[:, 2])
        source[available, group_id] = 0

        candidates = [
            (
                tables.team_hand,
                [team_ids, hands, seasons, groups],
                [
                    "pitcher_team_id",
                    "pitcher_hand",
                    "feature_season",
                    "pitch_group_id",
                ],
                1,
            ),
            (
                tables.league_hand,
                [hands, seasons, groups],
                ["pitcher_hand", "feature_season", "pitch_group_id"],
                2,
            ),
            (
                tables.league,
                [seasons, groups],
                ["feature_season", "pitch_group_id"],
                3,
            ),
        ]
        for table, arrays, names, source_id in candidates:
            missing = ~available
            if not missing.any():
                break
            candidate = _lookup_profile(table, arrays, names)
            candidate_available = np.isfinite(candidate[:, 2])
            take = missing & candidate_available
            resolved[take] = candidate[take]
            source[take, group_id] = source_id
            available |= take
        result[:, group_id, :] = resolved
    return result, source


def log_transform_arsenal(
    values: np.ndarray, feature_columns: list[str]
) -> np.ndarray:
    transformed = values.copy()
    for column in (
        "ars_pitch_count",
        "ars_total_pitch_count",
        "ars_seasons_observed",
        "ars_prevseason_pitch_count",
    ):
        index = feature_columns.index(column)
        transformed[:, :, index] = np.log1p(
            np.clip(transformed[:, :, index], 0, None)
        )
    return transformed


def fit_arsenal_scaler(
    values: np.ndarray, train_indices: np.ndarray
) -> dict[str, list[float]]:
    selected = values[train_indices].reshape(-1, values.shape[-1])
    with np.errstate(invalid="ignore"):
        center = np.nanmean(selected, axis=0)
        scale = np.nanstd(selected, axis=0)
    center = np.where(np.isfinite(center), center, 0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(
        np.float32
    )
    return {"center": center.tolist(), "scale": scale.tolist()}


def apply_arsenal_scaler(
    values: np.ndarray, scaler: dict[str, list[float]]
) -> np.ndarray:
    center = np.asarray(scaler["center"], dtype=np.float32).reshape(1, 1, -1)
    scale = np.asarray(scaler["scale"], dtype=np.float32).reshape(1, 1, -1)
    transformed = (values - center) / scale
    return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


@dataclass
class ModelConfig:
    category_cardinalities: list[int]
    context_numeric_dim: int
    arsenal_feature_dim: int
    physical_target_dim: int = len(PHYSICAL_TARGETS)
    d_model: int = 128
    hidden_dim: int = 256
    embedding_dim: int = 8
    attention_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.15
    arsenal_residual_scale: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class PhysicsArsenalMoE(nn.Module):
    """Direct R/F predictor with a bounded historical-arsenal residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, config.embedding_dim)
                for cardinality in config.category_cardinalities
            ]
        )
        context_input = (
            len(config.category_cardinalities) * config.embedding_dim
            + config.context_numeric_dim
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.d_model),
            ResidualBlock(config.d_model, config.dropout),
        )
        self.arsenal_projection = nn.Sequential(
            nn.Linear(config.arsenal_feature_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
        )
        self.pitch_group_embedding = nn.Embedding(len(PITCH_GROUPS), config.d_model)
        self.profile_source_embedding = nn.Embedding(5, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.arsenal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.attention_layers
        )
        joint_dim = config.d_model * 2
        self.pitch_selection_head = nn.Sequential(
            nn.Linear(joint_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.direct_success_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )
        self.arsenal_residual_head = nn.Sequential(
            nn.Linear(joint_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )
        self.physics_head = nn.Sequential(
            nn.Linear(joint_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.physical_target_dim),
        )

    def forward(
        self,
        categorical: torch.Tensor,
        context_numeric: torch.Tensor,
        arsenal: torch.Tensor,
        arsenal_source: torch.Tensor,
        regime: torch.Tensor,
        prior_logit: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded = [
            layer(categorical[:, index])
            for index, layer in enumerate(self.category_embeddings)
        ]
        context = self.context_encoder(
            torch.cat([*embedded, context_numeric], dim=-1)
        )
        group_ids = torch.arange(
            len(PITCH_GROUPS), device=arsenal.device, dtype=torch.long
        ).unsqueeze(0)
        tokens = (
            self.arsenal_projection(arsenal)
            + self.pitch_group_embedding(group_ids)
            + self.profile_source_embedding(arsenal_source)
        )
        tokens = self.arsenal_encoder(tokens)
        repeated_context = context.unsqueeze(1).expand(-1, len(PITCH_GROUPS), -1)
        joint = torch.cat([repeated_context, tokens], dim=-1)
        selection_logits = self.pitch_selection_head(joint).squeeze(-1)
        mixture_weights = torch.softmax(selection_logits, dim=-1)
        regime_index = regime.unsqueeze(1)
        direct_residual = torch.gather(
            self.direct_success_head(context), 1, regime_index
        ).squeeze(1)
        pooled_arsenal = tokens.mean(dim=1)
        arsenal_residual = torch.gather(
            self.arsenal_residual_head(
                torch.cat([context, pooled_arsenal], dim=-1)
            ),
            1,
            regime_index,
        ).squeeze(1)
        final_logit = (
            prior_logit
            + direct_residual
            + self.config.arsenal_residual_scale * torch.tanh(arsenal_residual)
        )
        probability = torch.sigmoid(final_logit)
        probability = probability.clamp(1e-6, 1 - 1e-6)
        return {
            "probability": probability,
            "logit": final_logit,
            "selection_logits": selection_logits,
            "mixture_weights": mixture_weights,
            "physics_prediction": self.physics_head(joint),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _shape_check() -> None:
    config = ModelConfig([3] * len(CONTEXT_CATEGORICAL), len(CONTEXT_NUMERIC), 38)
    model = PhysicsArsenalMoE(config)
    rows = 2
    output = model(
        torch.zeros(rows, len(CONTEXT_CATEGORICAL), dtype=torch.long),
        torch.zeros(rows, len(CONTEXT_NUMERIC)),
        torch.zeros(rows, len(PITCH_GROUPS), 38),
        torch.full((rows, len(PITCH_GROUPS)), 4, dtype=torch.long),
        torch.tensor([0, 1]),
        torch.zeros(rows),
    )
    assert output["probability"].shape == (rows,)
    assert output["selection_logits"].shape == (rows, len(PITCH_GROUPS))


if __name__ == "__main__":
    _shape_check()
