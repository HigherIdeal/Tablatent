from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


@dataclass
class FeatureGroups:
    current: list[str]
    history: list[str]
    excluded: list[str]


def _matches(name: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in patterns)


def split_feature_groups(df: pd.DataFrame, config: dict) -> FeatureGroups:
    data_cfg = config["data"]
    reserved = {data_cfg["target_col"], data_cfg["season_col"]}
    reserved.update(data_cfg.get("exclude_columns", []))
    post_patterns = data_cfg.get("post_event_patterns", [])
    available = [c for c in df.columns if c not in reserved and not _matches(c, post_patterns)]
    explicit_current = data_cfg.get("current_include", [])
    explicit_history = data_cfg.get("history_include", [])
    missing = (set(explicit_current) | set(explicit_history)) - set(df.columns)
    if missing:
        raise ValueError(f"설정에 지정했지만 데이터에 없는 컬럼: {sorted(missing)}")
    if explicit_current or explicit_history:
        history = list(dict.fromkeys(explicit_history))
        current = list(dict.fromkeys(explicit_current or [c for c in available if c not in history]))
    else:
        history = [c for c in available if _matches(c, data_cfg["history_patterns"])]
        current = [c for c in available if c not in history]
    # Constant/all-null fields carry no signal and can destabilize scalers.
    unusable = [c for c in current + history if df[c].nunique(dropna=True) <= 1]
    current = [c for c in current if c not in unusable]
    history = [c for c in history if c not in unusable]
    if not current:
        raise ValueError("현재 상황 feature가 없습니다. current_include 설정을 확인하세요.")
    if not history:
        raise ValueError("과거 이력 feature가 없습니다. history_patterns/history_include를 확인하세요.")
    excluded = [c for c in df.columns if c not in current and c not in history]
    return FeatureGroups(current, history, excluded)


class TabularPreprocessor:
    """Dense, bounded representation robust to unknown competition schemas."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler(quantile_range=(5, 95))),
        ])
        self.category_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)),
        ])

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        frame = frame.replace([np.inf, -np.inf], np.nan)
        self.numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in self.columns if c not in self.numeric]
        if self.numeric:
            self.numeric_pipe.fit(frame[self.numeric])
        if self.categorical:
            self.category_pipe.fit(frame[self.categorical].astype("string"))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.replace([np.inf, -np.inf], np.nan)
        parts = []
        if self.numeric:
            numeric = self.numeric_pipe.transform(frame[self.numeric])
            parts.append(np.clip(numeric, -10, 10).astype("float32"))
        if self.categorical:
            encoded = self.category_pipe.transform(frame[self.categorical].astype("string")).astype("float32")
            # Category rank is scaled per field to avoid large anonymous IDs dominating.
            sizes = np.array([max(len(v), 1) for v in self.category_pipe.named_steps["ordinal"].categories_], dtype="float32")
            parts.append(np.clip(encoded / sizes, -1, 1).astype("float32"))
        return np.concatenate(parts, axis=1) if parts else np.empty((len(frame), 0), dtype="float32")

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)
