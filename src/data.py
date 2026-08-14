from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

DATA_URL = "https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link"
EXPECTED_SEASONS = set(range(2019, 2025))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(root: str | Path = ".", force: bool = False) -> Path:
    root = Path(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    destination = raw / "dataset_download"

    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown이 없습니다. pip install -r configs/requirements.txt 를 먼저 실행하세요."
        ) from exc

    tmp = raw / "dataset_download.part"
    tmp.unlink(missing_ok=True)
    result = gdown.download(DATA_URL, str(tmp), quiet=False, fuzzy=True)
    if not result or not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("Google Drive 데이터 다운로드에 실패했습니다.")
    tmp.replace(destination)
    return destination


def _safe_extract_member_zip(zf: zipfile.ZipFile, member: zipfile.ZipInfo, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    name = Path(member.filename).name
    target = out / name
    with zf.open(member) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def _find_train_csv_from_archive(archive: Path, extract_dir: Path, target_col: str) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            candidates = [
                m for m in zf.infolist()
                if (not m.is_dir()) and m.filename.lower().endswith(".csv")
            ]
            candidates.sort(key=lambda m: (Path(m.filename).name.lower() != "train.csv", len(m.filename)))
            for member in candidates:
                path = _safe_extract_member_zip(zf, member, extract_dir)
                try:
                    cols = pd.read_csv(path, nrows=3, low_memory=False).columns.tolist()
                except Exception:
                    path.unlink(missing_ok=True)
                    continue
                if target_col in cols:
                    return path
                path.unlink(missing_ok=True)

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]
            members.sort(key=lambda m: (Path(m.name).name.lower() != "train.csv", len(m.name)))
            for member in members:
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                path = extract_dir / Path(member.name).name
                with path.open("wb") as dst:
                    shutil.copyfileobj(fh, dst)
                try:
                    cols = pd.read_csv(path, nrows=3, low_memory=False).columns.tolist()
                except Exception:
                    path.unlink(missing_ok=True)
                    continue
                if target_col in cols:
                    return path
                path.unlink(missing_ok=True)

    try:
        cols = pd.read_csv(archive, nrows=3, low_memory=False).columns.tolist()
        if target_col in cols:
            target = extract_dir / "train.csv"
            shutil.copy2(archive, target)
            return target
    except Exception:
        pass

    raise FileNotFoundError(f"{target_col!r}가 포함된 train CSV를 찾지 못했습니다.")


def prepare_dataset(
    root: str | Path = ".",
    target_col: str = "control_success",
    season_col: str = "season",
    force: bool = False,
) -> dict:
    root = Path(root)
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "train.pkl"

    if out.exists() and not force:
        frame = pd.read_pickle(out)[[season_col, target_col]]
        return {
            "processed_file": str(out),
            "rows": int(len(frame)),
            "seasons": sorted(pd.to_numeric(frame[season_col]).astype(int).unique().tolist()),
            "target_mean": float(pd.to_numeric(frame[target_col]).mean()),
            "reused": True,
        }

    archive = download_dataset(root, force=force)
    extract_dir = root / "data" / "raw" / "extracted"
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    train_csv = _find_train_csv_from_archive(archive, extract_dir, target_col)

    frame = pd.read_csv(train_csv, low_memory=False)
    if target_col not in frame.columns:
        raise ValueError(f"target column {target_col!r} 없음")
    if season_col not in frame.columns:
        raise ValueError(f"season column {season_col!r} 없음")

    target = pd.to_numeric(frame[target_col], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{target_col}은 결측 없는 0/1이어야 합니다.")

    season = pd.to_numeric(frame[season_col], errors="coerce")
    if season.isna().any():
        raise ValueError(f"{season_col}에 숫자로 변환되지 않는 값이 있습니다.")
    observed = set(season.astype(int).unique())
    missing = sorted(EXPECTED_SEASONS - observed)
    if missing:
        raise ValueError(f"2019~2024 중 누락 시즌: {missing}")

    # 학습 I/O를 위해 단일 pickle만 생성. 시즌별 중복 CSV는 만들지 않는다.
    frame.to_pickle(out)

    manifest = {
        "source_url": DATA_URL,
        "download_sha256": _sha256(archive),
        "source_csv": str(train_csv.relative_to(root)),
        "processed_file": str(out.relative_to(root)),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": frame.columns.tolist(),
        "target": target_col,
        "target_mean": float(target.mean()),
        "season_rows": {
            str(year): int((season.astype(int) == year).sum())
            for year in sorted(EXPECTED_SEASONS)
        },
    }
    with (processed_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_frame(config: dict) -> pd.DataFrame:
    path = Path(config["paths"]["processed_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 python scripts/prepare_data.py 를 실행하세요."
        )
    return pd.read_pickle(path)


def split_masks(frame: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    cfg = config["data"]
    season = pd.to_numeric(frame[cfg["season_col"]], errors="raise").astype(int)
    result = {
        "train": season.isin(cfg["train_seasons"]).to_numpy(),
        "val": season.isin(cfg["val_seasons"]).to_numpy(),
        "test": season.isin(cfg["test_seasons"]).to_numpy(),
    }
    if any(not mask.any() for mask in result.values()):
        raise ValueError({k: int(v.sum()) for k, v in result.items()})
    if (
        np.any(result["train"] & result["val"])
        or np.any(result["train"] & result["test"])
        or np.any(result["val"] & result["test"])
    ):
        raise ValueError("season split overlap")
    return result


def _normalize_token(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("<MISSING>")
        .str.strip()
        .str.lower()
    )


def _infer_pitcher_is_home(frame: pd.DataFrame) -> pd.Series:
    """
    top half = away batting = home team pitching.
    bottom half = home batting = away team pitching.

    문자열은 직접 해석하고, 0/1처럼 의미가 불명확한 값은
    score_diff_home vs score_diff_pitcher_team 관계로 mapping을 추론한다.
    """
    tb = _normalize_token(frame["top_bottom"])
    unique = [x for x in tb.dropna().unique().tolist() if x != "<MISSING>"]

    top_tokens = {"top", "t", "초", "top_inning"}
    bottom_tokens = {"bottom", "b", "말", "bottom_inning"}

    direct = pd.Series(index=frame.index, dtype="float64")
    direct[tb.isin(top_tokens)] = 1.0
    direct[tb.isin(bottom_tokens)] = 0.0

    unresolved = direct.isna()
    if not unresolved.any():
        return direct.astype(bool)

    if {"score_diff_home", "score_diff_pitcher_team"}.issubset(frame.columns):
        home = pd.to_numeric(frame["score_diff_home"], errors="coerce")
        pitcher = pd.to_numeric(frame["score_diff_pitcher_team"], errors="coerce")
        informative = home.notna() & pitcher.notna() & (home.abs() > 0)
        mapping = {}
        for token in unique:
            idx = (tb == token) & informative
            if idx.sum() < 20:
                continue
            same = np.isclose(
                home[idx].to_numpy(dtype=float),
                pitcher[idx].to_numpy(dtype=float),
                atol=1e-8,
            ).mean()
            opposite = np.isclose(
                home[idx].to_numpy(dtype=float),
                -pitcher[idx].to_numpy(dtype=float),
                atol=1e-8,
            ).mean()
            if max(same, opposite) >= 0.8:
                mapping[token] = 1.0 if same > opposite else 0.0
        direct[unresolved] = tb[unresolved].map(mapping)

    if direct.isna().any():
        bad = sorted(tb[direct.isna()].unique().tolist())
        raise ValueError(
            "top_bottom 의미를 확정할 수 없습니다. "
            f"미해석 값={bad}. 데이터 컬럼을 확인해 mapping을 추가하세요."
        )
    return direct.astype(bool)


@dataclass
class ContextArrays:
    categorical: np.ndarray
    numeric: np.ndarray


class ContextPreprocessor:
    """
    현재 상황을 identity-free canonical state로 만든다.

    categorical은 category ID -> embedding lookup 용 정수 index로만 반환한다.
    숫자 크기 자체는 의미가 없다.
    """

    categorical_columns = [
        "count_state",
        "base_state",
        "game_phase_state",
        "outs_before",
        "game_type",
        "game_month",
        "game_dayofweek",
        "pitcher_hand",
        "batter_hand",
    ]

    numeric_columns = [
        "score_diff_pitcher_team",
        "run_total_before",
        "pitcher_win_expectancy",
        "li",
    ]

    def __init__(self):
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.numeric_scaler = RobustScaler(quantile_range=(5, 95))
        self.fitted = False

    @staticmethod
    def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "balls_before",
            "strikes_before",
            "outs_before",
            "inning",
            "top_bottom",
            "runner_on_1b",
            "runner_on_2b",
            "runner_on_3b",
            "score_diff_pitcher_team",
            "home_win_expectancy",
            "away_win_expectancy",
            "li",
            "pitcher_hand",
            "batter_hand",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"context canonicalization 필수 컬럼 없음: {missing}")

        out = pd.DataFrame(index=frame.index)

        balls = pd.to_numeric(frame["balls_before"], errors="raise").astype(int)
        strikes = pd.to_numeric(frame["strikes_before"], errors="raise").astype(int)
        valid_count = balls.between(0, 3) & strikes.between(0, 2)
        if not valid_count.all():
            raise ValueError("balls_before/strikes_before에 유효하지 않은 값이 있습니다.")
        out["count_state"] = (balls * 3 + strikes).astype("int16").astype(str)

        r1 = pd.to_numeric(frame["runner_on_1b"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        r2 = pd.to_numeric(frame["runner_on_2b"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        r3 = pd.to_numeric(frame["runner_on_3b"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        # state ID: 1B=1, 2B=2, 3B=4. 0~7은 단지 category index.
        out["base_state"] = (r1 + 2 * r2 + 4 * r3).astype("int16").astype(str)

        inning = pd.to_numeric(frame["inning"], errors="raise").astype(int)
        tb = _normalize_token(frame["top_bottom"])
        out["game_phase_state"] = inning.astype(str) + "|" + tb

        for col in ["outs_before", "game_type", "game_month", "game_dayofweek", "pitcher_hand", "batter_hand"]:
            if col in frame.columns:
                out[col] = _normalize_token(frame[col])
            else:
                out[col] = "<MISSING>"

        out["score_diff_pitcher_team"] = pd.to_numeric(
            frame["score_diff_pitcher_team"], errors="coerce"
        )

        if "run_total_before" in frame.columns:
            out["run_total_before"] = pd.to_numeric(frame["run_total_before"], errors="coerce")
        elif {"run_top_before", "run_bot_before"}.issubset(frame.columns):
            out["run_total_before"] = (
                pd.to_numeric(frame["run_top_before"], errors="coerce")
                + pd.to_numeric(frame["run_bot_before"], errors="coerce")
            )
        else:
            raise ValueError("run_total_before 또는 run_top_before/run_bot_before가 필요합니다.")

        pitcher_is_home = _infer_pitcher_is_home(frame)
        home_we = pd.to_numeric(frame["home_win_expectancy"], errors="coerce")
        away_we = pd.to_numeric(frame["away_win_expectancy"], errors="coerce")
        out["pitcher_win_expectancy"] = np.where(pitcher_is_home, home_we, away_we)

        out["li"] = pd.to_numeric(frame["li"], errors="coerce")
        return out

    def fit(self, frame: pd.DataFrame) -> "ContextPreprocessor":
        x = self.canonicalize(frame)

        self.category_maps = {}
        for col in self.categorical_columns:
            values = _normalize_token(x[col])
            categories = sorted(values.unique().tolist())
            # 0은 unknown/masked 전용.
            self.category_maps[col] = {value: i + 1 for i, value in enumerate(categories)}

        numeric = x[self.numeric_columns].replace([np.inf, -np.inf], np.nan)
        imputed = self.numeric_imputer.fit_transform(numeric)
        self.numeric_scaler.fit(imputed)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> ContextArrays:
        if not self.fitted:
            raise RuntimeError("ContextPreprocessor.fit이 먼저 필요합니다.")
        x = self.canonicalize(frame)

        cat_parts = []
        for col in self.categorical_columns:
            values = _normalize_token(x[col])
            mapping = self.category_maps[col]
            cat_parts.append(values.map(mapping).fillna(0).to_numpy(dtype=np.int64))
        categorical = np.stack(cat_parts, axis=1)

        numeric = x[self.numeric_columns].replace([np.inf, -np.inf], np.nan)
        numeric = self.numeric_imputer.transform(numeric)
        numeric = self.numeric_scaler.transform(numeric)
        numeric = np.clip(numeric, -10, 10).astype(np.float32)

        return ContextArrays(categorical=categorical, numeric=numeric)

    @property
    def cardinalities(self) -> list[int]:
        # +1: index 0 unknown/masked
        return [len(self.category_maps[c]) + 1 for c in self.categorical_columns]

    @property
    def numeric_dim(self) -> int:
        if not self.fitted:
            raise RuntimeError("not fitted")
        return int(self.numeric_scaler.n_features_in_)


class HistoryPreprocessor:
    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.log_columns: list[str] = []
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.fitted = False

    def _raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.columns].copy()
        for col in self.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

        for col in self.log_columns:
            values = x[col]
            bad = values.dropna() < 0
            if bad.any():
                raise ValueError(f"count 성격 컬럼 {col}에 음수가 있습니다.")
            x[col] = np.log1p(values)
        return x.replace([np.inf, -np.inf], np.nan)

    def fit(self, frame: pd.DataFrame) -> "HistoryPreprocessor":
        self.log_columns = [
            c for c in self.columns
            if re.search(r"(^|_)n$", c, flags=re.IGNORECASE)
            or re.search(r"_count$", c, flags=re.IGNORECASE)
        ]
        x = self._raw(frame)
        imputed = self.imputer.fit_transform(x)
        self.scaler.fit(imputed)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("HistoryPreprocessor.fit이 먼저 필요합니다.")
        x = self._raw(frame)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x)
        return np.clip(x, -10, 10).astype(np.float32)

    @property
    def output_dim(self) -> int:
        if not self.fitted:
            raise RuntimeError("not fitted")
        return int(self.scaler.n_features_in_)


def select_history_columns(frame: pd.DataFrame, config: dict) -> list[str]:
    cfg = config["data"]
    patterns = [re.compile(p, flags=re.IGNORECASE) for p in cfg["history_patterns"]]
    excluded = set(cfg.get("exclude_columns", []))

    columns = []
    for col in frame.columns:
        if col in excluded:
            continue
        if any(p.search(col) for p in patterns):
            columns.append(col)

    result = []
    for col in columns:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        if numeric.nunique(dropna=True) <= 1:
            continue
        result.append(col)

    if not result:
        raise ValueError("history feature를 찾지 못했습니다.")
    return result
