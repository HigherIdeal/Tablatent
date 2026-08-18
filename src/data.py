from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXPECTED_SEASONS = set(range(2019, 2025))
CACHE_NAME = "train.parquet"


def _csv_has_target(path: Path, target_col: str) -> bool:
    try:
        return target_col in pd.read_csv(path, nrows=2, low_memory=False).columns
    except Exception:
        return False


def _find_train_csv(root: Path, target_col: str) -> Path:
    preferred = [
        root / "data" / "train.csv",
        root / "data" / "raw" / "train.csv",
        root / "data" / "raw" / "extracted" / "train.csv",
    ]
    for path in preferred:
        if path.is_file() and _csv_has_target(path, target_col):
            return path

    data_root = root / "data"
    if data_root.exists():
        candidates = sorted(
            data_root.rglob("*.csv"),
            key=lambda p: (p.name.lower() != "train.csv", len(str(p))),
        )
        for path in candidates:
            if _csv_has_target(path, target_col):
                return path

    raise FileNotFoundError(
        "Could not find the official training CSV under data/. "
        "Place train.csv at data/train.csv (or data/raw/extracted/train.csv) "
        "and rerun python scripts/prepare_data.py."
    )


def _validate_frame(
    frame: pd.DataFrame,
    *,
    target_col: str,
    season_col: str,
) -> tuple[pd.Series, pd.Series]:
    if target_col not in frame.columns:
        raise ValueError(f"missing target column: {target_col}")
    if season_col not in frame.columns:
        raise ValueError(f"missing season column: {season_col}")

    target = pd.to_numeric(frame[target_col], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{target_col} must be non-null binary 0/1")

    season = pd.to_numeric(frame[season_col], errors="coerce")
    if season.isna().any():
        raise ValueError(f"{season_col} contains non-numeric values")
    missing = sorted(EXPECTED_SEASONS - set(season.astype(int).unique()))
    if missing:
        raise ValueError(f"missing seasons: {missing}")
    return target, season


def prepare_dataset(
    root: str | Path = ".",
    target_col: str = "control_success",
    season_col: str = "season",
    force: bool = False,
) -> dict:
    root = Path(root)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    output = processed / CACHE_NAME

    if output.exists() and not force:
        frame = pd.read_parquet(output, engine="pyarrow")
        target, season = _validate_frame(
            frame,
            target_col=target_col,
            season_col=season_col,
        )
        return {
            "processed_file": str(output.relative_to(root)),
            "format": "parquet",
            "rows": int(len(frame)),
            "seasons": sorted(season.astype(int).unique().tolist()),
            "target_mean": float(target.mean()),
            "reused": True,
        }

    train_csv = _find_train_csv(root, target_col)
    frame = pd.read_csv(train_csv, low_memory=False)
    target, season = _validate_frame(
        frame,
        target_col=target_col,
        season_col=season_col,
    )

    tmp = processed / "train.parquet.part"
    tmp.unlink(missing_ok=True)
    frame.to_parquet(
        tmp,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    tmp.replace(output)

    # Validate the cache before any expensive training starts.
    check = pd.read_parquet(output, engine="pyarrow")
    if len(check) != len(frame) or list(check.columns) != list(frame.columns):
        raise RuntimeError("Parquet round-trip validation failed")
    del check

    manifest = {
        "source_csv": str(train_csv),
        "processed_file": str(output.relative_to(root)),
        "format": "parquet",
        "compression": "zstd",
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "target_mean": float(target.mean()),
        "season_rows": {
            str(y): int((season.astype(int) == y).sum())
            for y in sorted(EXPECTED_SEASONS)
        },
    }
    (processed / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_frame(config: dict) -> pd.DataFrame:
    path = Path(config["paths"]["processed_file"])
    if not path.exists():
        legacy = path.parent / "train.pkl"
        suffix = (
            f" Legacy pickle exists at {legacy}, but Bitaboost does not load pickle caches."
            if legacy.exists()
            else ""
        )
        raise FileNotFoundError(
            f"{path} not found; run python scripts/prepare_data.py first.{suffix}"
        )
    if path.suffix != ".parquet":
        raise ValueError(f"Bitaboost processed cache must be Parquet, got: {path}")
    return pd.read_parquet(path, engine="pyarrow")
