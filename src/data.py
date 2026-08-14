from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link"
EXPECTED_SEASONS = set(range(2019, 2025))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown이 없습니다. pip install -r configs/requirements.txt 를 먼저 실행하세요.") from exc
    temp = destination.with_suffix(destination.suffix + ".part")
    temp.unlink(missing_ok=True)
    result = gdown.download(url=url, output=str(temp), quiet=False, fuzzy=True)
    if not result or not temp.exists() or temp.stat().st_size == 0:
        raise RuntimeError("Google Drive 다운로드에 실패했습니다. 링크 공개 범위와 네트워크를 확인하세요.")
    temp.replace(destination)
    return destination


def _safe_extract(archive: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    root = directory.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = (directory / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"안전하지 않은 ZIP 경로: {member.filename}")
            zf.extractall(directory)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = (directory / member.name).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"안전하지 않은 TAR 경로: {member.name}")
            tf.extractall(directory, filter="data")
    else:
        shutil.copy2(archive, directory / "train.csv")


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"CSV 인코딩을 읽을 수 없습니다: {path}") from last_error


def _pick_train_csv(files: list[Path], target_col: str) -> Path:
    candidates = []
    for path in files:
        try:
            columns = list(_read_csv(path, nrows=3).columns)
            score = (100 if target_col in columns else 0) + (10 if "train" in path.name.lower() else 0) + path.stat().st_size / 1e9
            candidates.append((score, path, columns))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError("압축 해제 결과에서 읽을 수 있는 CSV를 찾지 못했습니다.")
    score, path, columns = max(candidates, key=lambda x: x[0])
    if target_col not in columns:
        raise ValueError(f"{target_col!r} 컬럼이 있는 CSV를 찾지 못했습니다. 발견 파일: {[p.name for p in files]}")
    return path


def _infer_season(df: pd.DataFrame, preferred: str) -> tuple[pd.Series, str]:
    names = [preferred, "season", "year", "game_year", "game_date", "date", "game_dt"]
    for name in names:
        if name not in df.columns:
            continue
        values = df[name]
        if "date" in name or name.endswith("_dt"):
            season = pd.to_datetime(values, errors="coerce").dt.year
        else:
            season = pd.to_numeric(values, errors="coerce")
        observed = set(season.dropna().astype(int).unique())
        if observed & EXPECTED_SEASONS:
            return season.astype("Int64"), name
    raise ValueError("시즌 컬럼을 찾지 못했습니다. season/year/game_date 중 하나가 필요합니다.")


def prepare(root: str | Path = ".", target_col: str = "control_success", season_col: str = "season", force: bool = False) -> dict:
    root = Path(root)
    raw_dir, processed_dir = root / "data" / "raw", root / "data" / "processed"
    archive = download(DATA_URL, raw_dir / "dataset_download", force=force)
    extracted = raw_dir / "extracted"
    if force and extracted.exists():
        shutil.rmtree(extracted)
    if not extracted.exists() or not any(extracted.rglob("*.csv")):
        _safe_extract(archive, extracted)
    csv_files = sorted(extracted.rglob("*.csv"))
    train_path = _pick_train_csv(csv_files, target_col)
    df = _read_csv(train_path)
    if df.empty or target_col not in df:
        raise ValueError("train.csv가 비었거나 target 컬럼이 없습니다.")
    target = pd.to_numeric(df[target_col], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{target_col}은 결측 없는 0/1이어야 합니다.")
    season, source_season_col = _infer_season(df, season_col)
    df[season_col] = season
    missing = sorted(EXPECTED_SEASONS - set(season.dropna().astype(int).unique()))
    if missing:
        raise ValueError(f"필수 시즌이 없습니다: {missing}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized = processed_dir / "train.csv"
    df.to_csv(normalized, index=False, encoding="utf-8")
    season_rows = {}
    for year in sorted(EXPECTED_SEASONS):
        part = df.loc[df[season_col] == year]
        part.to_csv(processed_dir / f"train_{year}.csv", index=False, encoding="utf-8")
        season_rows[str(year)] = int(len(part))
    manifest = {
        "source_url": DATA_URL,
        "download_sha256": _sha256(archive),
        "source_csv": str(train_path.relative_to(root)),
        "normalized_csv": str(normalized.relative_to(root)),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target": target_col,
        "season_column": season_col,
        "season_source_column": source_season_col,
        "season_rows": season_rows,
    }
    with (processed_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="LG Aimers 데이터 다운로드 및 시즌 분할")
    parser.add_argument("--root", default=".")
    parser.add_argument("--target", default="control_success")
    parser.add_argument("--season-col", default="season")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.root, args.target, args.season_col, args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
