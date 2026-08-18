from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

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
    import gdown
    tmp = raw / "dataset_download.part"
    tmp.unlink(missing_ok=True)
    result = gdown.download(DATA_URL, str(tmp), quiet=False, fuzzy=True)
    if not result or not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError("Google Drive dataset download failed")
    tmp.replace(destination)
    return destination


def _extract_train(archive: Path, out_dir: Path, target_col: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_vipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.infolist() if not m.is_dir() and m.filename.lower().endswith(".csv")]
            members.sort(key=lambda m: (Path(m.filename).name.lower() != "train.csv", len(m.filename)))
            for member in members:
                path = out_dir / Path(member.filename).name
                with zf.open(member) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                try:
                    if target_col in pd.read_csv(path, nrows=2, low_memory=False).columns:
                        return path
                except Exception:
                    pass
                path.unlink(missing_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]
            members.sort(key=lambda m: (Path(m.name).name.lower() != "train.csv", len(m.name)))
            for member in members:
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                path = out_dir / Path(member.name).name
                with path.open("wb") as dst:
                    shutil.copyfileobj(fh, dst)
                try:
                    if target_col in pd.read_csv(path, nrows=2, low_memory=False).columns:
                        return path
                except Exception:
                    pass
                path.unlink(missing_ok=True)
    try:
        if target_col in pd.read_csv(archive, nrows=2, low_memory=False).columns:
            path = out_dir / "train.csv"
            shutil.copy2(archive, path)
            return path
    except Exception:
        pass
    raise FileNotFoundError(f"could not find train CSV containing {target_col!r}")


def prepare_dataset(
    root: str | Path = ".",
    target_col: str = "control_success",
    season_col: str = "season",
    force: bool = False,
) -> dict:
    root = Path(root)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    output = processed / "train.pkl"
    if output.exists() and not force:
        frame = pd.read_pickle(output)
        return {
            "processed_file": str(output.relative_to(root)),
            "rows": int(len(frame)),
            "seasons": sorted(pd.to_numeric(frame[season_col]).astype(int).unique().tolist()),
            "target_mean": float(pd.to_numeric(frame[target_col]).mean()),
            "reused": True,
        }

    archive = download_dataset(root, force=force)
    extract_dir = root / "data" / "raw" / "extracted"
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    train_csv = _extract_train(archive, extract_dir, target_col)
    frame = pd.read_csv(train_csv, low_memory=False)

    target = pd.to_numeric(frame[target_col], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{target_col} must be non-null binary 0/1")
    season = pd.to_numeric(frame[season_col], errors="coerce")
    if season.isna().any():
        raise ValueError(f"{season_col} contains non-numeric values")
    missing = sorted(EXPECTED_SEASONS - set(season.astype(int).unique()))
    if missing:
        raise ValueError(f"missing seasons: {missing}")

    frame.to_pickle(output)
    manifest = {
        "source_url": DATA_URL,
        "download_sha256": _sha256(archive),
        "source_csv": str(train_csv.relative_to(root)),
        "processed_file": str(output.relative_to(root)),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "target_mean": float(target.mean()),
        "season_rows": {str(y): int((season.astype(int) == y).sum()) for y in sorted(EXPECTED_SEASONS)},
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_frame(config: dict) -> pd.DataFrame:
    path = Path(config["paths"]["processed_file"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python scripts/prepare_data.py first")
    return pd.read_pickle(path)
