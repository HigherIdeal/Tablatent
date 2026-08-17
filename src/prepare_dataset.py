#!/usr/bin/env python3
"""Download and organize the official LG Aimers 9 dataset.

This script deliberately stops at producing immutable raw files. Feature
engineering and model-ready preprocessing belong in separate scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_URL = (
    "https://drive.google.com/file/d/"
    "1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link"
)
PRIMARY_FILES = {
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "trackman_history.csv",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download, extract, and organize the official dataset."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Official archive URL")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Dataset root (default: <repo>/data)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing local archive instead of downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing raw files with files from this archive",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive under data/downloads",
    )
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_google_drive_url(url: str) -> bool:
    return urllib.parse.urlparse(url).netloc.lower().endswith("drive.google.com")


def download(url: str, destination: Path) -> None:
    if is_google_drive_url(url):
        try:
            import gdown  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Google Drive 다운로드에는 gdown이 필요합니다. "
                "먼저 `python -m pip install gdown`을 실행하세요."
            ) from exc
        result = gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True)
        if not result or not destination.is_file():
            raise RuntimeError("Google Drive download failed")
        return

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)


def ensure_safe_member(base: Path, member_name: str) -> None:
    target = (base / member_name).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise RuntimeError(f"Unsafe archive path: {member_name}")


def extract_archive(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                ensure_safe_member(destination, info.filename)
            bundle.extractall(destination)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                ensure_safe_member(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Archive links are not allowed: {member.name}")
            bundle.extractall(destination, filter="data")
        return

    raise RuntimeError(f"Unsupported archive format: {archive}")


def locate_primary_files(extracted: Path) -> dict[str, Path]:
    found: dict[str, list[Path]] = {name: [] for name in PRIMARY_FILES}
    for path in extracted.rglob("*"):
        if path.is_file() and path.name.lower() in found:
            found[path.name.lower()].append(path)

    return {
        name: sorted(paths, key=lambda path: (len(path.parts), path.as_posix()))[0]
        for name, paths in found.items()
        if paths
    }


def content_root(extracted: Path) -> Path:
    """Strip one archive-wide wrapper directory without flattening its contents."""
    children = list(extracted.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extracted


def install_raw_tree(extracted: Path, raw_dir: Path, force: bool) -> None:
    """Preserve every archive file, including docs and nested archives."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_root = content_root(extracted)
    sources = sorted(path for path in source_root.rglob("*") if path.is_file())
    if not sources:
        raise RuntimeError("The downloaded archive contains no files")

    for source in sources:
        relative = source.relative_to(source_root)
        target = raw_dir / relative
        if target.exists() and not force:
            if sha256(target) == sha256(source):
                print(f"[keep] {target} (identical)")
                continue
            raise FileExistsError(
                f"{target} already exists with different content; use --force to replace it"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"[write] {target} ({target.stat().st_size:,} bytes)")


def remove_content_duplicates(raw_dir: Path) -> None:
    """Keep one deterministic copy of each byte-identical raw file."""
    paths = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    paths.sort(key=lambda path: (len(path.relative_to(raw_dir).parts), path.as_posix()))

    kept_by_hash: dict[str, Path] = {}
    for path in paths:
        content_hash = sha256(path)
        kept = kept_by_hash.get(content_hash)
        if kept is None:
            kept_by_hash[content_hash] = path
            continue
        path.unlink()
        print(f"[deduplicate] removed {path}; identical to {kept}")

    directories = sorted(
        (path for path in raw_dir.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if not any(directory.iterdir()):
            directory.rmdir()


def write_manifest(raw_dir: Path, source_url: str, archive_hash: str) -> Path:
    entries = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        entries.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "archive_sha256": archive_hash,
        "files": entries,
    }
    destination = raw_dir / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return destination


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    raw_dir = data_dir / "raw"
    downloads_dir = data_dir / "downloads"
    data_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tablatent_dataset_") as temp_name:
        temp_dir = Path(temp_name)
        if args.archive:
            archive = args.archive.expanduser().resolve()
            if not archive.is_file():
                raise FileNotFoundError(archive)
            print(f"[source] local archive: {archive}")
        else:
            archive = temp_dir / "official_dataset.archive"
            print(f"[download] {args.url}")
            download(args.url, archive)

        archive_hash = sha256(archive)
        print(f"[archive sha256] {archive_hash}")
        extracted = temp_dir / "extracted"
        extracted.mkdir()
        extract_archive(archive, extracted)

        files = locate_primary_files(extracted)
        missing = sorted(PRIMARY_FILES - files.keys())
        if missing:
            print(
                "[warning] files not directly visible as CSV "
                f"(they may be nested archives): {', '.join(missing)}"
            )

        install_raw_tree(extracted, raw_dir, args.force)
        remove_content_duplicates(raw_dir)
        manifest = write_manifest(raw_dir, args.url, archive_hash)
        print(f"[manifest] {manifest}")

        if args.keep_archive and not args.archive:
            downloads_dir.mkdir(parents=True, exist_ok=True)
            saved_archive = downloads_dir / f"official_dataset_{archive_hash[:12]}.archive"
            shutil.copy2(archive, saved_archive)
            print(f"[archive kept] {saved_archive}")

    print("Dataset preparation complete. Raw files were not transformed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
