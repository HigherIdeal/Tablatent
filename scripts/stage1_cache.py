from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config

DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/Tablatent/stage1_cache")
CHUNK_SIZE = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _copy_with_hash(src: Path, dst: Path) -> tuple[int, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    digest = hashlib.sha256()
    size = 0
    with src.open("rb") as fin, tmp.open("wb") as fout:
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break
            fout.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    os.replace(tmp, dst)
    try:
        shutil.copystat(src, dst, follow_symlinks=True)
    except OSError:
        pass
    return size, digest.hexdigest()


def _maybe_mount_drive(drive_dir: Path) -> None:
    if drive_dir.exists():
        return
    if not str(drive_dir).startswith("/content/drive/"):
        drive_dir.mkdir(parents=True, exist_ok=True)
        return
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError(
            f"Google Drive가 마운트되어 있지 않습니다: {drive_dir}. "
            "Colab이 아니면 --drive-dir로 접근 가능한 경로를 지정하세요."
        ) from exc
    drive.mount("/content/drive")
    drive_dir.mkdir(parents=True, exist_ok=True)


def _cache_files(config: dict, exclude_data: bool) -> list[Path]:
    output_dir = ROOT / config["paths"]["output_dir"]
    files = [
        output_dir / "latents" / "context.npy",
        output_dir / "latents" / "history.npy",
        output_dir / "latents" / "context_logvar.npy",
        output_dir / "latents" / "history_logvar.npy",
        output_dir / "checkpoints" / "stage1_context.pt",
        output_dir / "checkpoints" / "stage1_history.pt",
        output_dir / "checkpoints" / "preprocessors.joblib",
        output_dir / "logs" / "stage1_training.json",
    ]
    if not exclude_data:
        files.append(ROOT / config["paths"]["processed_file"])
    return files


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def push(config_path: Path, drive_dir: Path, exclude_data: bool) -> None:
    _maybe_mount_drive(drive_dir)
    config = load_config(config_path)
    files = _cache_files(config, exclude_data)
    missing = [p for p in files if not p.exists()]
    if missing:
        listing = "\n".join(f"  - {_relative(p)}" for p in missing)
        raise FileNotFoundError(
            "Stage1 cache에 필요한 파일이 없습니다. Stage1 학습 완료 여부를 확인하세요:\n"
            + listing
        )

    entries = []
    print(f"[cache push] destination: {drive_dir}")
    for src in files:
        rel = _relative(src)
        dst = drive_dir / rel
        size, sha = _copy_with_hash(src, dst)
        entries.append({"path": rel, "size": size, "sha256": sha})
        print(f"  pushed {rel} ({size / (1024 ** 2):.1f} MiB)")

    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": "HigherIdeal/Tablatent",
        "config_path": _relative(config_path),
        "config_sha256": _sha256(config_path),
        "includes_processed_data": not exclude_data,
        "files": entries,
    }
    manifest_tmp = drive_dir / "manifest.json.tmp"
    manifest_path = drive_dir / "manifest.json"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    print(f"[cache push] complete: {len(entries)} files")


def pull(config_path: Path, drive_dir: Path, exclude_data: bool) -> None:
    _maybe_mount_drive(drive_dir)
    manifest_path = drive_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"cache manifest가 없습니다: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    current_config_sha = _sha256(config_path)
    cached_config_sha = manifest.get("config_sha256")
    if cached_config_sha and current_config_sha != cached_config_sha:
        print(
            "[cache pull] WARNING: 현재 config와 Stage1 cache 생성 당시 config의 SHA256이 다릅니다. "
            "latent 호환성을 확인하세요."
        )

    entries = manifest.get("files", [])
    if not entries:
        raise RuntimeError("cache manifest에 files 항목이 없습니다.")

    config = load_config(config_path)
    processed_rel = str(config["paths"]["processed_file"])
    print(f"[cache pull] source: {drive_dir}")
    restored = 0
    skipped = 0
    for entry in entries:
        rel = str(entry["path"])
        if exclude_data and rel == processed_rel:
            print(f"  skipped data {rel}")
            continue
        src = drive_dir / rel
        dst = ROOT / rel
        expected_sha = str(entry["sha256"])
        expected_size = int(entry["size"])
        if not src.exists():
            raise FileNotFoundError(f"cache file이 없습니다: {src}")
        if src.stat().st_size != expected_size:
            raise RuntimeError(f"cache file size mismatch: {rel}")

        if dst.exists() and dst.stat().st_size == expected_size and _sha256(dst) == expected_sha:
            print(f"  already valid {rel}")
            skipped += 1
            continue

        size, sha = _copy_with_hash(src, dst)
        if size != expected_size or sha != expected_sha:
            raise RuntimeError(f"복원 후 SHA256 검증 실패: {rel}")
        print(f"  restored {rel} ({size / (1024 ** 2):.1f} MiB)")
        restored += 1

    print(f"[cache pull] complete: restored={restored}, already_valid={skipped}")
    if exclude_data:
        print("[cache pull] processed data는 제외했습니다. Stage2 전에 data/processed/train.pkl을 별도로 준비하세요.")
    else:
        print("[cache pull] Stage2 resume ready.")
        print("python scripts/train_stage2.py --config configs/default.yaml --head bilinear")


def status(drive_dir: Path) -> None:
    _maybe_mount_drive(drive_dir)
    manifest_path = drive_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[cache status] no manifest: {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[cache status] created_at_utc={manifest.get('created_at_utc')}")
    print(f"[cache status] includes_processed_data={manifest.get('includes_processed_data')}")
    for entry in manifest.get("files", []):
        rel = str(entry["path"])
        local = ROOT / rel
        state = "missing"
        if local.exists():
            state = "size-ok" if local.stat().st_size == int(entry["size"]) else "size-mismatch"
        print(f"  {state:13s} {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist/restore Stage1 VAE latents and checkpoints through Google Drive."
    )
    parser.add_argument("action", choices=["push", "pull", "status"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--drive-dir",
        default=str(DEFAULT_DRIVE_DIR),
        help="Google Drive cache directory (default: /content/drive/MyDrive/Tablatent/stage1_cache)",
    )
    parser.add_argument(
        "--exclude-data",
        action="store_true",
        help="Do not push/pull data/processed/train.pkl. Default keeps it for one-command Stage2 resume.",
    )
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    drive_dir = Path(args.drive_dir).expanduser()

    if args.action == "push":
        push(config_path, drive_dir, args.exclude_data)
    elif args.action == "pull":
        pull(config_path, drive_dir, args.exclude_data)
    else:
        status(drive_dir)


if __name__ == "__main__":
    main()
