#!/usr/bin/env python3
"""Download a portable local snapshot of Qwen/Qwen3-1.7B for offline inference.

Run this ONCE on a machine with internet access. The inference script never
contacts Hugging Face and loads only from the resulting local directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "Qwen3-1.7B"
REPO_ID = "Qwen/Qwen3-1.7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch/tag/commit. Use a commit SHA for strict reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(REPO_ID, revision=args.revision)
    resolved_revision = str(info.sha)
    print(f"repo: {REPO_ID}")
    print(f"requested_revision: {args.revision}")
    print(f"resolved_revision: {resolved_revision}")
    print(f"output: {output}")

    snapshot_download(
        repo_id=REPO_ID,
        revision=resolved_revision,
        local_dir=str(output),
    )

    required_any = ["model.safetensors", "model.safetensors.index.json"]
    if not (output / "config.json").is_file():
        raise RuntimeError("Downloaded snapshot has no config.json")
    if not any((output / name).is_file() for name in required_any):
        raise RuntimeError("Downloaded snapshot has no safetensors model weights")
    if not (output / "tokenizer_config.json").is_file():
        raise RuntimeError("Downloaded snapshot has no tokenizer_config.json")

    manifest = {
        "repo_id": REPO_ID,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "portable_model_dir": str(output),
        "offline_inference_command": (
            "python scripts/run_qwen3_rag.py --model-dir models/Qwen3-1.7B "
            "--query-season 2024 --limit 100"
        ),
    }
    (output / "tablatent_model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("download complete; copy this entire directory to the offline machine")


if __name__ == "__main__":
    main()
