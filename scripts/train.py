from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bitaboost training entry point. The initial implementation is the frozen safe baseline."
    )
    ap.add_argument(
        "--gpus",
        default="2",
        help="Physical GPU id. Default=2 (single RTX 4090 on iclab4GPU).",
    )
    ap.add_argument("--output", default="dist/train_latest.zip")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "baseline_train.py"),
        "--gpus", args.gpus,
        "--output", args.output,
    ]
    if args.smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
