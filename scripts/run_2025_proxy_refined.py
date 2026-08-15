from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_2025_proxy_validation as core


DEFAULT_ITERATIONS_GRID = "250,300,400,500,600,800"
DEFAULT_ALPHA_STEP = 0.025
DEFAULT_OUTPUT_DIR = "outputs/proxy_2025_validation_refined"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Second-pass 2025 proxy screen. Extends CatBoost prefix search to 800 trees and "
            "uses a finer 0.025 blend-alpha grid while preserving the same three temporal folds."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--iterations-grid", default=DEFAULT_ITERATIONS_GRID)
    parser.add_argument("--alpha-step", type=float, default=DEFAULT_ALPHA_STEP)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    forwarded = [
        "run_2025_proxy_validation.py",
        "--config", args.config,
        "--iterations-grid", args.iterations_grid,
        "--alpha-step", str(args.alpha_step),
        "--task-type", args.task_type,
        "--devices", args.devices,
        "--verbose", str(args.verbose),
        "--output-dir", args.output_dir,
    ]
    previous_argv = sys.argv
    try:
        sys.argv = forwarded
        core.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
