from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex1.reverse_expert import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EX1: train a future-to-past reverse expert while keeping SAFE982 forward predictions frozen."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex1_reverse_expert.yaml",
        help="EX1 experiment config",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
