from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex1.backward_suite import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EX1: automatic four-stage pure backward pitcher-state research suite."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex1_backward_suite.yaml",
        help="EX1 backward suite config",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
