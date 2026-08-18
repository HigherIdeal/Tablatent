from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex2.hypothesis_backward import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EX2: candidate-label-conditioned backward reconstruction. "
            "Assume y=0 and y=1 separately, then score which hypothesis better reconstructs the known past state."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex2_hypothesis_backward.yaml",
        help="EX2 experiment config",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
