from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex4.bidirectional_traits import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EX4: discover pitcher-season traits that remain predictable in both temporal directions."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex4_bidirectional_stable_traits.yaml",
        help="EX4 experiment config",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
