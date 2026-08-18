from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex1.pitcher_season_backward import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EX1-B: evaluate a pure pitcher-season backward model "
            "(future season state -> previous season state), without SAFE blending."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex1_pitcher_season_backward.yaml",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
