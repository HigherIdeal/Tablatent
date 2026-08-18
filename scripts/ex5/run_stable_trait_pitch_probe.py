from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex5.stable_trait_pitch_probe import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EX5: test whether EX4 bidirectionally stable pitcher traits carry "
            "standalone pitch-level control_success information."
        )
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/ex5_stable_trait_pitch_probe.yaml",
        help="EX5 experiment config",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
