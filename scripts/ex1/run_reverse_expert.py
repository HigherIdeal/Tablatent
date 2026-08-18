from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bitaboost.ex1 import reverse_expert as reverse_core
from bitaboost.ex1.frozen_baseline import load_frozen_baseline

# Historical SAFE predictions.npz contains game_type as an object array.  Keep
# allow_pickle=False and replace only the EX1 frozen-baseline loader; the forward
# predictor itself remains completely frozen and is never retrained here.
reverse_core._load_frozen_baseline = load_frozen_baseline
run = reverse_core.run


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
