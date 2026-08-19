from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.cycle2.temporal_trackman_reliability import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cycle 2: strict temporal transfer + Trackman mechanics + causal reliability routing"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/cycle2_temporal_trackman_reliability.yaml",
    )
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args.config)


if __name__ == "__main__":
    main()
