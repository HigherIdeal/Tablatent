from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.stage2_bilinear import train_stage2 as train_bilinear
from src.stage2_linear import train_stage2 as train_linear
from src.stage2_mlp import train_stage2 as train_mlp
from src.utils import load_config


HEADS = {
    "bilinear": train_bilinear,
    "linear": train_linear,
    "mlp": train_mlp,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--head",
        choices=sorted(HEADS),
        default="bilinear",
        help="Stage2 probe. Default is the 2026-08-15 bilinear context-history experiment.",
    )
    args = parser.parse_args()
    HEADS[args.head](load_config(ROOT / args.config))


if __name__ == "__main__":
    main()
