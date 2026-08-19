from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bitaboost.research_audit.four_axes import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot audit of four research axes: conditional shift, latent pitcher state, Trackman mechanics, and cold-start information value"
    )
    parser.add_argument("--config", default="experiments/configs/research_audit_four_axes.yaml")
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args.config)


if __name__ == "__main__":
    main()
