from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import train_stage1
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="독립적인 현재 상황/과거 이력 latent 학습")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_stage1(load_config(args.config))


if __name__ == "__main__":
    main()

