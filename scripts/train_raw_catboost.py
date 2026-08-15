from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.raw_catboost import train_raw_catboost
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost on the canonical exactly de-duplicated raw feature set."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_raw_catboost(load_config(ROOT / args.config))


if __name__ == "__main__":
    main()
