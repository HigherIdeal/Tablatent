from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.data import prepare_dataset
from src.pipeline import evaluate_stage1, train_stage1
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    prepare_dataset(
        root=ROOT,
        target_col=config["data"]["target_col"],
        season_col=config["data"]["season_col"],
        force=args.force_data,
    )
    train_stage1(config)
    evaluate_stage1(config)


if __name__ == "__main__":
    main()
