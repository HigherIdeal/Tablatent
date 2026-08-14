from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.stage2 import build_stage2_dataset
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="2024 holdout용 Stage 2 feature도 생성합니다. 기본값은 생성하지 않습니다.",
    )
    args = parser.parse_args()
    build_stage2_dataset(
        load_config(ROOT / args.config),
        include_test=args.include_test,
    )


if __name__ == "__main__":
    main()
