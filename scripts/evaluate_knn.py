from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

from src.knn_probability import evaluate_latent_knn
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--test",
        action="store_true",
        help="2023에서 best k를 고른 뒤 2024 holdout도 평가합니다.",
    )
    args = parser.parse_args()
    evaluate_latent_knn(load_config(ROOT / args.config), evaluate_test=args.test)


if __name__ == "__main__":
    main()
