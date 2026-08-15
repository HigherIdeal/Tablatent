from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the canonical ablation runner from the sibling script so this follow-up
# stays on exactly the same preprocessing, CatBoost parameters, metrics, and
# feature policy as the completed canonical screening experiment.
import run_catboost_ablation as core
from src.canonical_features import PITCHER_TEAM_WIN_EXPECTANCY
from src.utils import load_config


# The 200-tree canonical screening showed that removing the entire game_phase
# group had by far the largest average improvement, but the effect was not
# consistent across both temporal folds. Decompose that group before pruning it.
core.GROUPS.update(
    {
        "inning": ["inning"],
        "top_bottom": ["top_bottom"],
        "game_type": ["game_type"],
        "inning_top_bottom": ["inning", "top_bottom"],
        "inning_game_type": ["inning", "game_type"],
        "top_bottom_game_type": ["top_bottom", "game_type"],
        # Near-zero groups from the first canonical screening. Test their joint
        # removal once, rather than spending another full broad ablation on them.
        "weak_context": [
            "game_month",
            "game_dayofweek",
            "base_state",
            PITCHER_TEAM_WIN_EXPECTANCY,
        ],
    }
)

FOLLOWUP_VARIANTS = [
    "reference_canonical",
    "drop_inning",
    "drop_top_bottom",
    "drop_game_type",
    "drop_inning_top_bottom",
    "drop_inning_game_type",
    "drop_top_bottom_game_type",
    "drop_game_phase",
    "drop_calendar",
    "drop_base_state",
    "drop_win_expectancy",
    "drop_weak_context",
]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Focused follow-up ablation: decompose inning/top_bottom/game_type and "
            "re-check near-zero context features."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", default="2023,2024")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)

    # Keep the previous broad-screening CSVs intact.
    original_output = Path(config["paths"]["output_dir"])
    config["paths"]["output_dir"] = str(original_output / "game_phase_followup")

    print("[Follow-up] focused variants:")
    print("  " + ", ".join(FOLLOWUP_VARIANTS))
    print("[Follow-up] default screening budget: 200 trees")

    core.run_ablation(
        config=config,
        folds=parse_ints(args.folds),
        variants=list(FOLLOWUP_VARIANTS),
        iterations=args.iterations,
        task_type=args.task_type,
        devices=args.devices,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
