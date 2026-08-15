from __future__ import annotations

import numpy as np
import pandas as pd


# Canonical CatBoost feature policy.
# Keep one representation when another official/engineered column can be
# reconstructed exactly from it. Near-duplicates are intentionally retained.
CANONICAL_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before",
    # Score state: these two recover run_top/run_bot exactly, and with top_bottom
    # also recover score_diff_pitcher_team.
    "run_total_before", "score_diff_home",
    # Base occupancy: base_state exactly recovers all three runner flags and count.
    "base_state",
    # Win expectancies are only approximately complementary because of rounding,
    # so both remain. LI is separate information.
    "home_win_expectancy", "away_win_expectancy", "li",
    # Entity IDs: pitcher_id/batter_id are excluded by the completed ablation;
    # team IDs remain until their own evidence is decisive.
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    # Long-term / recent history.
    "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    # pitchmix_n is exactly identical to asof_pitcher_n in the audited data.
    # The three rates are retained: their sum is not exactly one for many rows.
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

CANONICAL_CATEGORICAL = [
    "game_month", "game_dayofweek", "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]

# Exact deterministic duplicates that must not enter the canonical model.
EXACT_REDUNDANT_OFFICIAL = {
    "run_top_before": "recoverable from run_total_before and score_diff_home",
    "run_bot_before": "recoverable from run_total_before and score_diff_home",
    "score_diff_pitcher_team": "recoverable from score_diff_home and top_bottom",
    "runner_on_1b": "recoverable from base_state",
    "runner_on_2b": "recoverable from base_state",
    "runner_on_3b": "recoverable from base_state",
    "num_runners_on": "recoverable from base_state",
    "asof_pitcher_pitchmix_n": "exactly identical to asof_pitcher_n in audited train data",
}

# These were used by the older H2/J0-style experiment, but are exact transforms
# of retained raw columns plus the fold training prior. They are excluded from
# the canonical model to avoid duplicate encodings.
EXACT_REDUNDANT_ENGINEERED = {
    "pitcher_success_eb100": "deterministic from pitcher rate, pitcher n, and fold prior",
    "pitcher_success_eb500": "deterministic from pitcher rate, pitcher n, and fold prior",
    "batter_success_eb100": "deterministic from batter rate, batter n, and fold prior",
    "batter_success_eb500": "deterministic from batter rate, batter n, and fold prior",
    "pitcher_reliability_500": "deterministic from asof_pitcher_n",
    "batter_reliability_500": "deterministic from asof_batter_n",
    "pitcher_n_log": "deterministic from asof_pitcher_n",
    "batter_n_log": "deterministic from asof_batter_n",
    "pitchmix_n_log": "deterministic from asof_pitcher_n because pitchmix_n == pitcher_n",
}

# Deliberately NOT pruned because the audit did not establish exact equality.
NON_EXACT_OVERLAPS = {
    "home_win_expectancy / away_win_expectancy": "approximately complementary; rounding breaks exact recovery",
    "fastball / breaking / offspeed rates": "sum is not exactly one for many rows; an unrepresented remainder can exist",
}

BASE_STATE_TO_FLAGS = {
    "___": (0, 0, 0),
    "1__": (1, 0, 0),
    "_2_": (0, 1, 0),
    "__3": (0, 0, 1),
    "12_": (1, 1, 0),
    "1_3": (1, 0, 1),
    "_23": (0, 1, 1),
    "123": (1, 1, 1),
}


def validate_canonical_schema(frame: pd.DataFrame) -> dict[str, int]:
    """Assert exact redundancy assumptions before training/ablation.

    This is intentionally strict: if a future dataset violates an invariant used
    for pruning, the run stops instead of silently discarding information.
    """
    required = set(CANONICAL_FEATURES) | set(EXACT_REDUNDANT_OFFICIAL)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns required by canonical feature audit: {missing}")

    mismatches: dict[str, int] = {}

    run_total = pd.to_numeric(frame["run_total_before"], errors="raise").to_numpy(np.int64)
    score_diff = pd.to_numeric(frame["score_diff_home"], errors="raise").to_numpy(np.int64)
    run_top = pd.to_numeric(frame["run_top_before"], errors="raise").to_numpy(np.int64)
    run_bot = pd.to_numeric(frame["run_bot_before"], errors="raise").to_numpy(np.int64)

    mismatches["run_total"] = int(np.count_nonzero(run_total != run_top + run_bot))
    mismatches["score_diff_home"] = int(np.count_nonzero(score_diff != run_bot - run_top))
    mismatches["score_parity"] = int(np.count_nonzero((run_total + score_diff) % 2 != 0))

    top_bottom = frame["top_bottom"].astype(str).to_numpy()
    pitcher_diff = pd.to_numeric(
        frame["score_diff_pitcher_team"], errors="raise"
    ).to_numpy(np.int64)
    expected_pitcher_diff = np.where(top_bottom == "T", score_diff, -score_diff)
    mismatches["score_diff_pitcher_team"] = int(
        np.count_nonzero(pitcher_diff != expected_pitcher_diff)
    )

    base = frame["base_state"].astype(str)
    unknown_base = int((~base.isin(BASE_STATE_TO_FLAGS)).sum())
    mismatches["unknown_base_state"] = unknown_base
    if unknown_base == 0:
        decoded = np.asarray([BASE_STATE_TO_FLAGS[x] for x in base], dtype=np.int8)
        r1 = pd.to_numeric(frame["runner_on_1b"], errors="raise").to_numpy(np.int8)
        r2 = pd.to_numeric(frame["runner_on_2b"], errors="raise").to_numpy(np.int8)
        r3 = pd.to_numeric(frame["runner_on_3b"], errors="raise").to_numpy(np.int8)
        nr = pd.to_numeric(frame["num_runners_on"], errors="raise").to_numpy(np.int8)
        mismatches["runner_on_1b"] = int(np.count_nonzero(decoded[:, 0] != r1))
        mismatches["runner_on_2b"] = int(np.count_nonzero(decoded[:, 1] != r2))
        mismatches["runner_on_3b"] = int(np.count_nonzero(decoded[:, 2] != r3))
        mismatches["num_runners_on"] = int(
            np.count_nonzero(decoded.sum(axis=1) != nr)
        )

    pitcher_n = pd.to_numeric(frame["asof_pitcher_n"], errors="raise").to_numpy(np.int64)
    pitchmix_n = pd.to_numeric(
        frame["asof_pitcher_pitchmix_n"], errors="raise"
    ).to_numpy(np.int64)
    mismatches["pitchmix_n"] = int(np.count_nonzero(pitchmix_n != pitcher_n))

    failed = {k: v for k, v in mismatches.items() if v != 0}
    if failed:
        raise ValueError(
            "Canonical feature pruning invariant failed; refusing to drop information: "
            + ", ".join(f"{k}={v}" for k, v in failed.items())
        )

    if len(CANONICAL_FEATURES) != len(set(CANONICAL_FEATURES)):
        raise ValueError("CANONICAL_FEATURES contains duplicate column names")

    return mismatches
