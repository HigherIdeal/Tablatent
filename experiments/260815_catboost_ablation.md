# 2026-08-15 CatBoost Feature Ablation

## Objective

Use the previous ~790-point CatBoost experiment as historical evidence, but run all new CatBoost training/ablation on a **canonical feature set with deterministic redundancy removed or normalized**.

The first ablation established two identity decisions:

- `pitcher_id`: adding it hurt both 2023 and 2024 -> exclude.
- `batter_id`: removing it improved both 2023 and 2024 -> exclude.

Team IDs remain until further ablation is decisive.

## Canonical redundancy policy

### Score state

Exact relations:

```text
run_total_before = run_top_before + run_bot_before
score_diff_home = run_bot_before - run_top_before
score_diff_pitcher_team = +/- score_diff_home from top_bottom
```

Canonical representation:

```text
run_total_before
score_diff_home
top_bottom
```

Removed: `run_top_before`, `run_bot_before`, `score_diff_pitcher_team`.

### Runner state

`base_state` exactly encodes all three base occupancy flags and therefore also `num_runners_on`.

Canonical representation: `base_state`.

Removed: `runner_on_1b`, `runner_on_2b`, `runner_on_3b`, `num_runners_on`.

### Win expectancy

`home_win_expectancy` and `away_win_expectancy` are the same game-state concept from opposite team perspectives. Their complement can differ slightly because of rounding, but that small discrepancy is intentionally ignored.

They are converted into one pitcher-perspective feature:

```text
if top_bottom == "T":
    pitcher_team_win_expectancy = home_win_expectancy
else:  # "B"
    pitcher_team_win_expectancy = away_win_expectancy
```

Reason: in the top half the home team is pitching; in the bottom half the away team is pitching.

Canonical representation:

```text
pitcher_team_win_expectancy
li
```

Raw `home_win_expectancy` / `away_win_expectancy` are not fed to CatBoost separately. The maximum complement-rounding error is recorded as a diagnostic rather than treated as a failed invariant.

### History count

The audited training data satisfies:

```text
asof_pitcher_pitchmix_n == asof_pitcher_n
```

Therefore only `asof_pitcher_n` is kept.

### Deterministic engineered transforms

The older H2/J0-style pipeline duplicated raw information through EB, reliability, and log-count transforms. These are removed from the canonical model because they are determined by retained raw values plus the fold training prior:

```text
pitcher_success_eb100 / eb500
batter_success_eb100 / eb500
pitcher_reliability_500
batter_reliability_500
pitcher_n_log
batter_n_log
pitchmix_n_log
```

### Not pruned

The fastball / breaking / offspeed rates remain because their sum is not exactly one for many rows, so removing one can discard an unrepresented remainder.

`src/canonical_features.py` contains strict invariant checks for exact assumptions. If a future dataset violates an exact relation used for pruning, training stops rather than silently discarding information.

## Current canonical reference

The canonical reference has 36 features before any ablation. It keeps:

- season/calendar/game phase
- count
- minimal score state
- `base_state`
- pitcher-team win expectancy / LI
- pitcher/batter handedness
- pitcher/batter team IDs
- pitcher long-term profile
- pitcher recent 1/3/5-game profile
- batter profile
- pitch-mix rates

It excludes `row_id`, `pitcher_id`, `batter_id`, exact redundant official columns, duplicated win-expectancy orientation, and deterministic engineered duplicates.

## Validation protocol

Default folds:

```text
2019-2022 -> 2023
2019-2023 -> 2024
```

Ablation uses a fixed tree count and fixed CatBoost hyperparameters so the intended variable is the feature group. Primary metric is Brier score; AUC and prediction spread are diagnostics.

Fine-grained context ablations:

- calendar
- game phase
- count
- score
- base state
- pitcher-team win expectancy
- leverage (`pitcher_team_win_expectancy + li`)
- handedness
- team IDs
- pitcher profile
- pitcher recent form
- batter profile
- pitch mix

`pitcher_id` and `batter_id` are add-back controls rather than baseline inputs.

## Run

```powershell
python scripts/run_catboost_ablation.py --config configs/default.yaml
```

Outputs:

```text
outputs/catboost_ablation_canonical/fold_results.csv
outputs/catboost_ablation_canonical/summary.csv
outputs/catboost_ablation_canonical/feature_sets.json
outputs/catboost_ablation_canonical/run_config.json
```

The normal raw CatBoost path uses the same canonical policy:

```powershell
python scripts/train_raw_catboost.py --config configs/default.yaml
```

and writes to `outputs/raw_catboost_canonical/`.
