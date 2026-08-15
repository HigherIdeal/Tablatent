# 2026-08-15 Latest-Season Contribution

## Purpose

Before moving to calibration, directly test whether the immediately previous season actually helps next-year prediction.

This is different from the recent-window experiment. A recent window removes old data; this experiment keeps the old data and removes only the newest available training season.

The key comparison for the current question is:

```text
2019-2023 -> validate 2024   (full history)
2019-2022 -> validate 2024   (remove 2023 only)
```

This directly measures whether 2023, the apparent regime-change year, improved or damaged 2024 generalization.

## Default experiment

Validation folds:

```text
2022
2023
2024
```

For each fold:

```text
exclude_latest=0  use every prior season
exclude_latest=1  remove only the immediately previous season
```

Both canonical feature variants are tested:

```text
raw_game_type
 drop_game_type
```

The CatBoost screening setup remains unchanged at 200 trees by default.

## Run

```powershell
python scripts/run_latest_season_contribution.py --config configs/default.yaml
```

To probe farther back, for example removing the latest one or two prior seasons:

```powershell
python scripts/run_latest_season_contribution.py --config configs/default.yaml --exclude-latest 0,1,2
```

## Outputs

```text
outputs/latest_season_contribution/
  fold_results.csv
  summary.csv
  latest_season_contribution.csv
  run_config.json
```

The main diagnostic is:

```text
delta_brier_vs_full_history
```

Interpretation:

- positive: removing the latest season worsens Brier, so that latest season helped;
- negative: removing the latest season improves Brier, so that latest season hurt.

For fold 2024 with `exclude_latest=1`, the removed season is exactly 2023.

## Decision

If removing 2023 improves 2024, then the 2023 regime shift is not automatically useful training information and we should investigate season-specific weighting or conditional handling more carefully.

If removing 2023 hurts 2024, then the apparent recovery in the 2024 fold genuinely benefits from having observed 2023; in that case the main remaining issue is more likely probability calibration than whether to discard 2023.
