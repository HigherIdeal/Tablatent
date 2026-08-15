# 2026-08-15 Game-Type Drift Diagnostic

## Why this experiment

The 200-tree focused ablation showed that the large `drop_game_phase` effect is almost entirely explained by `game_type`:

- 2023: dropping `game_type` improves Brier strongly.
- 2024: dropping `game_type` hurts Brier slightly.
- dropping only `inning` or `top_bottom` does not reproduce the 2023 improvement.

Before permanently removing `game_type`, diagnose whether its distribution or conditional target relationship changes by season.

## Run

```powershell
python scripts/analyze_game_type_drift.py --config configs/default.yaml
```

This experiment does not train CatBoost and should run much faster than an ablation.

## Outputs

```text
outputs/game_type_drift/
  season_summary.csv
  season_game_type.csv
  share_by_season.csv
  success_rate_by_season.csv
  forward_validation.csv
```

`season_game_type.csv` reports, for every season and game type:

- rows
- within-season share
- control-success rate

`forward_validation.csv` treats every year as a forward validation year. It compares:

1. a constant historical train prior, and
2. historical mean success rate by `game_type`.

The key field is:

```text
delta_game_type_vs_train_prior
```

Interpretation:

- negative: historical `game_type` information helps the next year;
- positive: historical `game_type` information hurts the next year.

The diagnostic also reports `delta_rebased_vs_oracle_mean`, which removes global target-rate drift by rebasing historical game-type effects onto the actual validation-year target mean. This uses validation labels and is diagnostic only; it must never be used for inference or submission.

## Decision after the run

Use the output to decide among three next steps:

- keep `game_type` if its conditional effect is temporally stable;
- remove it if the conditional effect itself reverses or is unstable;
- retain only a drift-robust transformation if the category effect is stable but the global target rate shifts.
