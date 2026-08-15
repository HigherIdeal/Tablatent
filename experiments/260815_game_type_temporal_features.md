# 2026-08-15 Temporal Game-Type Feature Experiment

## Motivation

The drift diagnostic showed that the historical `game_type` effect changes sign around 2023. Raw `game_type` therefore risks teaching CatBoost a stale cross-season relationship.

This experiment compares four representations while keeping the canonical feature policy and CatBoost screening budget fixed.

## Variants

- `raw_game_type`: canonical feature set with raw categorical `game_type`.
- `drop_game_type`: raw `game_type` removed.
- `prev1_effect`: raw `game_type` removed; add previous-season `game_type` effect.
- `prev12_ewma_effect`: raw `game_type` removed; add `2/3 * prev1 + 1/3 * prev2` season-lagged effect.

The effect for game type `g` in source season `s` is:

```text
P(control_success | game_type=g, season=s)
- P(control_success | season=s)
```

For a row in season `t`, only seasons before `t` are used. No current-season target is used to construct the feature.

Thus, for validation 2024, `prev1_effect` uses 2023 labels; for a future 2025 submission, the same construction would use 2024 labels.

## Run

```powershell
python scripts/run_game_type_temporal_features.py --config configs/default.yaml
```

Defaults:

- folds: 2023, 2024
- iterations: 200
- task type: GPU

## Outputs

```text
outputs/game_type_temporal_features/
  fold_results.csv
  summary.csv
  feature_sets.json
  run_config.json
```

The console reports Brier, competition-style score, AUC, and prediction standard deviation. Summary deltas are reported against both the raw-`game_type` and dropped-`game_type` baselines.
