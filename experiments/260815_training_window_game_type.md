# 2026-08-15 Training Window × Game-Type

## Purpose

The previous diagnostics showed two simultaneous temporal effects:

- overall `control_success` falls strongly across seasons;
- `game_type` changes regime: F is strongly positive relative to R through 2022, then negative in 2023–2024.

Season-lagged `game_type` effect features did not generalize, so the next question is whether old seasons themselves are hurting adaptation.

## Experiment

For each validation year, compare recent training windows:

- `all`: every earlier season;
- `3y`: most recent three earlier seasons;
- `2y`: most recent two earlier seasons;
- `1y`: immediately previous season only.

Within every window train two otherwise identical canonical CatBoost models:

- `raw_game_type`: keep `game_type`;
- `drop_game_type`: remove `game_type`.

Default validation folds are 2022, 2023, and 2024. Including 2022 gives a pre-regime-change control, while 2023 and 2024 test adaptation after the relationship flips.

If two requested windows are identical for a fold (for example `all` and `3y` for 2022), the model is trained once and the result is reused. This keeps the screening efficient.

Canonical de-duplication and the CatBoost screening hyperparameters remain unchanged. Default budget is 200 trees per unique model.

## Run

```powershell
python scripts/run_training_window_game_type.py --config configs/default.yaml
```

Optional example:

```powershell
python scripts/run_training_window_game_type.py --config configs/default.yaml --folds 2023,2024 --windows all,3y,2y,1y --iterations 200
```

## Outputs

```text
outputs/training_window_game_type/
  fold_results.csv
  summary.csv
  best_by_fold.csv
  run_config.json
```

The console reports Brier, competition-style score, AUC, and prediction spread. `best_by_fold.csv` makes it easy to see whether the preferred amount of history changes before and after the 2023 regime shift.

## Decision

- If recent windows consistently beat `all`, move the main CatBoost pipeline to a recency-aware training policy.
- If the preferred window differs by regime, test weighting/decay rather than a hard cutoff.
- If `drop_game_type` remains best regardless of window, remove raw `game_type` from the canonical model.
- If recent-window `raw_game_type` recovers, the problem is stale history rather than the feature itself.
