# 2026-08-15 Recency-Weighted CatBoost

## Purpose

Hard recent-season cutoffs did not help: using only 1–3 recent seasons lost more signal than it gained from adaptation. The next experiment therefore keeps **all historical rows** but reduces the relative influence of older seasons.

This is a temporal-generalization experiment, not ordinary hyperparameter tuning. A successful decay can preserve sample size while adapting more quickly to the 2023+ distribution/regime shift.

## Experiment

For validation folds `2022, 2023, 2024`, train on every earlier season and assign season-level sample weights:

```text
raw_weight = decay ** (latest_train_year - row_season)
```

The weights are then normalized to mean 1 before CatBoost. This keeps the overall weight scale comparable across decay settings while changing only the relative emphasis on recent seasons.

Default decay values:

```text
1.00  no recency weighting; baseline
0.90  weak decay
0.75  moderate decay
0.50  strong decay
```

For each decay, compare:

- `raw_game_type`: canonical 36-feature model;
- `drop_game_type`: same model with raw `game_type` removed.

The CatBoost screening budget remains 200 trees so the result is comparable to the preceding experiments.

## Run

```powershell
python scripts/run_recency_weighting.py --config configs/default.yaml
```

Optional:

```powershell
python scripts/run_recency_weighting.py --config configs/default.yaml --folds 2023,2024 --decays 1.0,0.95,0.9,0.8,0.7 --iterations 200
```

## Outputs

```text
outputs/recency_weighting/
  fold_results.csv
  summary.csv
  best_by_fold.csv
  weights_by_fold.csv
  run_config.json
```

The console and CSVs report Brier, competition-style score, AUC, prediction spread, weighted target prior, and effective sample size.

## Decision

- If a decay clearly and consistently improves 2023/2024 without materially hurting 2022, use it as the base temporal training policy.
- If different folds prefer different mild decays, refine around the stable range rather than choosing the single best point.
- If strong decay hurts, old seasons still contain useful conditional signal and should not be discarded.
- After choosing the temporal weighting policy, test leakage-safe probability calibration using a prior temporal holdout. Calibration is intentionally a separate experiment so the source of any gain remains identifiable.
