# 2026-08-15 Controlled Unseen-Pitcher Exposure Diagnostic

## Why this follow-up exists

The first unseen-pitcher stress test produced two observations:

- naturally unseen 2024 pitchers were not harder than seen pitchers;
- removing every 2024 pitcher from historical training degraded Brier, but also removed roughly two thirds of the training rows.

Therefore the original strict purge confounds **pitcher exposure** with **training-data volume**.

This follow-up isolates those effects more carefully. It is a diagnostic only; it is not a new 1500-point model direction.

## Protocol

Use the canonical 36-feature CatBoost. `pitcher_id` itself remains excluded.

1. Train one ordinary full-history model on all pre-2024 rows.
2. Take only 2024 pitchers that have earlier historical rows.
3. Partition those pitchers into five groups balanced by historical row count.
4. For each group, evaluate exactly the same 2024 rows under three training conditions:

```text
full_history
    all pre-2024 rows

purged_same_pitchers
    remove all historical rows belonging to the evaluated pitcher group

matched_size_keeps_pitcher_history
    keep the evaluated pitchers' historical rows,
    but randomly remove other rows until the training set has exactly the
    same number of rows per season as purged_same_pitchers
```

The matched-size control is the important addition.

## Interpretation

For each group the script reports:

```text
purge_minus_full
control_minus_full
purge_minus_control
```

Interpret them as:

```text
purge_minus_full
= historical-pitcher removal + train-size loss

control_minus_full
= train-size loss while keeping the evaluated pitchers' history

purge_minus_control
= approximate extra penalty from removing that pitcher group's historical
  feature distribution beyond the matched train-size loss
```

Because `pitcher_id` is not a canonical model feature, `purge_minus_control` must not be interpreted as direct ID memorization. It tests whether having historical rows from the same pitchers / same feature region helps the learned mapping.

If `purge_minus_control` is near zero, the previous strict-unseen degradation was mostly a data-volume artifact.

If it is consistently positive across groups, historical exposure to the same pitchers' feature distributions matters even without pitcher ID.

## Cohort composition check

The script also compares natural unseen vs seen 2024 pitchers on:

- target rate;
- `asof_pitcher_n` mean/median/quartiles;
- `game_type=F` share;
- mean `asof_pitcher_success_rate`.

It additionally reports model metrics inside fixed `asof_pitcher_n` buckets:

```text
0-100
101-500
501-1000
1001-2000
2001-5000
5000+
```

This checks whether the surprisingly good natural-unseen result is simply cohort composition rather than a genuine unseen-pitcher advantage.

## Run

```powershell
python scripts/run_unseen_pitcher_exposure_diagnostic.py --config configs/default.yaml
```

Default cost is one full-history model plus two models per group: 11 CatBoost fits for five groups.

For a quicker screen:

```powershell
python scripts/run_unseen_pitcher_exposure_diagnostic.py --config configs/default.yaml --groups 3 --iterations 150
```

## Outputs

```text
outputs/unseen_pitcher_exposure_diagnostic/
  cohort_composition.csv
  experience_bucket_metrics.csv
  pitcher_groups.csv
  fold_results.csv
  aggregate_seen_pitcher_results.csv
  run_config.json
```

## Decision

After this diagnostic, stop spending model-development effort on unseen-pitcher handling unless there is a clear, repeated positive `purge_minus_control` penalty. The main 1500-point search should otherwise return to feature signal, temporal structure, model diversity, and later Trackman-as-privileged-information experiments.
