# EX1 — Backward modeling experiments

EX1 isolates the future-to-past idea from the SAFE982 forward predictor.

## EX1-A: row-level reverse expert

The first implementation asked whether current-row SAFE features could reconstruct the pitcher's previous-season profile and then blended the reconstructed success rate with SAFE982. The direct blend failed: both `structural` and `full` variants selected alpha 0. This result is retained as a negative control rather than promoted.

Runner:

```bash
python scripts/ex1/run_reverse_expert.py \
  --config experiments/configs/ex1_reverse_expert.yaml
```

## EX1-B: pure pitcher-season backward reconstruction

EX1-B removes SAFE982 completely and asks only whether a future pitcher-season state contains enough information to reconstruct the same pitcher's immediately previous-season state.

Each pitcher-season is represented by exactly one row:

```text
Z(p, s) = [success, reverse, middle, ball, strike, log(pitch_count)]
```

The supervised task is:

```text
Z(p, s) -> [success, reverse, middle, ball, strike] of Z(p, s-1)
```

This fixes the main statistical weakness of EX1-A: season-level targets are no longer repeated once per pitch row.

### Variants

- `state_only`: no player identity; tests temporal reversibility of the state itself.
- `state_plus_id`: adds `pitcher_id`; diagnoses how much reconstruction comes from memorizing persistent player identity.

### Rolling evaluation

The backward map is evaluated chronologically:

```text
test 2022 -> 2021 : train current-season pairs 2020-2021
test 2023 -> 2022 : train current-season pairs 2020-2022
test 2024 -> 2023 : train current-season pairs 2020-2023
```

Both current and previous seasons must satisfy the configured minimum pitch count. The default run checks thresholds 50 and 200.

### Baselines and metrics

No Brier/SOTA score is used because this experiment does not predict `control_success` for individual pitches. For each reconstructed state component it reports RMSE, MAE, Pearson and Spearman correlation. Three predictors are compared:

1. learned CatBoost backward map;
2. identity/persistence baseline: assume previous state equals future state;
3. training-set mean profile.

The critical diagnostic is whether the learned model beats the persistence baseline. If even raw persistence beats the mean strongly, the player trait is temporally reversible. If CatBoost additionally beats persistence across folds, a non-trivial backward mapping exists.

### Run

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost
python scripts/ex1/run_pitcher_season_backward.py \
  --config experiments/configs/ex1_pitcher_season_backward.yaml
```

Outputs are isolated under `outputs/experiments/ex1/pitcher_season_backward/`; final-fold model artifacts are stored under `models/ex1/`. Nothing in EX1-B loads, retrains, blends, or modifies SAFE982.
