# EX1 — Reverse future-to-past expert

EX1 tests one hypothesis without retraining the SAFE982 forward predictor:

> Can the current row reconstruct a pitcher's previous-season trait, and does that reverse signal improve the frozen SAFE982 prediction?

## Isolation

- `bitaboost-stable` is not modified.
- Experiment branch: `ex1-reverse-expert`.
- Runner: `scripts/ex1/run_reverse_expert.py`.
- Generated CatBoost models: `models/ex1/`.
- Generated metrics/predictions: `outputs/experiments/ex1/`.
- The forward control is read only from `outputs/baseline/predictions.npz`.

## Reverse target

For a row in season `s`, EX1 attaches a shrinkage profile computed from the same pitcher's season `s-1`:

- control-success rate;
- reverse rate;
- middle rate;
- ball rate;
- strike rate.

The reverse CatBoost is trained on current-row features from 2020–2023 and these previous-season profiles. It is then applied row-by-row to 2024. Thus 2024 `control_success` is never a reverse-training label.

Two variants are intentionally retained:

1. `structural`: raw player/team IDs removed. This is the main scientific test.
2. `full`: all SAFE rich features retained. This diagnoses how much improvement is merely identity reconstruction.

Each pitcher-season is approximately equally weighted so a high-volume pitcher does not dominate a repeated season-level reverse target.

## Evaluation

EX1 reports:

- previous-season profile reconstruction MSE on 2024 rows;
- reverse expert's standalone Brier;
- frozen SAFE982 Brier;
- fixed low-weight blends with SAFE982;
- R/F-domain diagnostic blends;
- correlation between the reverse correction direction and SAFE982 residual;
- batch-vs-single-row prediction equality audit.

The alpha sweep is diagnostic and uses 2024 labels only for evaluation/selection reporting. A positive result is not promoted automatically; it must survive a follow-up fold/reproduction test.

## Run

The stable baseline prediction artifact must already exist. EX1 never regenerates it.

```bash
conda activate bitaboost
cd ~/Aimers/Bitaboost
python scripts/ex1/run_reverse_expert.py \
  --config experiments/configs/ex1_reverse_expert.yaml
```
