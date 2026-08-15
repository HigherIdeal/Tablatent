# 2026-08-15 F Specialist Hybrid

## Goal

The previous target-correction retraining experiment found that normalizing 2023 `game_type=F` targets can improve the **2024 F-only Brier**, while the same corrected full model can slightly damage `R` and therefore fail to improve overall Brier.

This experiment protects the strong ordinary full-history model on `R` and asks whether the corrected signal should be used **only for F rows**.

Default temporal protocol:

```text
train / calibration information : 2019-2023
validation                       : 2024
```

The F correction is estimated only from 2019-2023. The 2024 target is used only for final scoring.

## Core idea

Always keep the ordinary full-history prediction for `R`:

```text
R -> full_raw prediction
```

For `F`, compare several alternatives:

```text
F -> full_raw
F -> F-only raw specialist
F -> corrected full-model F prediction
F -> corrected F-only specialist
```

The final hybrid is therefore:

```text
if game_type == R:
    p = p_full_raw
else:
    p = (1-w) * p_full_raw + w * p_F_candidate
```

where `w` is a fixed F-only blend weight.

## Models

### 1. Full-history raw baseline

Train the canonical CatBoost on all 2019-2023 rows with raw binary targets.

```text
full_raw
```

This is the reference model and its `R` predictions are never changed by any hybrid.

### 2. F-only raw specialist

Train a separate CatBoost using only `game_type=F` rows from 2019-2023.

`game_type` is removed from the specialist feature set because it is constant inside the F-only data.

This tests whether F is sufficiently different that a separate model helps even without target correction.

### 3. Corrected full-model F only

Use the same soft-target normalization as `run_f_target_correction_training.py`:

- 2019-2022 targets unchanged;
- 2023 R unchanged;
- 2023 F target distribution shifted toward the old measurement scale;
- train on all rows;
- invert F predictions back to the observed/new scale.

However, **only the model's F predictions are allowed into the hybrid**. Its R predictions are discarded.

This directly tests the hypothesis suggested by the previous result: the correction may help F while collateral movement in R causes the overall loss.

### 4. Corrected F-only specialists

Two F-only specialist families are tested:

```text
soft target + inverse affine transform
binary target reweighting + inverse odds transform
```

Because these specialists never train on R rows, target normalization cannot directly distort the R decision surface.

## Correction scales

The correction magnitude is estimated from the 2023 matched-subgroup F-R shift using only 2019-2023 data.

Default strengths:

```text
0.50, 0.75, 1.00, 1.25
```

`1.00` means the full 2023-derived correction.

## F-only blend weights

Default:

```text
0.25, 0.50, 0.75, 1.00
```

`1.00` fully replaces the raw F prediction. Lower values hedge toward the ordinary full-history model.

The R prediction is exactly the raw baseline for every hybrid, so any Brier difference comes only from F rows.

## Run

```powershell
python scripts/run_f_specialist_hybrid.py --config configs/default.yaml
```

Optional narrower sweep:

```powershell
python scripts/run_f_specialist_hybrid.py --config configs/default.yaml --scales 0.75,1.0,1.25 --blend-weights 0.25,0.5,0.75,1.0
```

Default CatBoost budget is 200 trees to match the recent screening experiments.

## Outputs

```text
outputs/f_specialist_hybrid/
  results.csv
  family_summary.csv
  correction_parameters.csv
  calibration_subgroup_pairs.csv
  run_config.json
```

The key result columns are:

```text
brier
competition_score
auc
f_brier
f_calibration_gap
r_brier
delta_brier_vs_raw
```

Because R predictions are protected, `r_brier` should remain exactly equal to the `full_raw` baseline for all hybrid rows up to numerical identity.

## What would count as a useful result?

The strongest evidence would be:

1. one or more F-specialist/corrected families improve overall Brier versus `full_raw`;
2. the gain is entirely attributable to improved `f_brier`, with unchanged `r_brier`;
3. nearby correction scales and blend weights also work rather than one isolated parameter combination;
4. a corrected F-only specialist beats both the F-only raw specialist and corrected-full-F hybrid.

If the corrected-full-F hybrid improves but the F-only specialist does not, then the correction is useful but F still benefits from learning shared structure with R.

If the F-only specialist wins, a mixture-of-experts design (`R` generalist + `F` specialist) becomes a serious candidate for the final 2025 model.

## Selection warning

Multiple correction scales and blend weights are printed on the same 2024 validation fold. The single best row is therefore diagnostic, not an automatically valid 2025 hyperparameter choice. Prefer broad/stable gains and use a conservative fixed rule before final deployment.
