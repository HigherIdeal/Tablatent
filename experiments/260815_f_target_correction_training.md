# 2026-08-15 F Target-Correction Retraining

## Goal

The `game_type=F` success-rate relationship shifts abruptly in 2023 and the previous diagnostic found that a common additive correction of about `+0.19` makes 2023 and 2024 look much more like the 2019-2022 measurement scale.

The previous correction suite mostly corrected predictions after training. This experiment answers the different question:

> What happens if the 2023 F training targets are normalized toward the old measurement scale **before CatBoost is trained**?

Default temporal protocol:

```text
old era          : 2019-2022
calibration year : 2023
validation year  : 2024
```

The 2024 target is used only for scoring. The F correction is estimated only from 2019-2023.

## Why a row target cannot literally be `y + 0.19`

The raw target is binary (`0/1`), while `+0.19` was discovered at the aggregated F success-rate level. Literal row-wise addition would produce invalid targets such as `1.19`.

Therefore the experiment uses two mathematically controlled ways to make the **2023 F training distribution** behave as if its success rate had been shifted upward by the learned correction.

## Correction size

The primary correction is estimated from the same matched-subgroup logic as `analyze_game_type_f_offset.py`, using only 2023 as the new year.

For each matched subgroup:

```text
residual = (2023 F-R) - (2019-2022 F-R)
```

The weighted-RMSE optimum is the negative weighted mean residual. This gives the old-scale correction `delta` without using 2024 labels.

The script also reports the simpler overall F-R correction for reference.

## Training methods

### 1. Raw baselines

```text
raw_logloss
raw_crossentropy
```

`raw_crossentropy` is included because the soft-target models use CatBoost `CrossEntropy`; it isolates loss-function effects from the correction itself.

### 2. Soft-target normalization

For 2023 F rows only, positives remain `1.0` and negatives are lifted to a soft target `c`:

```text
y_soft = c + (1-c) * y
```

`c` is chosen so that the mean 2023 F target becomes exactly:

```text
raw_2023_F_rate + scale * delta
```

CatBoost is trained with `loss_function=CrossEntropy`.

At validation time two outputs are reported:

```text
soft_*_latent  : leave prediction on the normalized/old scale
soft_*_inverse : map F predictions back to the observed/new scale
```

The inverse is:

```text
p_observed = (p_normalized - c) / (1-c)
```

with clipping to `[0,1]`.

### 3. Binary-label reweighting

Keep the original `0/1` labels and use `Logloss`, but rebalance positive and negative 2023 F rows so that their weighted positive rate equals the corrected target rate.

Weights are chosen to preserve total expected weight in the 2023 F group:

```text
w_pos = corrected_rate / raw_rate
w_neg = (1-corrected_rate) / (1-raw_rate)
```

Again two outputs are reported:

```text
weight_*_latent  : prediction on the reweighted scale
weight_*_inverse : undo the induced odds shift for 2024 F
```

The inverse-odds correction uses:

```text
odds_ratio = w_pos / w_neg
logit(p_observed) = logit(p_weighted) - log(odds_ratio)
```

## Correction-strength sweep

Default scales:

```text
0.50, 0.75, 1.00, 1.25
```

`scale=1.00` means use the 2023-derived correction as estimated. Nearby scales are included to see whether the method is robust or only works at one finely tuned value.

Do **not** automatically choose the single best 2024 scale for the final 2025 model. The important result is whether a correction family improves consistently around `scale=1`.

## Run

```powershell
python scripts/run_f_target_correction_training.py --config configs/default.yaml
```

Optional:

```powershell
python scripts/run_f_target_correction_training.py --config configs/default.yaml --scales 0.5,0.75,1.0,1.25 --iterations 200
```

## Outputs

```text
outputs/f_target_correction_training/
  results.csv
  correction_parameters.csv
  calibration_subgroup_pairs.csv
  run_config.json
```

`results.csv` includes:

```text
brier
competition_score
auc
prediction_std
f_brier
f_calibration_gap
r_brier
delta_brier_vs_raw
```

## Decision rule

The key comparison is against `raw_logloss` trained on the same 2019-2023 rows.

A convincing result would show:

1. corrected-target retraining beats raw full-history training on 2024 Brier;
2. the gain comes mainly from F without materially damaging R;
3. inverse mapping back to the observed scale behaves sensibly;
4. performance is reasonably stable around correction scale `1.0` rather than peaking at a single arbitrary scale.

If this works, the analogous final-2025 setup would normalize both 2023 and 2024 F training targets using a correction estimated only from already-known seasons, train on 2019-2024, and map 2025 F predictions back to the expected observed scale.
