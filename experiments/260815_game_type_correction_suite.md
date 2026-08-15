# 2026-08-15 Game-Type F Correction Suite

## Goal

The previous diagnostic found that the 2023 and 2024 `game_type=F` shift is unusually close to a constant additive offset. A common `+0.19` old-scale correction explained about 96.7% of the old-vs-new subgroup mismatch, while team remained the largest residual family.

This experiment does **not** rewrite binary labels. Instead, it asks whether a correction learned only from the first new-regime year can improve prediction of the next year.

Default temporal protocol:

```text
old era            : 2019-2022
calibration year    : 2023
validation year     : 2024
```

All F correction parameters are estimated without 2024 labels. The 2024 target is used only for final scoring.

## Experiments

The runner deliberately tests several interpretations of the same hypothesis.

### A. Can 2023 rescue an old-regime model by calibration alone?

Train the canonical raw CatBoost on 2019-2022 and predict both 2023 and 2024.

Test on 2024:

```text
old_only_raw
old_only_measurement_common
old_only_direct_FR
old_only_residual_add
old_only_residual_logit
```

`measurement_common` is estimated from the same multi-subgroup old-vs-new residual logic used by `analyze_game_type_f_offset.py`, but using **2023 only** as the new year. It is therefore a leakage-safe version of the previous common-offset finding.

`direct_FR` uses only the overall F-R effect shift.

`residual_add` and `residual_logit` calibrate the old model's 2023 F predictions directly to the observed 2023 F labels.

### B. Does the remaining team residual generalize?

For three correction families, estimate team-specific 2023 corrections and shrink them toward the corresponding global correction:

```text
measurement team residual
model additive residual
model logit residual
```

Default shrinkage strengths:

```text
alpha = 0, 100, 500, 2000, 10000
```

The reliability is:

```text
n / (n + alpha)
```

So `alpha=0` trusts each team completely, while large alpha approaches the global correction. This directly tests whether the team residual seen in the subgroup analysis is stable enough to carry from 2023 to 2024.

### C. Explicit regime representation inside CatBoost

Keep the full 2019-2023 training set but tell CatBoost that the meaning of `game_type` changed:

```text
game_type_regime = F_old / R_old / F_new / R_new
```

Two variants are tested:

```text
full_add_game_type_regime
full_replace_game_type_regime
```

The first keeps raw `game_type` and adds the interaction. The second replaces raw `game_type` with the interaction.

This is the cleanest training-side alternative to manually changing labels.

### D. Recent-only and fixed blends

For context, the suite retrains a 2023-only model and tests fixed blends with:

```text
full-history raw
recent-only
measurement-corrected old model
explicit-regime model
```

Blend weights are fixed at `0.25, 0.50, 0.75` and are diagnostic only.

## Run

```powershell
python scripts/run_game_type_correction_suite.py --config configs/default.yaml
```

Optional narrower team shrinkage sweep:

```powershell
python scripts/run_game_type_correction_suite.py --config configs/default.yaml --alphas 100,500,1000,2000,5000
```

Default CatBoost screening budget is 200 trees, matching the recent canonical experiments.

## Outputs

```text
outputs/game_type_correction_suite/
  results.csv
  calibration_subgroup_pairs.csv
  team_corrections_by_alpha.csv
  run_config.json
```

`results.csv` reports overall Brier/score/AUC plus separate F and R Brier and calibration gaps.

The most useful columns are:

```text
brier
competition_score
f_brier
f_calibration_gap
r_brier
delta_brier_vs_full
```

## How to read the result

The key questions are ordered deliberately:

1. Does a 2023-only common F correction substantially rescue `old_only_raw` on 2024?
2. Do shrunk team corrections improve over the common correction, and is the improvement stable over a range of alpha values rather than at one isolated setting?
3. Does `game_type_regime` beat the ordinary full-history model without post-processing?
4. Does a corrected old-model signal add value when blended with the full-history model?

A strong result is not merely the single lowest Brier. Prefer a correction that is simple, leakage-safe, improves F without damaging R, and remains good over nearby shrinkage values.

## Important selection warning

The correction parameters themselves are learned only from 2023, but the script prints multiple alpha/blend variants on 2024. Therefore the **best 2024 alpha or blend weight is not automatically a valid final 2025 choice**.

If a family clearly wins, use the 2024 results to choose the family and then define a conservative/fixed selection rule before building the final 2025 package.
