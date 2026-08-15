# 2026-08-15 Game-Type F Offset Diagnostic

## Question

The observed `game_type=F` control-success rate changes abruptly in 2023 while `R` remains comparatively stable. Test the specific hypothesis that the 2023-2024 `F` measurements are shifted downward by an approximately constant additive amount.

The proposed correction is deliberately narrow:

```text
2019-2022 F: unchanged
2019-2022 R: unchanged
2023-2024 R: unchanged
2023-2024 F: corrected_rate = observed_rate + delta
```

The same `delta` is applied to every 2023-2024 F subgroup. Binary `control_success` labels are **not** modified.

## Why this is a diagnostic first

Adding a constant to binary labels would be invalid. The experiment therefore operates only on aggregated success rates and asks whether one common F-only offset restores the old `F-R` relationship.

A convincing result requires more than matching the overall season mean. The same delta should also reduce mismatch across several subgroup families:

- pitcher team;
- month;
- ball/strike count;
- pitcher/batter handedness;
- inning;
- pitcher experience bucket.

Groups must have enough F and R observations in both eras; the default minimum is 100 rows for each side.

## Reference and objective

The old-era global reference is the macro mean of the yearly `F-R` success-rate effects over 2019-2022.

For subgroup diagnostics, the old reference for each subgroup is pooled from 2019-2022. For 2023 and 2024, the script measures:

```text
raw_residual = (new_F - new_R) - (old_F - old_R)
corrected_residual = raw_residual + delta
```

The primary sweep objective is weighted subgroup RMSE. Weight for each matched subgroup is the minimum of its old/new F/R row counts, so sparse cells cannot dominate.

The script also reports global-effect RMSE, MAE, out-of-bounds corrected subgroup rates, and the fraction of squared mismatch removed by the constant shift.

## Important cross-check

The script additionally fits the best delta for 2023 and 2024 independently.

If the two preferred deltas are close **and** the common delta substantially reduces subgroup RMSE, that supports a stable measurement-offset hypothesis.

If the preferred deltas differ materially, or a large residual RMSE remains after correction, the 2023 change is more complex than a single additive bias.

## Run

```powershell
python scripts/analyze_game_type_f_offset.py --config configs/default.yaml
```

Default sweep:

```text
delta = 0.000 ... 0.300, step 0.005
old era = 2019,2020,2021,2022
new era = 2023,2024
minimum subgroup rows = 100 per old/new F/R cell
```

Optional finer sweep:

```powershell
python scripts/analyze_game_type_f_offset.py --config configs/default.yaml --deltas 0.10:0.24:0.002 --min-group-rows 150
```

## Outputs

```text
outputs/game_type_f_offset/
  season_raw_effects.csv
  season_corrected_effects.csv
  common_delta_sweep.csv
  per_year_delta_sweep.csv
  best_delta_by_year.csv
  subgroup_pairs_raw.csv
  subgroup_pairs_best_common.csv
  family_summary_best_common.csv
  run_config.json
```

## Decision rule

Do not use the correction for model training merely because the season-level F-R means line up.

Treat the constant-offset hypothesis as credible only if:

1. 2023 and 2024 independently prefer similar deltas;
2. the common delta removes a large fraction of subgroup mismatch;
3. residual mismatch after correction is small across multiple subgroup families;
4. corrected subgroup F rates rarely or never leave the valid probability range.

If those conditions hold, the next experiment can test a leakage-safe model representation that separates stable baseball signal from the apparent F measurement shift. If not, keep the raw labels and treat the change as a broader regime shift.
