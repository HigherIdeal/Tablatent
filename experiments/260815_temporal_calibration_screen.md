# Strict Temporal Calibration Screen

## Goal
Determine whether a meaningful part of Brier loss comes from probability calibration rather than ranking/discrimination.

## Leakage-safe protocol
For evaluation season `Y`:

1. Train a calibration-generator model on seasons `< Y-1`.
2. Predict season `Y-1` and fit the calibrator using only `Y-1` labels.
3. Train a fresh evaluation model on seasons `< Y`.
4. Predict season `Y` and apply the frozen calibrator from step 2.

This mirrors deployment: a calibrator fitted on the most recently completed labeled season is transferred to a newly trained model for the next season.

Default folds are 2022, 2023, and 2024. No row sampling is used.

## Base models
- `reference_canonical`
- `add_success_state`

Both exclude pitcher/batter ID memorization.

## Calibration methods
- raw
- probability mean shift
- shrinkage toward previous-season target prior
- logit intercept
- temperature scaling
- affine logit (`sigmoid(a * logit(p) + b)`)

All fitted calibration parameters minimize Brier or have the closed-form Brier solution where applicable.

## Score reporting
New diagnostics report both:
- `raw_score = 100000 * BrierSkill`, which may be negative
- `clipped_score = max(0, raw_score)`

This prevents the 2023 regime-shift fold from collapsing all poor models to an indistinguishable score of zero.

## Run
```powershell
python scripts/run_temporal_calibration_screen.py --config configs/default.yaml
```

Outputs are written to `outputs/temporal_calibration_screen/`.
