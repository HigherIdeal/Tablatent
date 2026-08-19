# Cycle 2 Research Audit

This cycle follows the four-axis audit and deliberately avoids local SAFE982 tuning.

It answers three surviving questions in one run:

1. **A — strict temporal transfer**: learn residual corrections from one completed OOF season and apply them only to the next season. This tests whether the 2023/2024 bias is actually transferable rather than merely detectable.
2. **C — Trackman mechanics**: repair `pitcher_trackman_id` handling, verify ID overlap, aggregate prior-season mechanical means/dispersion and within-pitch-type dispersion, then test `mechanics_(s-1) -> residual_s` with rolling ridge diagnostics.
3. **D — reliability/cold-start**: use fine experience bins (`0`, `1-9`, `10-49`, `50-199`, `200-499`, `500+`) and select among context/pitcher-history/batter-history/full-history using only the immediately previous OOF fold. This distinguishes true causal routing from a descriptive 2024 cohort effect.

The run first builds rolling OOF predictions for 2021–2024. Every downstream diagnostic uses those frozen predictions.

```bash
CUDA_VISIBLE_DEVICES=2 \
python scripts/run_cycle2_temporal_trackman_reliability.py --gpu 2
```

Outputs:

- `outputs/experiments/cycle2_temporal_trackman_reliability/report.md`
- `outputs/experiments/cycle2_temporal_trackman_reliability/metrics.json`
- `outputs/experiments/cycle2_temporal_trackman_reliability/rolling_predictions.npz`

Do not promote any correction, Trackman feature family, or router unless it survives the causal rolling decision rules printed in the report.
