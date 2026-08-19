# EX9 — Density-ratio reweighting

Purpose: test whether the persistent 2023 regime change is partly covariate shift rather than only target-mapping shift.

The domain classifier distinguishes 2023 rows from 2019–2022 using input features only. It excludes season and entity IDs. For old rows, its posterior is converted to an approximate density ratio `p_2023(x) / p_old(x)`, clipped and normalized. The SAFE direct MultiRMSE head is retrained with interpolated density weights, while all downstream SAFE components and blend weights remain frozen.

Run:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/ex9/run_density_ratio_reweight.py --gpu 2
```

Interpretation:
- `alpha=0` is the exact retrain control.
- If `alpha>0` improves the frozen final SAFE prediction, covariate-similarity weighting is useful.
- If alpha returns to 0, the 2023 problem is unlikely to be repaired by feature-space density adaptation and this direction should be terminated.

Safety: 2024 labels and 2024 input distribution are evaluation-only and never used to form training weights.
