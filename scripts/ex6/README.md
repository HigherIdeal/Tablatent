# EX6 — SAFE complementarity

Purpose: decide whether the two strongest overnight candidates are useful **because their errors differ from SAFE**, even though they are weaker standalone models.

Inputs are existing local artifacts:

- `outputs/baseline/predictions.npz` — SAFE 2024 vector.
- `outputs/night_20260819/gpu2/final_summary.json` — selects the overnight GPU2 structural winner.
- `outputs/night_20260819/gpu3/final_summary.json` and `base_predictions/*.npz` — selects/reuses the overnight GPU3 OOF base artifact.

The GPU2 winner is retrained once for the 2024 fold so its row-level prediction vector can be recovered. GPU3 is read from the saved OOF artifact. No full overnight search is repeated.

Run from repository root:

```bash
conda activate bitaboost
git switch ex6-safe-complementarity
CUDA_VISIBLE_DEVICES=2 python scripts/ex6/run_safe_complementarity.py --gpu 2
```

Expected runtime is roughly minutes, not hours. Most work is frozen-history preparation plus one GPU2 CatBoost fit.

Outputs:

```text
outputs/experiments/ex6_safe_complementarity/
  metrics.json
  report.md
  vectors_2024.npz
```

The report measures prediction/residual/error correlation with SAFE, rows where the candidate has smaller error, SAFE-worst-decile recovery, R/F and experience buckets, pair blend curves, and one SAFE+GPU2+GPU3 blend diagnostic.

Important: blend weights are optimized on 2024 **only to measure complementarity**. They are not automatically promoted as final 2025 deployment weights. The next promotion step requires deciding whether the gain is broad enough to justify either a small fixed SAFE blend or transplanting the surviving `career + deviation` features into the SAFE backbone.

The GPU3 report also separates:

- the overnight calibration exactly as recorded;
- a true intercept-only calibration with slope fixed at 1;
- a corrected affine calibration.

This explicitly audits the overnight naming bug where `kind=intercept` still allowed local slope refinement.
