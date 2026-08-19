# Four-axis research audit

One command explores four independent hypotheses without locally tuning SAFE982:

- **A — Conditional shift:** pooled cross-fit with/without explicit season, rolling residual drift, and same-pitcher/context matched season transitions.
- **B — Latent pitcher state:** adjacent pitcher-season state persistence plus rolling tests of whether prior state predicts next-season success and row-model residuals.
- **C — Trackman mechanics:** auto-detect `trackman_history.csv`, aggregate prior-season mechanical dispersion, and test whether it predicts later residuals. Trackman features are shifted forward one season before evaluation.
- **D — Cold start:** rolling 2022/2023/2024 context/pitcher-history/batter-history/full-history models, then compare information value in cold/mid/rich pitcher × batter cohorts.

Run:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_research_audit_four_axes.py --gpu 2
```

Primary output:

```text
outputs/experiments/research_audit_four_axes/report.md
```

Supporting outputs are `metrics.json` and `rolling_predictions.npz`. The audit is diagnostic only; no weight or model selected from the audit is automatically promotable to a submission.
