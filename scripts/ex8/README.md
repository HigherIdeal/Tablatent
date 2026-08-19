# EX8 cohort routing diagnostic

EX7 showed that direct injection of career success/strike features did not improve SAFE982. EX8 therefore asks a different question: does the existing SAFE component family behave differently across current-row pitcher/batter experience cohorts?

The script uses the already saved 2024 SAFE component vectors and current-row `asof_pitcher_n` / `asof_batter_n`. It reports component Brier by cohort plus 2024-only oracle simplex upper bounds over `mixed`, `offset`, `joint`, and `structured`.

Oracle weights are diagnostic only. They must not be used for 2025. If the oracle gap is material, the next experiment is to reproduce cohort weights from earlier rolling OOF folds (2022/2023) and evaluate them untouched on 2024.

Run:

```bash
python scripts/ex8/run_cohort_component_diagnostic.py
cat outputs/experiments/ex8_cohort_component_diagnostic/report.md
```
