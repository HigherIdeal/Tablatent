# EX7 — Stable-trait injection into SAFE direct

EX6 showed that every standalone overnight expert receives zero optimal blend weight against SAFE982. EX7 therefore tests the remaining useful hypothesis at the **feature level**, not as another output ensemble.

The experiment reconstructs the exact SAFE982 feature state, adds frozen pitcher-career traits built only from seasons before the row's season, and retrains only the direct MultiRMSE head. All downstream SAFE components and all historical R/F blend/simplex weights are frozen. This isolates whether `career baseline + current deviation` improves the representation itself.

A `retrain_control` with zero added features is included to expose CatBoost GPU retrain noise. Feature variants then test career success, career strike, their combination, reliability/count support, and available current-vs-career deviations. Missing `asof_pitcher_strike_rate` is skipped rather than synthesized.

Run:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/ex7/run_stable_trait_injection.py --gpu 2
```

Primary output: `outputs/experiments/ex7_stable_trait_injection/report.md`.

Safety: no evaluation/test-row interaction is introduced. A season-s row's career profile reads only source seasons `< s`; downstream weights are not re-fit on 2024.
