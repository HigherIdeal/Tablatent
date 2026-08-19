# Cycle 3 — SAFE post-2023 regime bridge

Purpose: test whether the large 2023->2024 game-type residual transfer found in Cycle 2 survives when attached directly to the current SAFE family.

Protocol:

1. Train the existing SAFE-family components on seasons <2023 and predict 2023.
2. Ignore the 2023-fitted ensemble weights. Recombine the 2023 components with the frozen current SAFE mixed/simplex weights.
3. Estimate only 2023 residual maps: global, game_type, and centered game_type differential.
4. Transfer those maps unchanged to the frozen SAFE982 2024 prediction.
5. The predeclared candidate is `game_type`, alpha `0.50`, inherited from Cycle 2. Other candidates are exploratory diagnostics only.

This experiment does not use Trackman and does not use any 2024 label to fit the transferred correction. It is a development diagnostic, not a submission recipe.

Run:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_cycle3_sota_regime_bridge.py --gpu 2
```

Report:

```text
outputs/experiments/cycle3_sota_regime_bridge/report.md
```
