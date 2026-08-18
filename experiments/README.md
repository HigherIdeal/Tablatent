# Experiments

Research stays out of the baseline entry points.

- Put configuration overrides under `experiments/configs/` first.
- Add reusable implementation to `src/bitaboost/`, not a new top-level script per idea.
- Add an experiment runner only when the workflow cannot be expressed as a config of the baseline runner.
- Store experiment outputs under `outputs/experiments/<name>/`.
- Do not edit the frozen SAFE baseline config to record an exploratory idea.

Submission/ZIP building is intentionally absent from this tree. It will be a separate boundary after a model is frozen.
