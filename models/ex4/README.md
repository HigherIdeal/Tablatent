# EX4 model artifacts

Generated locally by `scripts/ex4/run_bidirectional_traits.py`.

For the final chronological fold, EX4 stores paired CatBoost models for each configured reliability threshold and variant:

```text
forward_<variant>_min<threshold>.cbm
backward_<variant>_min<threshold>.cbm
```

These are research artifacts for temporal-trait discovery only. They are not competition submission models and are excluded from Git tracking by the repository model ignore rules.
