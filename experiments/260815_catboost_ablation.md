# 2026-08-15 CatBoost Feature Ablation

## Objective

Use the previously submitted ~790-point H2/J0 CatBoost model as the reference and determine which feature groups genuinely improve temporal generalization.

The reference is reconstructed from the uploaded submission package:

- 55 input features
- `pitcher_id` excluded
- `batter_id`, `pitcher_team_id`, `batter_team_id` retained as categorical features
- Empirical-Bayes success features at alpha 100/500
- reliability and log-count features
- CatBoost: 520 trees, depth 8, learning rate 0.03, L2 10, random strength 0.5, Bayesian bootstrap, bagging temperature 0.5, `has_time=True`

## Validation protocol

Ablation comparisons use fixed model hyperparameters and a fixed tree count so that feature removal is the only intended experimental variable.

Default temporal folds:

- train 2019-2022 -> validate 2023
- train 2019-2023 -> validate 2024

The Empirical-Bayes prior is computed from each fold's training portion only. 2024 is now used as a labeled temporal holdout because the 2025 leaderboard has already been probed and robustness across years is more important than preserving a single untouched year.

Primary metric: Brier score. Secondary diagnostics: Brier skill, AUC, prediction mean/std.

## Ablations

Reference/addition tests:

- `reference_790`
- `add_pitcher_id`

ID tests:

- drop `batter_id`
- drop each team ID independently
- drop both team IDs
- drop all IDs retained by the reference

Feature-group tests:

- season
- game context
- leverage / win expectancy
- handedness
- long-term pitcher profile
- recent pitcher form
- batter profile
- pitch mix
- Empirical-Bayes features
- reliability/log-count features
- all engineered features

Interpretation uses `delta_brier_vs_reference`:

- positive: removing the group hurts -> evidence to keep it
- negative: removing the group helps -> evidence the group may be harmful
- near zero or inconsistent across years -> treat as redundant/unstable and confirm before removal

Feature importance alone is not used as the keep/drop rule because correlated features can split importance without being causally useful.

## Run

```powershell
python scripts/run_catboost_ablation.py --config configs/default.yaml
```

Full three-fold confirmation if needed:

```powershell
python scripts/run_catboost_ablation.py --config configs/default.yaml --folds 2022,2023,2024
```

Outputs:

- `outputs/catboost_ablation/fold_results.csv`
- `outputs/catboost_ablation/summary.csv`
- `outputs/catboost_ablation/feature_sets.json`
- `outputs/catboost_ablation/run_config.json`

Do not choose the final feature set from a single season. Prefer groups whose removal consistently improves or degrades Brier across temporal folds.
