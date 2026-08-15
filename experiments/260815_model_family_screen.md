# Model Family Screen

## Question

Does the current ceiling come from the CatBoost model family rather than only from feature design?

## Models

- CatBoost
- LightGBM
- XGBoost

Two inference-safe feature sets are compared:

1. `success_state`
2. `success_plus_hand_matchup`

No pitcher/batter IDs, Trackman-at-inference features, target encoding, or row sampling are introduced.

## Temporal model selection

For outer evaluation year `Y`:

1. train a candidate model on seasons `< Y-1`;
2. use season `Y-1` only as inner validation;
3. choose the boosting iteration by **Brier score** with early stopping;
4. discard that model;
5. refit the same family and hyperparameters on all seasons `< Y` using the selected number of estimators;
6. evaluate once on season `Y`.

Thus the outer validation target never selects the model complexity. This mirrors the intended 2025 workflow: use 2024 as the latest labeled model-selection season, then refit through 2024 for 2025 inference.

## Categorical handling

- CatBoost uses its native categorical handling.
- LightGBM and XGBoost use pandas categorical columns.
- Their category vocabularies are learned from the fit rows only; unseen validation categories become missing.
- `ctx_hand_matchup = pitcher_hand × batter_hand` remains categorical.

## Default screening parameters

- max estimators: 1200
- learning rate: 0.03
- early stopping patience: 100
- CatBoost: GPU by default
- LightGBM: CPU by default for installation portability
- XGBoost: CUDA by default

The purpose is a model-family screen, not exhaustive hyperparameter tuning. If another family wins consistently, tune only that family afterward.

## Primary decision rule

Prioritize:

1. Brier score on each temporal fold;
2. mean and worst `delta_brier_vs_catboost_success`;
3. raw unclipped Brier-skill score;
4. AUC as a discrimination diagnostic.

A family that only wins one regime but loses badly on another is not promoted directly to the final baseline.
