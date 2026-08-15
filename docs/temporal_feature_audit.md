# Refined 2025 proxy and temporal feature audit

This stage has two goals.

1. Re-run the existing three-fold temporal proxy with a wider CatBoost tree range and finer blend alpha.
2. Rank features whose target relationship changes between 2019-2022 and 2023-2024, so future stable-expert ablations are evidence-driven rather than guessed.

## Refined proxy

Run:

```powershell
python scripts/run_2025_proxy_refined.py --config configs/default.yaml
```

Defaults:

- tree prefixes: 250, 300, 400, 500, 600, 800
- alpha step: 0.025
- same three temporal folds and weights as `run_2025_proxy_validation.py`
- output: `outputs/proxy_2025_validation_refined/`

The wrapper reuses the existing proxy implementation. Each expert is still fit only once at the maximum tree count per fold, and lower tree counts are evaluated as prefixes.

## Temporal feature stability audit

Run:

```powershell
python scripts/run_temporal_feature_stability_audit.py --config configs/default.yaml
```

The audit compares conditional target effects after subtracting each season's target prior:

- old era: 2019-2022
- recent era: 2023-2024

Categorical features use their observed categories. Numeric features use target-independent global quantile bins. For every feature the audit reports:

- old-vs-recent conditional-effect RMSE
- within-old and within-recent instability
- sign-flip rate for non-trivial effects
- old/recent effect correlation
- a changepoint ratio and ranking score

The output candidate list is intentionally **not** an automatic drop list. It is the shortlist for the next stable-expert ablation screen.

Outputs:

- `feature_stability.csv`
- `per_season_group_effects.csv`
- `regime_sensitive_candidates.json`
- `run_config.json`

A useful candidate is one with a large old-to-recent shift but relatively small 2023-vs-2024 instability. `game_type` should act as a positive-control example of that pattern.
