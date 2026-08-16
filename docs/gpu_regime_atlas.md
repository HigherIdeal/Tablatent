# GPU Regime Atlas

This is the Colab/GPU follow-up to `run_phone_regime_atlas.py`.

It does two different checks:

1. model-free target regime analysis over 2019-2024;
2. strict temporal CatBoost OOF residual analysis on 2022/2023/2024.

The residual step is the important addition. A signal is interesting for a third/fourth expert only if its temporal relationship changes **and** the current raw-`game_type` CatBoost still leaves a persistent post-2023 error pattern.

## Colab run

```bash
git pull
pip install -r configs/requirements.txt
python scripts/prepare_data.py
python scripts/run_gpu_regime_atlas.py --config configs/default.yaml --task-type GPU --devices 0
```

The default run fits two controlled temporal baselines at 400 trees:

- `raw_game_type`: canonical + success-state features with raw `game_type`;
- `drop_game_type`: the same feature set with only `game_type` removed.

For each validation year, training uses only strictly earlier seasons.

## Main outputs

`outputs/gpu_regime_atlas/`

- `target_regime_atlas.csv`: marginal/conditional target relationship shifts.
- `oof_fold_metrics.csv`: temporal baseline quality for 2022/2023/2024.
- `residual_regime_atlas_raw_game_type.csv`: what the current raw baseline still misses.
- `residual_regime_atlas_drop_game_type.csv`: controlled comparison without `game_type`.
- `expert_candidate_ranking.csv`: combined evidence for possible additional experts.
- `top_residual_group_profiles.csv`: 2022/2023/2024 group-level residual effects for the strongest residual candidates.
- `run_config.json`: exact run policy.

Use `--save-oof` only when row-level predictions are needed; it writes a much larger CSV.

## Evidence labels

- `STRONG_NEW_EXPERT_CANDIDATE`: target regime change plus persistent residual structure after raw `game_type` modeling.
- `UNEXPLAINED_RECENT_SIGNAL`: persistent recent residual structure even without a strong marginal regime label.
- `RAW_GAME_TYPE_ABSORBS_SHIFT`: the marginal shift is real, but raw `game_type` substantially absorbs it relative to the drop control.
- `TARGET_SHIFT_MOSTLY_MODELED`: target relation changed, but the current baseline largely handles it.
- `LOW_PRIORITY`: no current evidence for a separate expert.

The labels are screening heuristics, not an automatic instruction to create an expert. The next step should test only the strongest few candidates in the existing three-fold 2025 proxy.
