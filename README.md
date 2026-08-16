# Tablatent — LG Aimers 9 Baseball Hackathon

Pitch-level probability modeling for `control_success` using only information available immediately before each pitch.

## Current reference system

The current deployable backbone is CatBoost-based:

```text
full-history raw-game_type expert
          +
recent raw-game_type expert
          +
R-fast specialist
```

Reference mixture:

- `alpha_recent = 0.20`
- `beta_r = 0.10`
- final recent regime for 2025 training: 2023+2024
- raw `game_type` is retained
- pitcher/batter IDs are excluded from the canonical predictor after temporal ablation

The hidden evaluation rows must be independent at inference. A prediction may use only the current row plus parameters learned from training data. No hidden-test aggregation, sequential state update, test-prior estimation, batch adaptation, or transductive test-set feature is allowed.

Latest experiment conclusions are in `EXPERIMENT_STATUS_2026-08-16.md`.

## Fresh 4x RTX 4090 server

Project policy: **use physical GPU 2 only** and use a **Conda environment named `tablatent`**.

Clone the active branch:

```bash
git clone --branch agent/stable-player-dynamics-gru --single-branch https://github.com/HigherIdeal/Tablatent.git
cd Tablatent
```

Bootstrap once:

```bash
bash scripts/bootstrap_4090_gpu2.sh
```

The bootstrap creates/reuses `conda` env `tablatent` with Python 3.11, installs the pinned GPU stack and project requirements, isolates physical GPU 2, and runs PyTorch/CatBoost GPU smoke tests.

For every new shell:

```bash
conda activate tablatent
source scripts/activate_gpu2.sh
```

`activate_gpu2.sh` sets `CUDA_VISIBLE_DEVICES=2`, so physical GPU 2 is the only visible device and is process-local `cuda:0` / CatBoost `devices=0`.

Verify:

```bash
python scripts/check_gpu2_environment.py --smoke-catboost
```

Full setup and performance notes: `docs/server_4090_gpu2.md`.

## Data

Fixed source URL:

`https://drive.google.com/file/d/1RqoOknOl39FnNMgHZ-DQrVim8Of-odKM/view?usp=drive_link`

Prepare once:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

The loader validates 2019–2024 seasons and stores the processed frame at:

```text
data/processed/train.pkl
```

## Canonical CatBoost features

`src/canonical_features.py` removes exact deterministic duplicates rather than feeding multiple encodings of the same state into CatBoost. Examples:

- score state keeps `run_total_before + score_diff_home`
- runner state keeps `base_state`
- `asof_pitcher_pitchmix_n` is removed because it equals `asof_pitcher_n` in the audited train data
- home/away win expectancy is normalized to the pitcher's team perspective
- pitcher/batter IDs are excluded after temporal ablation

Raw `game_type` remains because removing it hurt the 2025 leaderboard proxy/submission evidence.

## Temporal validation

Random splitting is not used for model selection. The main proxy suite combines chronological 2024 folds:

- season-forward 2024
- mid-2024 expanding window
- late-2024 expanding window

2024 has now been used repeatedly for model development and is no longer treated as a pristine one-shot holdout. Promotion decisions should therefore rely on consistency across temporal folds, not a single best 2024 score.

## Recent regime-adaptation results

Three controlled follow-ups were completed:

1. **Row-wise dynamic full/recent gate** — not promoted. The context gate selected zero adaptation strength; the outputs-only gate produced only noise-scale gain and improved 1/3 folds.
2. **Season-aware latent HMM robustness** — diagnostic only. After splitting off-season boundaries correctly, 2023 and 2024 consistently share the broad recent state; exact HMM partitions are not stable enough for prediction.
3. **Smooth recency weighting** — not promoted. Every tested half-life was worse than the unweighted full-history expert in weighted proxy Brier.

This supports keeping old seasons in the full expert and using the recent expert as a complementary correction rather than replacing the full-history model.

## Main experiment scripts

Current CatBoost/regime tooling includes:

```text
scripts/run_2025_proxy_validation.py
scripts/run_gated_r_specialist_suite.py
scripts/build_gated_r_specialist_submission.py
scripts/run_rowwise_dynamic_gate.py
scripts/run_latent_regime_robustness.py
scripts/run_recency_weighted_full_expert.py
```

Example under the GPU2-isolated Conda shell:

```bash
conda activate tablatent
source scripts/activate_gpu2.sh
python scripts/run_gated_r_specialist_suite.py \
  --config configs/default.yaml \
  --task-type GPU \
  --devices 0
```

Because physical GPU 2 has been isolated by `CUDA_VISIBLE_DEVICES=2`, `--devices 0` means physical GPU 2 here.

## Performance policy

Do not alter model semantics merely to occupy more GPU memory. Keep score-affecting settings such as tree depth, border count, training window, and feature policy controlled by the experiment rather than by server size.

The main speed strategy is:

- physical GPU 2 isolation;
- local processed-data storage;
- prediction/artifact reuse where supported;
- GPU CatBoost for every fit;
- low logging overhead;
- one large training job on GPU 2 at a time;
- larger PyTorch batches only for future neural experiments after a memory probe.

## Legacy representation-learning experiments

The repository still contains Stage1 VAE, latent CatBoost, kNN, GRU, and Transformer experiments for reproducibility. They are not the current reference path. Historical artifacts can still be reproduced with the existing scripts/configuration, but new model development should start from the CatBoost full+recent+R-fast reference system above.
