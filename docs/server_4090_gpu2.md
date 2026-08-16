# Fresh 4x RTX 4090 server setup — use physical GPU 2 only

This profile is for the current Tablatent experiment branch on a Linux server with four RTX 4090 GPUs.

## 1. Clone the current working branch

```bash
git clone --branch agent/stable-player-dynamics-gru --single-branch https://github.com/HigherIdeal/Tablatent.git
cd Tablatent
```

If the repository already exists:

```bash
git fetch origin
git switch agent/stable-player-dynamics-gru
git pull --ff-only
```

## 2. Bootstrap the Conda environment

The standard environment name is `tablatent`.

```bash
bash scripts/bootstrap_4090_gpu2.sh
```

The bootstrap requires `conda` to already be installed. It creates/reuses:

```text
conda env: tablatent
Python:    3.11
```

Then it installs the pinned PyTorch GPU wheel and the repository requirements, isolates physical GPU 2, and runs PyTorch and CatBoost GPU smoke tests.

To use another Conda environment name deliberately:

```bash
CONDA_ENV_NAME=my_env bash scripts/bootstrap_4090_gpu2.sh
```

## 3. Activate for every shell

```bash
conda activate tablatent
source scripts/activate_gpu2.sh
```

The second command sets:

```text
CUDA_VISIBLE_DEVICES=2
```

Therefore only **physical GPU 2** is visible to the process. Inside PyTorch/CatBoost it is renumbered to logical GPU `0`.

Use:

```text
PyTorch: cuda:0
CatBoost: --devices 0
```

Do **not** pass `--devices 2` after sourcing `activate_gpu2.sh`; logical device 2 no longer exists inside the isolated process.

This isolation protects legacy scripts whose defaults are `cuda:0` / `--devices 0`: they hit physical GPU 2 rather than physical GPU 0.

## 4. Verify the server at any time

```bash
python scripts/check_gpu2_environment.py --smoke-catboost
```

Expected properties:

- active Conda environment is `tablatent`
- `CUDA_VISIBLE_DEVICES='2'`
- `torch.cuda.device_count=1`
- logical `cuda:0` reports an RTX 4090
- PyTorch CUDA smoke passes
- CatBoost GPU smoke passes

To watch only the physical GPU being used:

```bash
watch -n 1 nvidia-smi -i 2
```

## 5. Prepare data once

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

The repository already contains the fixed Google Drive source URL and validates the 2019–2024 seasons before writing `data/processed/train.pkl`.

## Performance policy

The primary experiments are CatBoost-based. Do not change model semantics merely to consume more VRAM; changing tree depth, border count, feature policy, or the temporal split changes the experiment.

For speed without changing the model:

- keep physical GPU 2 isolated;
- keep processed data on local SSD/NVMe;
- reuse cached predictions/artifacts where supported;
- use GPU CatBoost for every fit;
- avoid verbose per-tree logging unless diagnosing a run;
- run one large CatBoost training job on GPU 2 at a time.

For future PyTorch experiments, increase batch size only after a short memory probe on the 4090. Historical configs should remain unchanged when reproducing prior results.

## Current reference architecture

The current deployable reference remains:

```text
full-history raw-game_type CatBoost
          +
recent raw-game_type CatBoost
          +
R-fast specialist
```

with fixed `alpha_recent=0.20` and `beta_r=0.10`. The latest dynamic-gate, HMM, and recency-weighting conclusions are recorded in `EXPERIMENT_STATUS_2026-08-16.md`.
