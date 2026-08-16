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

## 2. Bootstrap the Python environment

```bash
bash scripts/bootstrap_4090_gpu2.sh
```

The bootstrap creates `.venv`, installs PyTorch 2.7.1 from the official CUDA 12.8 wheel index, installs the project requirements, isolates physical GPU 2, and runs both PyTorch and CatBoost GPU smoke tests.

A system CUDA toolkit is not required for these Python wheels; a working NVIDIA driver is required.

## 3. Activate for every shell

```bash
source .venv/bin/activate
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

This isolation is deliberate because several legacy scripts default to `cuda:0` / `--devices 0`. They will now safely hit physical GPU 2 rather than physical GPU 0.

## 4. Verify the server at any time

```bash
python scripts/check_gpu2_environment.py --smoke-catboost
```

Expected properties:

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

The primary experiments are currently CatBoost-based. Do not change model semantics merely to consume more VRAM. CatBoost GPU training already keeps categorical features in GPU RAM by default and its default GPU-RAM fraction is 0.95. More VRAM therefore gives headroom automatically; changing `border_count`, tree depth, or the temporal split only for speed would change the experiment itself.

For speed without changing the model:

- keep physical GPU 2 isolated so no other project process can spill to another GPU;
- reuse cached predictions/artifacts when a script supports them;
- use the GPU implementation for all CatBoost fits;
- avoid verbose per-tree logging unless diagnosing a run;
- keep processed data on local SSD/NVMe rather than a network-mounted directory;
- run only one large CatBoost training job on GPU 2 at a time.

For future PyTorch experiments, tune batch size upward on this server only after a short memory probe. Do not globally change the historical experiment config solely because the server has more VRAM; that would make old comparisons non-equivalent.

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
