#!/usr/bin/env bash
set -euo pipefail

# Fresh Linux server bootstrap for the Tablatent experiment branch.
# Usage from repository root:
#   bash scripts/bootstrap_4090_gpu2.sh
#
# Environment policy:
# - Conda environment name defaults to `tablatent`
# - Python 3.11
# - PyTorch 2.7.1 CUDA 12.8 wheel installed with pip inside the conda env
# - CatBoost 1.2.10 from configs/requirements.txt
# - physical GPU 2 only via scripts/activate_gpu2.sh

CONDA_ENV_NAME="${CONDA_ENV_NAME:-tablatent}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable. Install/repair the NVIDIA driver first." >&2
  exit 1
fi

printf '\n[1/6] NVIDIA inventory\n'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

printf '\n[2/6] Initializing conda shell\n'
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

printf '\n[3/6] Creating/reusing conda env: %s (Python %s)\n' "${CONDA_ENV_NAME}" "${PYTHON_VERSION}"
if conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  echo "Conda env '${CONDA_ENV_NAME}' already exists; reusing it."
else
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi
conda activate "${CONDA_ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

printf '\n[4/6] Installing pinned PyTorch CUDA wheel\n'
python -m pip install --upgrade --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

printf '\n[5/6] Installing project dependencies\n'
python -m pip install -r configs/requirements.txt

printf '\n[6/6] Verifying physical GPU 2 isolation and GPU libraries\n'
# shellcheck disable=SC1091
source scripts/activate_gpu2.sh
python scripts/check_gpu2_environment.py --smoke-catboost

cat <<EOF

Bootstrap complete.
For every new shell:
  conda activate ${CONDA_ENV_NAME}
  source scripts/activate_gpu2.sh

Because CUDA_VISIBLE_DEVICES=2 isolates physical GPU 2, scripts should use their
normal logical GPU argument '--devices 0' (not 2) inside this shell.

Prepare data once:
  python scripts/prepare_data.py --config configs/default.yaml
EOF
