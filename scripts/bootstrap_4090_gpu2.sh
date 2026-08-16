#!/usr/bin/env bash
set -euo pipefail

# Fresh Linux server bootstrap for the Tablatent experiment branch.
# Usage from repository root:
#   bash scripts/bootstrap_4090_gpu2.sh
#
# We intentionally install the same PyTorch family used in the existing project
# instead of silently moving to a newer major/minor build. CUDA 12.8 wheels are
# self-contained apart from the NVIDIA driver; a system CUDA toolkit is not
# required for ordinary PyTorch/CatBoost training.

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable. Install/repair the NVIDIA driver first." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
then
  echo "ERROR: Python >= 3.9 is required." >&2
  exit 1
fi

printf '\n[1/5] NVIDIA inventory\n'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

printf '\n[2/5] Creating virtual environment: %s\n' "${VENV_DIR}"
if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
  echo "ERROR: Python venv support is missing." >&2
  echo "On Ubuntu/Debian install it first, e.g. sudo apt install python3-venv" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

printf '\n[3/5] Installing PyTorch 2.7.1 CUDA 12.8 wheel\n'
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

printf '\n[4/5] Installing project dependencies\n'
python -m pip install -r configs/requirements.txt

printf '\n[5/5] Verifying physical GPU 2 isolation and GPU libraries\n'
# shellcheck disable=SC1091
source scripts/activate_gpu2.sh
python scripts/check_gpu2_environment.py --smoke-catboost

cat <<'EOF'

Bootstrap complete.
For every new shell:
  source .venv/bin/activate
  source scripts/activate_gpu2.sh

Because CUDA_VISIBLE_DEVICES=2 isolates physical GPU 2, scripts should use their
normal logical GPU argument `--devices 0` (not 2) inside this shell.

Prepare data once:
  python scripts/prepare_data.py --config configs/default.yaml
EOF
