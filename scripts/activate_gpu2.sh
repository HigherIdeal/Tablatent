#!/usr/bin/env bash
# Source this file on the 4x RTX 4090 server before running experiments:
#   source scripts/activate_gpu2.sh
#
# Physical GPU 2 becomes the only CUDA-visible device. Inside Python/CatBoost it
# is therefore addressed as logical device 0. This prevents an old script whose
# default is --devices 0 / cuda:0 from accidentally using physical GPU 0.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

printf '[GPU isolation] physical GPU 2 only; process-local CUDA device = 0\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -i 2 --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
else
  printf '[warning] nvidia-smi not found in PATH\n'
fi
