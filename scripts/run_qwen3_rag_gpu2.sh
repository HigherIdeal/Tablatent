#!/usr/bin/env bash
set -euo pipefail

# This experiment is reserved for physical NVIDIA GPU index 2.
# CUDA remaps physical GPU 2 to logical cuda:0 inside the Python process.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

printf '[gpu] mandatory physical GPU: 2\n'
printf '[gpu] CUDA_VISIBLE_DEVICES=%s (physical GPU 2 -> logical cuda:0)\n' "$CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -i 2 --query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu --format=csv,noheader
fi

exec python scripts/run_qwen3_rag.py "$@"
