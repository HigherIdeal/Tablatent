#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HOURS="${1:-7.67}"
CONFIG="${2:-experiments/configs/night_campaign_20260819.yaml}"
OUT="outputs/night_20260819"
mkdir -p "$OUT/gpu2" "$OUT/gpu3"

if [[ -f "$OUT/gpu2/pid" ]] && kill -0 "$(cat "$OUT/gpu2/pid")" 2>/dev/null; then
  echo "GPU2 worker already running: PID $(cat "$OUT/gpu2/pid")"
  exit 1
fi
if [[ -f "$OUT/gpu3/pid" ]] && kill -0 "$(cat "$OUT/gpu3/pid")" 2>/dev/null; then
  echo "GPU3 worker already running: PID $(cat "$OUT/gpu3/pid")"
  exit 1
fi

nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
  python scripts/night/run_gpu2_structure.py \
    --config "$CONFIG" --hours "$HOURS" --gpu 2 \
  > "$OUT/gpu2/worker.log" 2>&1 < /dev/null &
GPU2_PID=$!
echo "$GPU2_PID" > "$OUT/gpu2/pid"

nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
  python scripts/night/run_gpu3_calibration.py \
    --config "$CONFIG" --hours "$HOURS" --gpu 3 \
  > "$OUT/gpu3/worker.log" 2>&1 < /dev/null &
GPU3_PID=$!
echo "$GPU3_PID" > "$OUT/gpu3/pid"

nohup env PYTHONUNBUFFERED=1 \
  python scripts/night/watch_summary.py --root "$OUT" --interval 60 --hours 8.5 \
  > "$OUT/summary_watcher.log" 2>&1 < /dev/null &
WATCH_PID=$!
echo "$WATCH_PID" > "$OUT/summary_watcher.pid"

cat <<EOF
Overnight campaign launched.
  GPU2 structure     PID=$GPU2_PID  log=$OUT/gpu2/worker.log
  GPU3 calibration   PID=$GPU3_PID  log=$OUT/gpu3/worker.log
  summary watcher    PID=$WATCH_PID log=$OUT/summary_watcher.log

Live report: $OUT/overnight_report.md
GPU2 best:   $OUT/gpu2/best.md
GPU3 best:   $OUT/gpu3/best.md
EOF
