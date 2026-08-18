#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HOURS="${1:-8.0}"
CONFIG="${2:-experiments/configs/night_campaign_20260819.yaml}"
OUT="outputs/night_20260819"
mkdir -p "$OUT/gpu2" "$OUT/gpu3"

# Fail before detaching if the newly added Python files contain a syntax/import error.
python -m py_compile \
  src/bitaboost/night/common.py \
  src/bitaboost/night/gpu2_structure.py \
  src/bitaboost/night/gpu3_calibration.py \
  src/bitaboost/night/summary.py \
  scripts/night/run_gpu2_structure.py \
  scripts/night/run_gpu3_calibration.py \
  scripts/night/summarize_night.py \
  scripts/night/watch_summary.py
NIGHT_CONFIG="$CONFIG" PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import os
from bitaboost.night.common import load_yaml
from bitaboost.night.gpu2_structure import _initial_trials
from bitaboost.night.gpu3_calibration import _method_queue
cfg = load_yaml(os.environ["NIGHT_CONFIG"])
assert _initial_trials(cfg), "GPU2 trial queue is empty"
assert _method_queue(), "GPU3 method queue is empty"
print("[preflight] night campaign Python import/config check passed", flush=True)
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -i 2,3 --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
fi

# A one-tree synthetic fit catches CUDA/CatBoost device problems before the terminal
# is detached. It is not part of the research results.
for GPU in 2 3; do
  CUDA_VISIBLE_DEVICES="$GPU" python - <<'PY'
import numpy as np
from catboost import CatBoostClassifier
rng = np.random.default_rng(0)
x = rng.normal(size=(512, 4)).astype(np.float32)
y = (x[:, 0] + 0.2 * x[:, 1] > 0).astype(np.int8)
model = CatBoostClassifier(
    iterations=1,
    depth=2,
    loss_function="Logloss",
    task_type="GPU",
    devices="0",
    allow_writing_files=False,
    logging_level="Silent",
)
model.fit(x, y)
print("[preflight] CatBoost GPU synthetic fit passed", flush=True)
PY
done

if [[ -f "$OUT/gpu2/pid" ]] && kill -0 "$(cat "$OUT/gpu2/pid")" 2>/dev/null; then
  echo "GPU2 worker already running: PID $(cat "$OUT/gpu2/pid")"
  echo "Nothing new was launched. It is safe to close this terminal."
  exit 0
fi
if [[ -f "$OUT/gpu3/pid" ]] && kill -0 "$(cat "$OUT/gpu3/pid")" 2>/dev/null; then
  echo "GPU3 worker already running: PID $(cat "$OUT/gpu3/pid")"
  echo "Nothing new was launched. It is safe to close this terminal."
  exit 0
fi

# nohup + redirected stdin/stdout/stderr makes the workers independent of the SSH PTY.
# setsid adds a separate session when available, making terminal closure even more explicit.
launch_detached() {
  local logfile="$1"
  shift
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$@" > "$logfile" 2>&1 < /dev/null &
  else
    nohup "$@" > "$logfile" 2>&1 < /dev/null &
  fi
  echo $!
}

GPU2_PID="$(launch_detached "$OUT/gpu2/worker.log" \
  env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
  python scripts/night/run_gpu2_structure.py \
    --config "$CONFIG" --hours "$HOURS" --gpu 2)"
echo "$GPU2_PID" > "$OUT/gpu2/pid"

GPU3_PID="$(launch_detached "$OUT/gpu3/worker.log" \
  env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
  python scripts/night/run_gpu3_calibration.py \
    --config "$CONFIG" --hours "$HOURS" --gpu 3)"
echo "$GPU3_PID" > "$OUT/gpu3/pid"

WATCH_PID="$(launch_detached "$OUT/summary_watcher.log" \
  env PYTHONUNBUFFERED=1 \
  python scripts/night/watch_summary.py --root "$OUT" --interval 60 --hours 8.5)"
echo "$WATCH_PID" > "$OUT/summary_watcher.pid"

# The launcher itself does NOT wait for the startup check. A detached checker records
# whether the workers survived their first 15 seconds, while this shell returns now.
CHECK_LOG="$OUT/startup_check.log"
if command -v setsid >/dev/null 2>&1; then
  nohup setsid bash -c '
    sleep 15
    OUT="$1"; G2="$2"; G3="$3"; W="$4"
    {
      date -Is
      for ITEM in "gpu2:$G2" "gpu3:$G3" "summary:$W"; do
        NAME="${ITEM%%:*}"; PID="${ITEM##*:}"
        if kill -0 "$PID" 2>/dev/null; then
          echo "OK $NAME PID=$PID alive"
        else
          echo "ERROR $NAME PID=$PID exited"
          if [[ "$NAME" == "gpu2" || "$NAME" == "gpu3" ]]; then
            tail -n 100 "$OUT/$NAME/worker.log" 2>/dev/null || true
          fi
        fi
      done
    } > "$OUT/startup_check.log" 2>&1
  ' _ "$OUT" "$GPU2_PID" "$GPU3_PID" "$WATCH_PID" > /dev/null 2>&1 < /dev/null &
else
  nohup bash -c '
    sleep 15
    OUT="$1"; G2="$2"; G3="$3"; W="$4"
    {
      date -Is
      for ITEM in "gpu2:$G2" "gpu3:$G3" "summary:$W"; do
        NAME="${ITEM%%:*}"; PID="${ITEM##*:}"
        if kill -0 "$PID" 2>/dev/null; then
          echo "OK $NAME PID=$PID alive"
        else
          echo "ERROR $NAME PID=$PID exited"
        fi
      done
    } > "$OUT/startup_check.log" 2>&1
  ' _ "$OUT" "$GPU2_PID" "$GPU3_PID" "$WATCH_PID" > /dev/null 2>&1 < /dev/null &
fi

echo
cat <<EOF
Overnight campaign DETACHED.
  GPU2 structure     PID=$GPU2_PID  log=$OUT/gpu2/worker.log
  GPU3 calibration   PID=$GPU3_PID  log=$OUT/gpu3/worker.log
  summary watcher    PID=$WATCH_PID log=$OUT/summary_watcher.log

>>> YOU MAY CLOSE THIS TERMINAL / SSH SESSION NOW. <<<

A detached 15-second startup check will be written to:
  $CHECK_LOG

Live report: $OUT/overnight_report.md
GPU2 best:   $OUT/gpu2/best.md
GPU3 best:   $OUT/gpu3/best.md
EOF
exit 0
