#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HOURS="${1:-8.0}"
CONFIG="${2:-experiments/configs/night_campaign_20260819.yaml}"
OUT="outputs/night_20260819"
mkdir -p "$OUT/gpu2" "$OUT/gpu3"

# Check for an already-running campaign BEFORE touching CUDA. A previous launcher can
# have detached the workers even if its foreground shell was interrupted with Ctrl+C.
for ITEM in "gpu2:$OUT/gpu2/pid" "gpu3:$OUT/gpu3/pid"; do
  NAME="${ITEM%%:*}"
  PIDFILE="${ITEM#*:}"
  if [[ -f "$PIDFILE" ]]; then
    PID="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      echo "$NAME worker is already running: PID=$PID"
      echo "No new workers were launched. It is safe to close this terminal."
      exit 0
    fi
  fi
done

# Also catch detached workers whose pid files were lost/stale.
EXISTING="$(pgrep -af 'scripts/night/run_gpu2_structure.py|scripts/night/run_gpu3_calibration.py' || true)"
if [[ -n "$EXISTING" ]]; then
  echo "Existing overnight worker process(es) detected:"
  echo "$EXISTING"
  echo "Refusing to launch duplicates. Stop the old campaign first if you want a restart."
  exit 0
fi

# CPU-only preflight. Do not perform a throw-away CatBoost GPU fit here: CatBoost's
# allocator can reserve most of a GPU even for a tiny synthetic dataset and can race
# with a previously detached worker. The real workers are checked asynchronously.
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
  echo "[preflight] GPU state before detach:"
  nvidia-smi -i 2,3 --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader || true
fi

# nohup + setsid + redirected stdin/stdout/stderr makes the workers independent of
# the current SSH PTY. The shell returns immediately after PIDs are written.
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

# Detached checker records startup success/failure without holding the SSH shell.
CHECK_LOG="$OUT/startup_check.log"
nohup setsid bash -c '
  sleep 20
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
          tail -n 120 "$OUT/$NAME/worker.log" 2>/dev/null || true
        fi
      fi
    done
  } > "$OUT/startup_check.log" 2>&1
' _ "$OUT" "$GPU2_PID" "$GPU3_PID" "$WATCH_PID" > /dev/null 2>&1 < /dev/null &

echo
cat <<EOF
Overnight campaign DETACHED.
  GPU2 structure     PID=$GPU2_PID  log=$OUT/gpu2/worker.log
  GPU3 calibration   PID=$GPU3_PID  log=$OUT/gpu3/worker.log
  summary watcher    PID=$WATCH_PID log=$OUT/summary_watcher.log

>>> YOU MAY CLOSE THIS TERMINAL / SSH SESSION NOW. <<<

Detached startup check (written ~20s later):
  $CHECK_LOG

Live report: $OUT/overnight_report.md
GPU2 best:   $OUT/gpu2/best.md
GPU3 best:   $OUT/gpu3/best.md
EOF
exit 0
