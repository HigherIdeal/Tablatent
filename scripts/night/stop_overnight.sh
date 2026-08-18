#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-outputs/night_20260819}"
for name in gpu2 gpu3; do
  pid_file="$OUT/$name/pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name PID=$pid"
      kill "$pid"
    else
      echo "$name PID=$pid is not running"
    fi
  fi
done
if [[ -f "$OUT/summary_watcher.pid" ]]; then
  pid="$(cat "$OUT/summary_watcher.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping summary watcher PID=$pid"
    kill "$pid"
  fi
fi
