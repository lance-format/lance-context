#!/usr/bin/env bash
# Stop the containerless harness started by native-up.sh.
#
# Usage:
#   test/harness/native-down.sh              # stop processes, keep data
#   test/harness/native-down.sh --purge      # stop + delete data/logs/etcd state
set -euo pipefail

HARNESS_DIR="${HARNESS_DIR:-/tmp/lance-harness}"
PID_DIR="${HARNESS_DIR}/pids"

for name in master worker-0 etcd; do
  pidf="$PID_DIR/$name.pid"
  if [[ -f "$pidf" ]]; then
    pid="$(cat "$pidf")"
    if kill -0 "$pid" 2>/dev/null; then
      echo ">> stopping $name (pid $pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidf"
  fi
done

if [[ "${1:-}" == "--purge" ]]; then
  echo ">> purging ${HARNESS_DIR} data/logs/etcd state"
  rm -rf "${HARNESS_DIR}/data" "${HARNESS_DIR}/logs" "${HARNESS_DIR}/etcd-data"
fi

echo ">> down"
