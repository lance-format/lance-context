#!/usr/bin/env bash
# Containerless test environment for sandboxes where Docker can't run.
#
# Some environments (this repo's dev sandbox included) have a Docker daemon but
# a kernel that forbids `unshare`/netlink, so no container can start. This script
# brings the SAME stack up as plain host processes instead:
#
#   etcd     : local static binary (downloaded on first run) on 127.0.0.1:2379
#   worker-0 : lance-context-server on 127.0.0.1:3001, shard "worker-0"
#   master   : lance-context-master on 127.0.0.1:8090
#
# Storage is the local filesystem (DATA_DIR=<workdir>/data), not MinIO — the
# object-store abstraction treats a plain path as file://, so master + worker
# share it exactly like they'd share an S3 prefix.
#
# Usage:
#   test/harness/native-up.sh            # build (if needed), start etcd+worker+master
#   test/harness/native-up.sh --smoke    # ...then run smoke.sh against it
#   test/harness/native-down.sh          # stop everything
#
# State (pids, logs, data, etcd binary) lives under $HARNESS_DIR
# (default /tmp/lance-harness).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS_DIR="${HARNESS_DIR:-/tmp/lance-harness}"
ETCD_VERSION="${ETCD_VERSION:-v3.5.13}"
DATA_DIR="${HARNESS_DIR}/data"
LOG_DIR="${HARNESS_DIR}/logs"
PID_DIR="${HARNESS_DIR}/pids"

MASTER_PORT="${MASTER_PORT:-8090}"
WORKER_PORT="${WORKER_PORT:-3001}"
ETCD_CLIENT_URL="${ETCD_CLIENT_URL:-http://127.0.0.1:2379}"

mkdir -p "$DATA_DIR" "$LOG_DIR" "$PID_DIR"

log() { echo ">> $*"; }
die() { echo "FAIL: $*" >&2; exit 1; }

wait_http() {
  local url="$1" name="$2" tries="${3:-60}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  die "$name did not become healthy at $url"
}

start_bg() {
  # start_bg <name> <logfile> <cmd...>
  local name="$1" logf="$2"; shift 2
  if [[ -f "$PID_DIR/$name.pid" ]] && kill -0 "$(cat "$PID_DIR/$name.pid")" 2>/dev/null; then
    log "$name already running (pid $(cat "$PID_DIR/$name.pid"))"
    return 0
  fi
  log "starting $name -> $logf"
  nohup "$@" >"$logf" 2>&1 &
  echo $! >"$PID_DIR/$name.pid"
}

# ---------------------------------------------------------------------------
# etcd (download static binary once)
# ---------------------------------------------------------------------------
ETCD_BIN="${HARNESS_DIR}/etcd"
if [[ ! -x "$ETCD_BIN" ]]; then
  log "downloading etcd ${ETCD_VERSION}"
  tarball="etcd-${ETCD_VERSION}-linux-amd64.tar.gz"
  curl -sfL -o "${HARNESS_DIR}/${tarball}" \
    "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/${tarball}" \
    || die "could not download etcd"
  tar xzf "${HARNESS_DIR}/${tarball}" -C "$HARNESS_DIR"
  cp "${HARNESS_DIR}/etcd-${ETCD_VERSION}-linux-amd64/etcd" "$ETCD_BIN"
fi

start_bg etcd "$LOG_DIR/etcd.log" \
  "$ETCD_BIN" --name harness --data-dir "${HARNESS_DIR}/etcd-data" \
    --advertise-client-urls "$ETCD_CLIENT_URL" \
    --listen-client-urls "$ETCD_CLIENT_URL" \
    --listen-peer-urls http://127.0.0.1:2380 \
    --initial-cluster harness=http://127.0.0.1:2380 \
    --initial-advertise-peer-urls http://127.0.0.1:2380
# etcd has no plain HTTP health path we curl easily; give it a moment.
sleep 3

# ---------------------------------------------------------------------------
# Build the Rust binaries (release) if missing
# ---------------------------------------------------------------------------
export PATH="$HOME/.cargo/bin:$PATH"
MASTER_BIN="${REPO}/target/release/lance-context-master"
WORKER_BIN="${REPO}/target/release/lance-context-server"
if [[ ! -x "$MASTER_BIN" || ! -x "$WORKER_BIN" ]]; then
  log "building lance-context-master + lance-context-server (release)"
  (cd "$REPO" && cargo build --release \
     -p lance-context-master -p lance-context-server) \
     || die "cargo build failed"
fi

# ---------------------------------------------------------------------------
# Build the UI (optional; master serves it when UI_DIR is set)
# ---------------------------------------------------------------------------
UI_DIST="${REPO}/crates/lance-context-master/ui/dist"
if [[ ! -f "${UI_DIST}/index.html" ]] && command -v npm >/dev/null 2>&1; then
  log "building admin UI"
  (cd "${REPO}/crates/lance-context-master/ui" && npm ci --silent && npm run build) || true
fi

# ---------------------------------------------------------------------------
# worker-0 (data-plane)
# ---------------------------------------------------------------------------
start_bg worker-0 "$LOG_DIR/worker-0.log" \
  env INSTANCE_ID=worker-0 ROLLOUT_MERGE_AFTER_GENERATIONS=0 \
  "$WORKER_BIN" --host 127.0.0.1 --port "$WORKER_PORT" --data-dir "$DATA_DIR"
wait_http "http://127.0.0.1:${WORKER_PORT}/api/v1/health" worker-0

# ---------------------------------------------------------------------------
# master (control-plane + UI)
# ---------------------------------------------------------------------------
UI_ARG=()
[[ -f "${UI_DIST}/index.html" ]] && UI_ARG=(--ui-dir "$UI_DIST")
start_bg master "$LOG_DIR/master.log" \
  env ETCD_ENDPOINTS="$ETCD_CLIENT_URL" \
      WORKER_ENDPOINTS="http://127.0.0.1:${WORKER_PORT}" \
      STATS_SCAN_INTERVAL_SECS=10 \
      MERGE_WAL_INTERVAL_SECS=0 \
      COMPACTION_INTERVAL_SECS=0 \
  "$MASTER_BIN" --host 127.0.0.1 --port "$MASTER_PORT" --data-dir "$DATA_DIR" "${UI_ARG[@]}"
wait_http "http://127.0.0.1:${MASTER_PORT}/metrics" master

log "stack is up:"
echo "   etcd     : ${ETCD_CLIENT_URL}"
echo "   worker-0 : http://127.0.0.1:${WORKER_PORT}"
echo "   master   : http://127.0.0.1:${MASTER_PORT}  (UI + admin API)"
echo "   data dir : ${DATA_DIR}"
echo "   logs     : ${LOG_DIR}"

if [[ "${1:-}" == "--smoke" ]]; then
  log "running smoke test"
  MASTER_URL="http://127.0.0.1:${MASTER_PORT}" \
  WORKER_URL="http://127.0.0.1:${WORKER_PORT}" \
    "$(dirname "${BASH_SOURCE[0]}")/smoke.sh"
fi
