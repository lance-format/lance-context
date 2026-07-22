#!/usr/bin/env bash
# Bring up the lance-context test environment (MinIO + etcd + master + workers).
#
# Usage: test/harness/up.sh [--no-build]
#
# Idempotent-ish: `docker compose up` will reuse running containers. Waits for
# every service healthcheck (--wait) so callers can immediately hit the API.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${HERE}/../docker-compose.yml"

BUILD_FLAG="--build"
if [[ "${1:-}" == "--no-build" ]]; then
  BUILD_FLAG=""
fi

echo ">> bringing up lance-context test stack (${COMPOSE_FILE})"
# shellcheck disable=SC2086
docker compose -f "${COMPOSE_FILE}" up -d ${BUILD_FLAG} --wait

echo ">> stack is up:"
docker compose -f "${COMPOSE_FILE}" ps

cat <<'EOF'

Endpoints:
  master admin API + UI : http://localhost:8090
  master metrics        : http://localhost:8090/metrics
  worker-0 (data-plane) : http://localhost:3001
  worker-1 (data-plane) : http://localhost:3002
  MinIO S3 API          : http://localhost:9000  (minioadmin/minioadmin)
  MinIO console         : http://localhost:9001

Next: test/harness/smoke.sh
EOF
