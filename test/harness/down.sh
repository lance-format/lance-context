#!/usr/bin/env bash
# Tear down the lance-context test environment.
#
# Usage: test/harness/down.sh [--keep-volumes]
#
# By default removes containers AND volumes (fresh state next `up`). Pass
# --keep-volumes to retain the MinIO bucket + etcd data across runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${HERE}/../docker-compose.yml"

VOL_FLAG="-v"
if [[ "${1:-}" == "--keep-volumes" ]]; then
  VOL_FLAG=""
fi

echo ">> tearing down lance-context test stack"
# shellcheck disable=SC2086
docker compose -f "${COMPOSE_FILE}" down ${VOL_FLAG}
