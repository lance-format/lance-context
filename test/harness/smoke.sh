#!/usr/bin/env bash
# End-to-end smoke test for the lance-context test environment.
#
# This exercises the real HTTP surface that today is only covered by #[ignore]d
# Rust tests (they need a live etcd + object store). In particular it asserts
# the record-list SOURCE SELECTOR shipped in PR #170:
#
#   GET /api/v1/experiments/{name}/records?source=fragments  -> base table only
#   GET /api/v1/experiments/{name}/records?source=wal        -> flushed WAL only
#   GET /api/v1/experiments/{name}/records?source=all        -> base ∪ WAL
#
# Flow:
#   1. worker-0 creates a rollout store + appends rows (they land in MemWAL,
#      un-merged because ROLLOUT_MERGE_AFTER_GENERATIONS=0).
#   2. master discovers the experiment and browses it under each source.
#   3. Assert: fragments omits the un-merged rows; wal shows exactly them; all
#      is the union.
#
# Usage: test/harness/smoke.sh
# Exit code 0 == all assertions passed.
set -euo pipefail

MASTER="${MASTER_URL:-http://localhost:8090}"
WORKER="${WORKER_URL:-http://localhost:3001}"
EXP="smoke-$(date +%s)"

pass() { echo "  PASS: $*"; }
fail() { echo "  FAIL: $*" >&2; exit 1; }

require() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required tool: $1"
}
require curl
require jq

echo ">> smoke test against master=${MASTER} worker=${WORKER} experiment=${EXP}"

# ---------------------------------------------------------------------------
# 1. Create a rollout store on the worker and append un-merged rows.
# ---------------------------------------------------------------------------
echo ">> [1] create rollout store '${EXP}' on worker"
curl -sf -X POST "${WORKER}/api/v1/rollouts" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"${EXP}\"}" >/dev/null \
  || fail "could not create rollout store"
pass "store created"

echo ">> [2] append 3 rollout records"
curl -sf -X POST "${WORKER}/api/v1/rollouts/${EXP}/records" \
  -H 'Content-Type: application/json' \
  -d "$(cat <<JSON
{
  "records": [
    {"id":"r1","rollout_id":"roll-1","role":"assistant","content":"alpha"},
    {"id":"r2","rollout_id":"roll-1","role":"assistant","content":"bravo"},
    {"id":"r3","rollout_id":"roll-2","role":"assistant","content":"charlie"}
  ]
}
JSON
)" >/dev/null \
  || fail "could not append records"
pass "3 records appended (pending in MemWAL)"

# Give the master's stats scanner a moment to discover the new dataset.
echo ">> [3] wait for master to discover experiment '${EXP}'"
for _ in $(seq 1 30); do
  if curl -sf "${MASTER}/api/v1/experiments/${EXP}?fresh=true" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -sf "${MASTER}/api/v1/experiments/${EXP}?fresh=true" >/dev/null \
  || fail "master never discovered experiment"
pass "experiment discovered by master"

# ---------------------------------------------------------------------------
# Helper: count records returned by a given source, echo the JSON's source.
# ---------------------------------------------------------------------------
records_json() {
  local src="$1"
  curl -sf "${MASTER}/api/v1/experiments/${EXP}/records?source=${src}&limit=100"
}

echo ">> [4] assert source selector semantics"

FRAG_JSON="$(records_json fragments)"
WAL_JSON="$(records_json wal)"
ALL_JSON="$(records_json all)"

frag_n=$(echo "$FRAG_JSON" | jq '.records | length')
wal_n=$(echo "$WAL_JSON" | jq '.records | length')
all_n=$(echo "$ALL_JSON" | jq '.records | length')

frag_src=$(echo "$FRAG_JSON" | jq -r '.source')
wal_src=$(echo "$WAL_JSON" | jq -r '.source')
all_src=$(echo "$ALL_JSON" | jq -r '.source')

echo "     fragments: n=${frag_n} source=${frag_src}"
echo "     wal:       n=${wal_n} source=${wal_src}"
echo "     all:       n=${all_n} source=${all_src}"

# Response echoes the resolved source (added in #170).
[[ "$frag_src" == "fragments" ]] || fail "fragments response source != fragments"
[[ "$wal_src" == "wal" ]]        || fail "wal response source != wal"
[[ "$all_src" == "all" ]]        || fail "all response source != all"
pass "response echoes resolved source"

# Un-merged rows: fragments (base only) must NOT see them; wal must; all == union.
[[ "$frag_n" -eq 0 ]] || fail "fragments should be empty before merge, got ${frag_n}"
[[ "$wal_n"  -eq 3 ]] || fail "wal should show 3 pending rows, got ${wal_n}"
[[ "$all_n"  -eq 3 ]] || fail "all should show 3 rows (base ∪ wal), got ${all_n}"
pass "fragments omits un-merged rows; wal shows exactly them; all is the union"

# Default (no source param) == fragments per #170.
DEF_JSON="$(curl -sf "${MASTER}/api/v1/experiments/${EXP}/records?limit=100")"
def_src=$(echo "$DEF_JSON" | jq -r '.source')
[[ "$def_src" == "fragments" ]] || fail "default source should be fragments, got ${def_src}"
pass "default source is fragments"

# Unknown source is rejected with 400.
code=$(curl -s -o /dev/null -w '%{http_code}' \
  "${MASTER}/api/v1/experiments/${EXP}/records?source=bogus")
[[ "$code" == "400" ]] || fail "unknown source should be 400, got ${code}"
pass "unknown source rejected with 400"

echo ""
echo ">> ALL SMOKE ASSERTIONS PASSED"
