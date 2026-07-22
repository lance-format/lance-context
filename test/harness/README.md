# lance-context test environment + harness

A Docker Compose stack that brings up the **full control-plane + data-plane** so an
AI agent (or human) can exercise the real HTTP surface end to end — the surface that
today is only covered by `#[ignore]`d Rust tests because they need a live etcd and
object store.

## Why this exists

The master's record-list endpoint (and its `?source=fragments|wal|all` selector shipped
in PR #170) can only be integration-tested against a running etcd + object store. Those
tests are marked `#[ignore]` and never run in CI, so the HTTP layer and UI had no
automated coverage. This harness closes that gap: `smoke.sh` makes real HTTP calls and
asserts the source-selector semantics.

## Topology

| Service    | Role                                   | Host port |
|------------|----------------------------------------|-----------|
| `minio`    | S3-compatible object store (DATA_DIR)  | 9000 (API), 9001 (console) |
| `etcd`     | master scheduler queue / locks         | 2379      |
| `worker-0` | data-plane writer, MemWAL shard `worker-0` | 3001  |
| `worker-1` | data-plane writer, MemWAL shard `worker-1` | 3002  |
| `master`   | control-plane admin API + UI           | 8090      |

All services share `DATA_DIR=s3://lance-context` on MinIO. Workers write rollout
datasets there; the master discovers and reads them. MemWAL self-merge is disabled
(`ROLLOUT_MERGE_AFTER_GENERATIONS=0`) so appended rows stay **pending in the WAL** —
which is exactly what makes `?source=fragments` vs `wal` vs `all` observably different.

## Prerequisites

- Docker with Compose v2 (`docker compose`)
- `curl` and `jq` on the host (for `smoke.sh`)

## Usage

```bash
# Build images and bring the stack up (waits for all healthchecks):
test/harness/up.sh

# Run the end-to-end smoke test (asserts source-selector semantics):
test/harness/smoke.sh

# Tear down (removes volumes for a clean slate):
test/harness/down.sh
```

`up.sh --no-build` skips the image build if images already exist.
`down.sh --keep-volumes` retains MinIO + etcd data across runs.

## What `smoke.sh` asserts

1. Create a rollout store on `worker-0` and append 3 records (they land in MemWAL,
   un-merged).
2. Wait for the master to discover the experiment.
3. `GET .../records?source=fragments` → **0** rows (base table only, nothing merged yet).
4. `GET .../records?source=wal` → **3** rows (the pending generations).
5. `GET .../records?source=all` → **3** rows (base ∪ WAL union).
6. Response JSON echoes the resolved `source`.
7. No `source` param defaults to `fragments`.
8. An unknown `source` value returns HTTP `400`.

## Manual poking (for agents)

```bash
# Create a store + append on a worker:
curl -X POST localhost:3001/api/v1/rollouts \
  -H 'Content-Type: application/json' -d '{"name":"exp1"}'
curl -X POST localhost:3001/api/v1/rollouts/exp1/records \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"id":"a","rollout_id":"r","role":"assistant","content":"hi"}]}'

# Browse via the master under each source:
curl 'localhost:8090/api/v1/experiments/exp1/records?source=fragments' | jq
curl 'localhost:8090/api/v1/experiments/exp1/records?source=wal' | jq
curl 'localhost:8090/api/v1/experiments/exp1/records?source=all' | jq

# List experiments / open the UI:
curl localhost:8090/api/v1/experiments | jq
open http://localhost:8090            # admin UI (Fragments / WAL / All tabs)
```

## Notes / limitations

- The images build the Rust workspace from source (protoc + openssl), so the first
  `up.sh` is slow (a full release build). Subsequent runs reuse the BuildKit cache.
- The master's WAL-merge and compaction sweeps are disabled in this stack
  (`MERGE_WAL_INTERVAL_SECS=0`, `COMPACTION_INTERVAL_SECS=0`) so WAL rows stay pending
  and the source split is deterministic. Enqueue merges/compactions manually via the
  `/api/v1/tasks` endpoint if you want to test the scheduler.
