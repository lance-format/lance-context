# Maintaining application-owned tables

The master can maintain tables with an application-defined schema through one
additional store registry. Set `MANAGED_STORE_KIND=events` (or
`--managed-store-kind events`) on every master replica sharing the task queue.
The default is unset, which preserves existing behavior.

This lets an application use `StorageBase` without making the master depend on
its record types. The application still owns table creation, writes, reads and
its worker HTTP routes.

## Registry and schema contract

- Use a nonempty kind containing only lowercase ASCII letters, digits and `_`.
  `rollout` and `generic` are reserved for the existing built-in paths.
- Register each table's logical name and actual URI in that kind's registry.
  With the Lance registry backend, the registry is
  `${DATA_DIR}/_registry.events.lance`. `REGISTRY_BACKEND` and
  `REGISTRY_MIRROR` also apply to this registry; `open_registry` accepts
  the runtime kind for applications using the core Rust API.
- The master resolves the URI from the registry. It does not derive the table's
  path from its name or create a missing table.
- The persisted Arrow schema must declare exactly one primary-key field with
  field metadata `lance-schema:unenforced-primary-key = "true"`. The key does not
  have to be called `id`. Missing or composite declarations are rejected.
- Task targets use `events:<name>`, including the prefix in deduplication and
  coordination keys. A rollout table, generic store and registered table with
  the same bare name remain separate targets.

For example, after the application registers `trial` at its chosen table URI,
submit maintenance through the existing master task API:

```sh
curl -X POST http://localhost:8090/api/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"kind":"index_id","target":"events:trial"}'
```

`index_id` is the existing task API name: for a registered table it indexes the
**declared key**, using `ROLLOUT_KEY_INDEX_TYPE` (`btree` by default). Keep that
setting consistent with the application's workers.

## What the master does

The periodic scanner reads the registered table's manifest, row/fragment counts
and published WAL-generation counts. It does not open a worker's WAL for writing.
Those statistics feed the existing WAL-merge and compaction scheduling policies.
Registered tables are excluded from rollout-specific cold retirement and native
merge-demand publication.

`compact` opens the registry URI as a `StorageBase` using the table's persisted
schema and declared primary key. Rollout tables use this same path; compaction
does not need a rollout-specific store interface or evolve business columns.
Legacy rollout tasks still accept the conventional URI until discovery registers
the table. After rewriting fragments, the shared scheduler queues a dependent
`index_id` task when `INDEX_AFTER_COMPACTION` is enabled. Statistics refresh and
unchanged-compaction suppression are shared. `index_id` builds or replaces the
configured primary-key index and maintains existing application BTree indexes;
it does not invent secondary index definitions.

`merge_wal` fans out to each configured worker using:

```text
POST <worker>/api/v1/events/<name>/merge-wal
```

The name is encoded as a URL path segment. The application must implement this
route and return a successful HTTP response with JSON such as `{"reclaimed":3}`.
A worker with nothing to merge returns `{"reclaimed":0}`; HTTP 404 is a failure.
Each request has a 600-second deadline. A timeout does not prove that the worker
stopped; the existing durable legacy-execution record must be reconciled before
another execution is admitted. See the [maintenance runbook](maintenance-runbook.md)
for queue, backlog, failure-ledger and worker-progress diagnostics.

## Optional retained index snapshot

A registered-table `index_id` request can include `"snapshot_tag":"resume-<unique-id>"`.
Omitting it preserves the existing indexing behavior and creates no tag. Snapshot
requests require BTree key indexes and no task dependencies. Stop application
writers, publish all worker WAL, and finish merging it before requesting a snapshot;
the task checks for pending published WAL but is not a distributed write barrier.

The master builds the primary-key index, optimizes existing BTree indexes, and
checks their fragment coverage on the resulting dataset handle. Under the same
table ownership, it creates the tag at that exact version before marking the task
`done`. The task response includes `snapshot_tag` and `snapshot_version`; callers
must use that version rather than reopen latest. Indexing and tag creation are
separate storage operations, and a failed tag write fails the task.

Active requests with the same target and tag deduplicate. Different tags and
ordinary index tasks remain distinct, with the usual per-table serialization.
Retrying an existing tag validates and returns its retained version without moving
it. The caller owns the tag's lifetime and must delete it after all readers stop,
including cleanup of tags left by crashed clients. This API does not itself change
any reader's base/WAL semantics.

## Current boundaries

This adapter uses worker-owned legacy WAL merge. Startup rejects registered-target
or wildcard native ownership, drain, prepared-compaction/index and local-merge
selections, as well as global legacy fencing and automatic local merge. It does
not support `repair` for registered tables. An unresolved registered execution
requires owner reconciliation; rollout-specific storage recovery is not applied
to an unknown schema.

This capability does not define application checkpoint semantics, durable write
acknowledgements, retry idempotency or read-after-write guarantees. Those remain
part of the application's use of `StorageBase` and its WAL publication protocol.
