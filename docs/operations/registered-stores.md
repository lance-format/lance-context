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
unchanged-compaction suppression are shared. `index_id` creates a missing
configured primary-key index and extends incomplete indexes, including existing application BTree indexes;
it does not invent secondary index definitions.

### Independent periodic indexing

Set `INDEX_INTERVAL_SECS=60` (or `--index-interval-secs 60`) on the masters to
check base-table index coverage once a minute. The default is `0` (disabled).
This sweep uses the existing stats scanner's inventory of nonempty tables, so
keep `STATS_SCAN_INTERVAL_SECS` enabled too. It does not depend on `MIN_FRAGMENTS`,
quiet hours, or the compaction interval. It can run with compaction disabled.

Set `INDEX_PREPARE_TARGETS=*` to reuse the existing prepared executor for
rollout and registered tables (or select exact targets). Ordinary and prepared
execution share the same coverage plan. Missing key indexes use the existing
uncommitted key builder; incomplete indexes use Lance's incremental optimize
plan/build/merge stages. Preparation writes immutable files without holding the
table writer. Publication acquires the existing maintenance ownership, with
`INDEX_COMMIT_WAIT_SECS` bounding time without progress from the current writer.
Source rewrites or same-index changes invalidate preparation and require retry;
concurrent appends remain valid and are indexed by a later pass.

Upgrade all index executors sharing the queue before enabling periodic discovery:
older prepared executors only maintain the primary key. This setting does not
enable native merge or compaction preparation for registered stores. Retained
snapshot requests always keep their existing full-coverage validation and tag
creation under table ownership, even with `INDEX_PREPARE_TARGETS=*`.

Only tables missing their configured key index or with uncovered fragments in
that index or an existing BTree index get an `index_id` task. Checks read current
manifest/index metadata, including fragment-reuse mappings, not row payloads.
Coverage unions all deltas of each named index. Fully covered tables do not
produce tasks or index commits; executing tasks recheck coverage before writing.
Index updates merge deltas so later data compaction can still combine fragments.
The separate retained-snapshot API below continues to validate and pin its result.

The sweep elects one owner among identically configured masters, bounds metadata
checks with `SCAN_CONCURRENCY`, and queues at most 256 new tasks per round. It
uses normal task deduplication, failure cooldown, per-table maintenance ownership,
and the `TASK_CONCURRENCY` pool. A slow round delays the next one rather than
building a backlog of timer ticks. This is eventual maintenance, not a guarantee
that every table is fully indexed within 60 seconds.

This checks rollout tables and the configured registered-store kind. Generic
stores remain excluded. WAL-only data must first reach base through the separate
WAL-merge policy; the index sweep neither merges WAL nor compacts data files.
`INDEX_AFTER_COMPACTION` remains independent and can stay enabled.

Observe `master_periodic_sweep_{started,enqueued,errors}_total{kind="index_id"}`,
`master_index_sweep_checks_total{result="needed|skipped|error"}`, and the existing
`index_id` task duration/failure metrics. Per-table lookup failures are logged
and counted without stopping checks for other tables.

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
