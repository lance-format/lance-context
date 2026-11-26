# Immutable application WAL merges

Registered applications can use the same Master-local append executor as
RolloutStore. Workers still ingest, persist WAL, publish generations, and replay
their stable shard after a restart. The Master uses the existing
`plan → stage → commit → reconcile` path; no second merge implementation or
scheduler is introduced.

`AppendCoordinator::open_with_options` and `stage_with_options` accept an
`AppendOptions` containing the application schema and one Utf8 identity column.
The original `open` and `stage` APIs keep the rollout schema, `id` key, and
cutover/replay checks against base. Updates and tombstones must keep using keyed
replacement.

`ReplayPolicy::AllowDuplicates` skips base-key lookups and replay-boundary metadata
for immutable applications whose readers deduplicate identities. It retains
within-stage key reduction. This makes **merge retries** idempotent, not arbitrary
**ingest retries**: new fragments and per-shard generation watermarks commit
atomically, and reconciliation drains only covered generations. An SDK retry in
another shard or generation can still produce another physical row.

Readers must apply each full event identity once before replaying changes.
B-trees and ordinary compaction are not uniqueness constraints. Read limits must
still account for physical duplicates. The configured WAL lifecycle determines
physical file retention; draining a generation does not necessarily delete its
blobs.

## Master configuration

Opt in with `MANAGED_IMMUTABLE_APPEND=true` and a `MANAGED_STORE_KIND`. The Master
resolves the registered URI and reads its persisted schema/primary-key metadata;
it never infers a rollout path for that application. Also configure:

- `ROLLOUT_APPEND_LOCAL=true` and `FENCE_LEGACY_MAINTENANCE=true`.
- `CATCHUP_SHARDS` with stable current and historical Worker identities.
- The existing append concurrency, generation, byte, and shared-memory limits.
- A positive `MERGE_WAL_CONCURRENCY`. Registered tables use this ordinary queue
  pool; set `ROLLOUT_APPEND_LOCAL_TASK_CONCURRENCY=0` when no rollout targets use
  the separate resident-only pool.

Registered-store statistics scans discover pending WAL; bounded append passes
request continuation through the existing queue. Compaction and indexing retain
their existing scheduling and share the maintenance ownership/version fence.
Master replacement recovers the recorded execution against the registered URI.
Unresolved historical Worker executions remain blocked until their owner is
reconciled; enabling this option does not certify a safe cutover.

Before enabling Master-local writes, stop admitting Worker base maintenance,
drain previously admitted work, and verify no unresolved Worker executions remain.
Worker ingestion/publication may continue. Reverse this order for rollback:
drain Master base writers before re-enabling Worker merging. Do not delete etcd
ownership records to bypass an uncertain write.

Tests in `rollout_append/application_tests.rs` cover application schemas, replay
policy, and commit-before-drain recovery. Master managed tests exercise two shards
through the real queue, registered-URI recovery, and refusal to take over an
unresolved legacy Worker write. The original rollout regression suite remains
the compatibility check for the shared path.
