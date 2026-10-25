# Table-wide WAL demand admission

A writer currently requests a merge when its own shard reaches the configured
count threshold. With twenty writers and a threshold of eight, a table can hold
140 sealed generations without any writer requesting a merge. Periodic storage
scans eventually discover this work, but their cadence is not prompt admission.

The existing elected demand-controller leader now also consumes the existing
`demand-events` records. Other masters continue watching the leader key; they do
not independently read all writer events or contend to enqueue the same work.
The aggregate snapshot runs independently of dispatch and history maintenance.
It uses sequential bounded pages at a single revision, then resumes one Watch
per input prefix from that revision. Transport reconnect retains each prefix's
cursor; compaction requires a new consistent snapshot.

The input is a hint. Executor progress can be missing or delayed, so a high sum
of `sealed - merged` alone must not continuously create empty merges. Each table
has a durable `demand-controller/admitted-wal/<encoded-target>` frontier containing
the seal counts already entrusted to queued work. Admission uses the sum of
`sealed - max(merged, admitted)` across all shards, with saturating arithmetic.
Generation numbers retain the existing monotonic-per-shard semantics across
writer epochs. Admission does not infer storage commits from these hints.

A single transaction compares the controller's lease-backed token, the previous
frontier revision and the absence of task dedupe, then creates task, queue,
dedupe and frontier records together. A lost response is safe to replay. A new
leader reconstructs the same obligation from persisted frontiers and events.
Neither queued nor running dedupe advances the frontier; the active task may
not observe new writes. Dedupe is checked once per identity and its deletion
wakes the table through Watch. A new pass can then consume the retained seals.
No lease, storage fence, merge ownership or ambiguous commit is cleared here.
Existing execution recovery, bounded-pass continuation and failure handling
remain responsible for admitted tasks; this frontier is not a success receipt.

Admission preserves the union of automatic master policies, using the lowest
known positive count threshold among policies admitting that target. Old member
records without a threshold do not invent one. Draining policies do not admit
new aggregate work. The ordinary request and periodic discovery paths remain
available during mixed-version deployment and rollback.

The in-memory projection omits payloads, byte history and `sealed_times`. Bounds
cover 32,768 tables, 524,288 shard entries, 1,024 shards per table and a 128 MiB
conservative retained-state budget. Inputs and frontiers are bounded to 256 KiB.
A malformed shard event, frontier or dedupe value quarantines its table until a valid update,
while other tables can proceed. Capacity exhaustion is an observable failure,
not permission to discard durable demand or broaden the configured limits.
Invalid member policies are withdrawn without restarting every table. Failed
bootstrap retries are paced, with capacity failures limited to one attempt per
five minutes; a healthy informer never polls. Fresh writer events cannot bypass
a failed admission retry deadline. These error paths do not stop dispatch or
the existing periodic storage scans.

Historical hints are subject to an admission budget: at most twice the sum of
ordinary WAL merge slots advertised by live members, capped at 4,096 tasks.
Members which disable WAL claiming advertise zero; old members without the
field contribute zero. Shared-pool configuration uses the scheduler's effective
minimum of one slot. Reserved local-only pools are not counted. This budget
bounds this new admission source, not the existing total queue or execution
concurrency. Capacity shrink never cancels admitted tasks.

Frontiers also retain the admitted task identity. Matching it to the watched
dedupe reconstructs aggregate-owned queued/running work across restart. Excess
demand waits in the bounded cache without advancing any seal or issuing RPCs;
dedupe completion returns credit through Watch. Newer writer evidence receives
freed credit ahead of historical hints. Existing periodic discovery remains
responsible for older storage backlog, including hints already entrusted to a
task; it does not depend on new writer events.

Before publishing, the leader reserves one credit. An ambiguous response keeps
that reservation and suspends further aggregate publication until resolved.
A leader-fenced CAS rewrites only the PREVIOUS admitted frontier at the original
revision. Either the original enqueue already won and its actual state is read,
or this neutral rewrite invalidates the delayed enqueue's CAS without consuming
new seals. Resolution is required before same-leader resnapshot after Watch
compaction as well. New leaders are protected by the existing leadership fence.
These transport-error retries retain their deadlines; healthy full-capacity
operation does not poll. Existing dispatch and recovery run independently.

Metrics are low-cardinality: snapshot/list RPC counts, admission outcomes,
errors by phase, invalid inputs by kind, cache table/shard/byte gauges, and
`master_aggregate_wal_capacity`, `master_aggregate_wal_outstanding` and
`master_aggregate_wal_waiting`. Gauges
reset when this loop loses leadership; counters remain cumulative per process.
The additional Watch does not require periodic per-table reads or writes.

Validation must cover the two-writer threshold gap, missing merged-watermark
updates, new seals during a running task, transaction-response ambiguity,
leader loss and reconstruction, consistent paginated snapshots and compaction,
quiet active dedupe, 13,000 tables with twenty shards, and actual process-level
merging and readback. Local unit/integration results do not constitute image or
production qualification, or a claim of reduced fleet etcd traffic.

Regression coverage also includes a 100-table bootstrap bounded to four tasks
with two slots, fresh demand receiving completion-driven credit, member capacity
growth/shrink, queued restart, and completed or delayed ambiguous publication.
