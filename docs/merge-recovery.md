# Owned merge recovery

A serial merge fan-out previously had no request deadline and could report
success after one worker failed if another reclaimed generations. An HTTP
timeout alone cannot hand off a table: the worker or object store may still be
committing after the master advances.

Protocol 2 owns the worker execution independently of its HTTP connection,
records every admitted manifest version, and can fence those versions in storage
before handing off a stuck or crashed executor. Recovery does not depend on a
replacement pod being Ready or on guessing that an old process has died.

## Admission, cancellation, and recovery

`MergeWal`, `Compact`, `IndexId`, and `Repair` share the existing target lock.
Merge admission replaces that lease-backed lock with a persistent execution
marker. A separate leased merge claim permits only one scheduler to reconcile
it, including tasks that bypass dedupe through dependency chains. Helpers that
respect the existing target lock stay excluded throughout recovery.

Before **each** base-manifest commit and **each** shard-drain write attempt, the
worker atomically checks that its exact execution is still Running and records
the resource's maximum admitted immutable version. The permit also binds the
worker's dataset URI. A shard drain guards each conditional write, not a whole
retry loop that could acquire fresh versions after revocation.

Normal cancellation drops preparation and joins already-started leaf manifest
writes before publishing Finished. Worker merge slots remain held during that
join. A definite conditional-write conflict is distinguished from an ambiguous
storage error or panic. An ambiguous outcome publishes Uncertain and retains
ownership, even though the Rust future has ended.

For a missing/stuck executor or an uncertain commit result, the master:

1. Atomically changes the execution to Recovering, closing further commit
   admission. A racing permit either precedes this transaction and is included
   in the durable maximum, or fails its ownership comparison.
2. Reads the closed set of admitted versions and checks the worker/master URI.
3. Uses metadata-only conditional writes to advance the base and affected shard
   manifests beyond those versions. The base barrier updates reserved table
   metadata `lance-context.merge-recovery`. Shard barriers copy the current
   manifest without changing the WAL list or writer epoch. They do not open a
   shard writer, scan payloads, drop fragments, or fence ongoing ingestion.
4. Publishes Recovered only after every barrier is confirmed. Only then can a
   live task release the target lock and continue merging. A delayed old PUT
   targets an occupied immutable version, and any new old-executor commit is
   denied by the closed admission gate.

A surviving executor may abort/join its pending leaf futures after observing a
completed barrier or a subsequent execution. Before that evidence it retains
those futures. This allows a healthy shard to progress without waiting forever
for a stalled old storage request. The master permits one recovery/restart of
fan-out within a task; further faults are durably queued for another task so a
single target cannot monopolize a scheduler slot indefinitely.

Reserved work is cancelled by CAS, so a delayed HTTP POST cannot resurrect it.
The base handle refreshes before retrying, and WAL drain remains a relative edit
that preserves generations flushed during recovery. Graceful worker shutdown
cancels/drains owned actors before closing resident stores.

## Queue, execution, and no-progress limits

Admission and slot acquisition have a separate `MERGE_QUEUE_TIMEOUT_SECS`
(default 600). The execution clock starts only after a merge slot is acquired.
`MERGE_IDLE_TIMEOUT_SECS` (default 600) limits the interval without observed
completed work. Progressing owned merges have no total runtime ceiling.
`MERGE_EXECUTION_TIMEOUT_SECS` remains a positive legacy wire field for rolling
compatibility; older masters/workers can still enforce it until upgraded. Queue
and idle limits must also be positive.

The worker counts completed nonempty batch reads and successful manifest commits,
excluding conditional-write conflicts and failed storage operations. It publishes a changed sequence at most once a second, guarded by
execution ownership. The master tracks the same sequence with bounded RPC
allowance. Repeated heartbeats do not extend the idle limit. Progressing work
may continue beyond the legacy total ceiling. A missing progress record means the
executor has not reported acquiring its slot. Readiness and these intermediate
checkpoints are NOT proof of WAL reclamation; only committed shard drains are.

Progress observation is coarse inside opaque Lance index/build/storage calls.
A single such operation exceeding the idle threshold can still be cancelled.
Measure its p99 on realistic blobs and tune the idle limit before enabling
a large target. These defaults are not production throughput measurements.
Cancellation retains the slot until leaf commits drain or a barrier proves them
fenced. It does not restart the pod. An etcd outage cannot indefinitely occupy a
master reconciliation slot; unresolved ownership remains durable for recovery.

## Count- and time-triggered merge compatibility

`ROLLOUT_MERGE_AFTER_GENERATIONS` and `ROLLOUT_CLEANUP_INTERVAL_SECS` remain
valid with owned merging. For enabled rollout/generic targets, the flush
sweeper checks its own shard's count using manifest metadata only. At the
threshold it writes a coalesced merge request; the cleanup timer uses the same
request path. The master consumes requests every 15 seconds in pages of at most
256 through normal task dedupe, and executes the table's workers serially under
owned commit protection and their existing merge slots/byte budgets.

This preserves #289's pending trigger, but changes execution for opted-in tables:
it is scheduled by the master, rather than performed inline by the flush
sweeper. Queue latency therefore remains relevant; 15 seconds is a polling
interval, not a completion SLA. A later flush tick reasserts demand if the active
task has already passed that shard. Requests never clear failure backoff. A busy
merge slot does not block flushing or require preparing any payload to request
work. Unselected targets retain #289's direct, slot-bounded count merge. Datagen
is not a scheduler-owned target and retains its existing sweeper path.

## Repeated failures

Per-target/endpoint records survive task recreation and master restart. They
contain consecutive failed attempt counts, diagnostic class, last error (capped
at 4096 characters), next retry time, and `needs_attention`. Executed failures
are recorded atomically with releasing terminal execution ownership; a crash
between release and bookkeeping cannot erase the retry budget.

The first three transient failures use two-second targeted retries, with healthy
shards visited before failed shards are retried. After that, each due task makes
one probe, backing off from 30 seconds to at most 15 minutes. At 15 consecutive
failures, probes are hourly and `needs_attention` is set. Missing data,
corruption, and configuration/authentication errors require attention immediately
and use hourly probes. Diagnostic text classification is conservative and never
authorizes data removal.

A failed **storage barrier** retains the execution/target lock and releases the
scheduler slot. Metadata recovery probes back off from 30 seconds to at most
five minutes, with attention flagged immediately. They keep trying to establish
a safe handoff when storage recovers. They cannot authorize a new writer while
the barrier is incomplete. Other tables continue running.

Successful shard merges clear their failure records. Failed/deferred shards keep
the overall task failed even if healthy shards reclaim WAL. Per-table merge
cooldown is replaced by per-endpoint backoff so healthy shards can process new
WAL. A 15-second scheduler poll reads at most 256 failure records per page and
enqueues due targets through existing dedupe. It does not wait for the
600-second stats sweep. Large ledgers add page traversal latency; removed worker
endpoints are not automatically probed.

`needs_attention`, `last_error`, and `next_retry_ms` are persisted in the failure
ledger. `master_merge_storage_recoveries_total{result}` records barrier outcomes
alongside existing task and reclaimed-generation metrics. The read-only failure
inspection HTTP API is a separate follow-up; no external notification integration
is included here.

Merge errors do **not** enqueue destructive fragment repair. Missing/corrupt
files require diagnosis and restoration or an explicitly reviewed repair.
Other maintenance tasks retain their existing repair policy.

## Storage requirements

- Masters/workers use the same physical storage namespace, etcd prefix, and
  storage configuration/credentials. URI mismatch fails closed.
- Version fencing requires Lance's immutable conditional manifest handlers:
  local files, S3, GCS, Azure, memory, OSS, COS, TOS, and shared memory. Unsafe
  fallback handlers and external catalogues such as `s3+ddb` are rejected.
  Preserve manifest immutability and retention; never delete fencing files to
  unblock a live writer.
- Workers with owned targets need `ETCD_ENDPOINTS`. Connection is lazy and
  retried on demand: etcd unavailability does not fail worker startup, flush, or
  ingestion. New owned admission and commit authorization fail closed when etcd
  is unavailable. Already authorized storage writes may still finish.
- Cancellation reconciliation has a 60-second allowance; each metadata barrier
  attempt has a 120-second deadline. These never authorize an unsafe handoff.
- `INDEX_BEFORE_MERGE` remains accepted; owned workers prepare their key indexes
  inside execution. Keep ordinary workers at six merge slots / 20 GiB. Neither
  limit is increased by this patch.

## Default-off, per-table rollout

### Legacy worker progress evidence

Workers advertising `legacy_progress_protocol: 1` expose
`GET /api/v1/internal/legacy-merges?target=<scheduler-target>`. The response
identifies the process incarnation and individual executions, distinguishes
slot waits from running work, and reports actual completed steps, WAL reads,
encoded output, base commits, reclaimed generations, and local outstanding or
uncertain commits. Polling and changing phase do not advance completed work.
HTTP merges and both worker self-merge triggers share this instrumentation;
ingestion flushes are not included. Cancelling the caller retains an
`interrupted` observation. Cancelling an admitted storage future marks its
commit uncertain: a remote write may still finish. This observation-only scope
does not spawn storage tasks or change legacy cancellation and slot lifetimes.

The endpoint reads only process memory, without opening tables or querying
etcd. It returns at most 32 entries, prioritizing active calls, with explicit
truncation and history-eviction counters. Tracking is capped at 256 active and
128 returned/interrupted entries; overflow increments an untracked-admission
counter without rejecting work. Query an exact target when the global response
is truncated. Master captures a bounded snapshot after an ambiguous legacy RPC
failure (2 seconds, 64 KiB input maximum). The task error stores a small typed
summary of at most eight executions; serial fan-out still stops.
Older workers returning 404 simply have unavailable evidence.

This is **process-local evidence, not ownership or permission to take over**.
Even an empty response or zero active commits cannot prove storage quiescence:
other masters/helpers, evicted history, prior incarnations, and old binaries
are outside its coverage. `takeover_authorized` is always false. These snapshots
help identify legacy execution progress; they do not replace the migration
barriers below or implement automatic enrollment of new tables.

`MERGE_OWNED_TARGETS` and `MERGE_DRAIN_TARGETS` are comma-separated **exact**
scheduler targets (`name` for rollout, `generic:name` for generic). Both default
to empty; wildcards and overlapping lists are rejected. Merely configuring
etcd or deploying the binaries does not enable the protocol. Unselected tables
use the legacy serial path, including its inability to safely recover a stuck
untracked write. An owned RPC failure never falls back to that path.

This is an explicit operator-controlled migration, not automatic discovery or
a proof that old writes have drained. All replicas and helpers must follow the
sequence; mixed admission policies for the same table are unsupported.

1. Deploy capability-aware binaries with both lists empty. Keep self-merge
   thresholds/timers and unrelated master jobs running. No worker etcd
   connection is opened merely to advertise capabilities.
2. Select one table. Put it in `MERGE_DRAIN_TARGETS` on all masters and workers;
   prevent legacy helper admission for that table as well. Workers reject new
   legacy merge requests and skip its self-merge; masters reject new maintenance
   mutations for that table while still reconciling any already-owned merge. Other tables, ingestion, and master job pools continue.
3. Verify that previously admitted legacy worker/helper writes have actually
   completed, including ambiguous remote storage operations. Ready pods, lease
   expiry, elapsed time, or a completed HTTP request alone are insufficient.
   An untracked legacy write has no admitted-version watermark and cannot be
   automatically fenced by this protocol. If it is unresolved, keep **that
   table** draining until storage completion is established; do not clear its
   ownership or pretend an empty watermark permits recovery.
4. Move the target from draining to owned on every worker. Inspect
   `GET /api/v1/internal/merge-executor`: check the exact target lists,
   incarnation, progress protocol, and all three timeout values. Keep masters
   draining while workers transition. Worker triggers can queue durable demand
   but cannot start an unowned merge. Once all writers are ready, move the
   target to owned on all masters and update helpers to use owned admission.
5. Soak that table under sustained writes, worker death, delayed storage, and
   etcd disruption. Measure committed generations, memory/OOM, commit conflicts,
   etcd overhead, queue time, longest opaque phase, and p95/p99 task latency.
   Expand one table at a time only after the results justify it.

Rollback also goes through draining: stop new admission, finish or storage-fence
all owned executions with the capable binaries, then return the selected table
to legacy configuration. Do not simply remove an owned target or roll back a
worker while its execution is unresolved. The master explicitly refuses a
legacy path while a durable execution record exists. Do not clear a helper's
pause while its delegate is alive. There is no fleet-wide stop requirement.

## Review and deployment boundary

Timeout, cancellation, commit authorization, storage fencing, and ownership
handoff are one correctness unit. Shipping an HTTP timeout alone would leave
old writes alive and permit conflicting retries. Persistent retry budgets also
remain in the runtime so newly triggered tasks cannot repeatedly reset attempts;
the read-only failure inspection API can be reviewed separately.

The fence covers scheduler-coordinated maintenance and helpers that honor target
locks. It does not retrofit lease-loss protection onto master's local
compaction/index jobs or arbitrary manual writers. It cannot repair missing data
or guarantee a pending ceiling while storage is unavailable. Local fault tests
and CI do not replace the realistic staging soak. This change is not a production
deployment, and its default-off rollout is deliberate.
