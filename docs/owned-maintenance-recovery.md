# Recovering stalled local table maintenance

Index, compaction and repair tasks can hold the same table lock needed by WAL
merging. On explicitly owned merge targets, these local mutations now reserve a
durable execution using the existing version-fencing protocol. Legacy targets
retain their previous execution path. Worker merge fan-out remains serial;
compaction concurrency and all merge memory/byte budgets are unchanged.

## Cancellation and takeover

The default total deadline is 3600 seconds (`MAINTENANCE_TIMEOUT_SECS`), with a
600-second no-progress deadline (`MAINTENANCE_IDLE_TIMEOUT_SECS`) and 30-second
commit drain grace (`MAINTENANCE_DRAIN_TIMEOUT_SECS`). All must be positive.
Progress counts completed scoped processing steps or manifest commits, never
heartbeats. Lance index building and individual compaction rewrites do not
expose fine-grained scan progress here: a healthy long operation can reach the
idle deadline. Size that deadline above measured large-table operation times;
it is an execution bound, not proof of an internal deadlock.

Cancellation drops the local work future and closes commit admission. If every
manifest commit has a definite result, the task can complete and release its
lock. Ambiguous or still-running commits require durable admission revocation
and metadata-only conditional manifest writes beyond every admitted version.
Only a successful barrier permits another writer to take over. Newly opened
maintenance handles retain their execution guard even when passed to a child
task. Cached worker handles continue using the current execution dynamically.

If storage fencing fails, keep the durable table ownership, return the scheduler
slot, and retry recovery with persistent backoff. A small supervisor retains any
unfinished manifest leaves until they drain or a later reconciler proves the
storage fence. Other tables keep running. A master crash or an uncertain
admission response is recoverable from the durable execution inventory, scanned
in pages of 256 every 15 seconds independently of stats sweeps.

## Repeated failures and diagnostics

Failures use the existing `/merge-failures` API and durable per-target ledger,
with endpoints `master:compact`, `master:index_id`, and `master:repair`. A new
task or process does not reset the attempt count. A repair that actually commits
a changed manifest clears the corresponding missing-file failures so dependent
work can resume; no-op repairs and unrelated failures keep their budget.
Retryable failures back off;
persistent failures get `needs_attention` and hourly probes. An unresolved
storage barrier retains ownership and probes at up to five-minute intervals.
An unavailable dataset, bad configuration or repeatable corruption requires
repair of that cause; retrying cannot manufacture a successful storage fence.

## Activation and compatibility

This uses the existing owned-target configuration, not a new rollout switch.
Deploy compatible masters before activating more owned targets. Include every
auxiliary maintenance master in that inventory. Old readers cannot CAS records
containing the maintenance discriminator, so they fail closed rather than
acting as a compatible recovery controller. Worker execution admission rejects
local maintenance records. Existing worker execution JSON remains unchanged.

The existing protocol transition still requires legacy writes to be drained:
this change cannot retroactively fence an old unguarded write. It does not enable
owned targets, alter production pods, or complete registry migration. Recovery
continues on draining targets while an owned execution remains. Validate large
table timings and fault recovery before selecting tighter deadlines.

## Continuous WAL catch-up

A successful owned WAL execution that reclaims generations now writes a coalesced
merge request in the same etcd transaction that releases its execution. The
master's 15-second demand poll retains that request while the current task runs,
then queues another pass. Each pass returns to the normal task queue, preserving
serial worker fan-out and giving other tables and maintenance a scheduling
opportunity. A final zero-progress pass stops the continuation. Failed executions
still use their durable retry budgets; successful work on another shard cannot
reset those budgets. This also drains historical batches when no new writes or
stats sweeps arrive.

Workers use `OWNED_MERGE_AFTER_GENERATIONS` (0 disables) to request work
for explicitly selected owned targets from the existing flush timer. The check
reads only this worker's shard manifest, without loading WAL payloads or taking a
merge slot. Unset, it inherits a positive `ROLLOUT_MERGE_AFTER_GENERATIONS` or
uses 32 when legacy count-triggered merging is disabled. An explicit threshold
is independent of `ROLLOUT_MERGE_AFTER_GENERATIONS`,
which can remain 0 for legacy targets during a per-table rollout. The flush timer
must be enabled. Draining targets request no new work. Count thresholds bound
triggering per shard, not total table pending or merge memory.

Deployment is not activation: all regular and auxiliary masters must support
owned maintenance before selecting a table. Keep legacy writers excluded until
old work is drained or its admitted storage writes have been fenced. Transfer a
dedicated executor's claim only at a verified terminal boundary, then remove the
drain selector and select the table for owned execution on masters and workers.
Never remove a live execution record just because its request timed out.

Validate both backlog-only and sustained-write workloads with small byte-bounded
batches, plus worker/master interruption and repeated shard failures. Check that
ordinary services resume without a helper, another table gets service between
passes, empty tables stop producing continuations, and memory remains bounded.
