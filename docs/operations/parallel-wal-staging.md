# Concurrent native WAL staging

`CATCHUP_STAGING_TARGETS=table_a,table_b` opts exact tables into concurrent
immutable WAL reads and file encoding inside their existing master executor.
The default empty list preserves the current pipeline. No worker fan-out,
independent merge pool, or extra target owner is introduced.

`CATCHUP_STAGING_CONCURRENCY` sets the maximum number of admitted staging
operations per wave (default 2; supported range 1–8). The existing process
merge memory budget and per-pass byte/generation limits still apply. Capacity
for additional readers is acquired without waiting before any payload reader
opens. If only one batch fits, processing continues serially. A generation
larger than its initial reservation may cause the speculative wave to be
joined and discarded, then retried serially; an oversized generation can grow
beyond the budget only as the sole holder, as in the existing merge path.
Decoded reservations do not include every scanner/encoder/upload allocation,
so actual RSS and OOM behavior must be measured before increasing concurrency.

All encoders in a wave finish before its commits begin. Commits run one at a
time in shard order. Encoded files do not publish data or coverage. Deletion
masks are prepared against the latest base at commit time, so later shards
see preceding upserts. Schema/field-ID changes invalidate staged files.
Decoded batches stay reserved through publication, preserving the key-only
target-delete behavior for missing indexes and ZoneMaps. Definitive commit
conflicts may reuse immutable files; unknown outcomes do not trigger replay.

Read/encode failure joins all admitted encoders before returning, and closes
admission to later waves. Successfully prepared predecessors may still commit
in order; nothing after the failed shard is published. Commit failure discards uncommitted successors;
there are no speculative commits. Payload preparation runs in children bound
to the durable maintenance scope. If the watchdog cancels the caller, scope
drain still waits for those children and retains their memory reservations.
A version fence may abort manifest leaves, but does not abort immutable
encoders or make their memory available early. Observation-only legacy
execution cannot enable this path. Discarded files
may remain unreferenced, as with other uncommitted Lance writes.

This path trades inter-wave overlap for a simple bounded admission boundary.
It is an experiment, not a claim of higher production throughput. Compare
end-to-end reclaimed bytes/generations, commit counts, RSS, object-store
requests/retries, and read latency on the same representative input. Keep
batch size, memory budget, CPU, and index coverage fixed; do not sum overlapping
phase timers or compare unrelated tables as an A/B test. The existing
`CATCHUP_PARALLEL_UPSERT_IO_TARGETS` option only overlaps one batch's deletion
and encoding; staging instead overlaps independent shards' read/encode work.

Lance 13 can drop remaining CPU encoder handles on an internal write/encoding
error. The scoped CPU task join patch in `vendor/lance-core` tracks the actual
blocking closures, including nested submissions. A separate reservation hold
retains the writer's final memory charge even when the writer returns an error
or panics before its children finish. Scope drain waits for that cleanup before
acknowledging completion; successful prepared output keeps its own reservation.
This does not turn the decoded-byte budget into a hard RSS bound.

The same group propagates through Lance's fragment scanner and batch decoder
children, including existing abort-on-drop wrappers. Shared blob sources keep
one token per logical request, and physical schedulers keep tokens until the
actual request state and response buffers are destroyed. Upload children and
multipart-abort cleanup are scoped; local blocking IO stays charged until its
actual closure exits. Cancellation before a child's first poll destroys its
captured payload before releasing membership. V1/V2.0 tables keep encoding in
the existing ordered commit path.

Qualification is not complete. Keep this option disabled until all targeted
failure/cancellation checks, CI, and the fixed-image Azure comparison pass.
Tests and task tracking do not establish a hard RSS bound or account for all
shared transport/cache allocations. All performance claims require the
representative comparison above.
