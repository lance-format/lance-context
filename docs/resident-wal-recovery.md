# Autonomous resident WAL demand

Resident masters execute bounded WAL merges using the ordinary
durable queue, shared process memory budget, progress watchdog and publication
fences. Two scheduling gaps could still require an external enqueue loop:

* A successful slice stops after 16 passes or its configured service duration,
  even when another slice could reclaim more WAL.
* Discovery through the full stats scan can lag behind newly sealed generations.

A resident slice that exhausts its service budget now writes a durable merge
request before returning. The request consumer only acknowledges demand when a
task is queued, never when it finds the current task still running. A replacement
master can consume this demand. The successor yields through the ordinary queue
and existing maintenance admission; a hot table does not retain its claim
indefinitely. If the last pass drained everything, at most one empty successor
is needed; a zero-work pass does not perpetuate itself.

With `ROLLOUT_APPEND_LOCAL=true` or a nonempty `MASTER_LOCAL_MERGE_TARGETS`,
`MERGE_WAL_INTERVAL_SECS>0`, and
`ROLLOUT_APPEND_RECONCILE_INTERVAL_SECS>0` (default 30), masters also discover WAL
demand separately from the stats scanner. One fleet-wide etcd compare-and-swap
reserves four targets per batch, with two metadata readers, a ten-second limit
per target, and at least 30 seconds between batch reservations. Each reader
loads the base manifest and latest manifests for up to 256 configured
`CATCHUP_SHARDS`; it does not read generation payloads, open shard writers,
replay raw WAL, or modify table/WAL manifests. Local staging's small shared
session bounds metadata caches. Missing or slow tables consume their bounded
visit and do not prevent rotation to later names. A reserved batch lost to a
crash is revisited on the next rotation.

Explicit owned/append allowlists supply names even when stats are absent or
stale. Wildcard configurations additionally use known stats-cache names; this
does not replace registry discovery. Different rollout configurations have
separate durable cursors and share the fleet-wide read budget. Configured
historical shards participate; unconfigured historical shards still depend on
ordinary stats discovery or a merge pass's shard enumeration.

Active canonical tasks, external catch-up owners, draining/unowned tables and
unexpired catch-up failure backoff are skipped. Repeated failures keep their
original attempt counters and retry deadlines. Demand only schedules an
ordinary task: it never removes an execution record, bypasses a target lock,
changes ownership allowlists, or authorizes a competing publisher.

Metrics: `master_resident_merge_continuations_total` counts durable continuation
requests; `master_resident_wal_probes_total{result="requested|empty|failed"}`
describes discovery outcomes. Neither counts committed generations; use actual
merge completion metrics and pending observations for throughput.

This removes external refill for already-owned resident tables. It does not
certify cancellation of legacy unfenced worker handlers, fix serial worker
flush scheduling, or replay an unsealed historical raw WAL tail. Those cases
must retain their existing recovery protection until separately validated.

Resident masters also reserve `ROLLOUT_APPEND_LOCAL_TASK_CONCURRENCY` task slots
(default 1, 0 disables the reservation) in addition to `MERGE_WAL_CONCURRENCY`.
Only locally enabled, owned, non-draining merges can use these slots.
Eligibility is checked before dependency reads or claim/lock mutation. The
ordinary merge pool remains available to all merge tasks, including legacy
worker RPCs; a legacy request cannot consume the reserved capacity. Local
Arrow buffers still share `ROLLOUT_APPEND_LOCAL_MEMORY_BYTES` across both pools.
Canonical claims, external catch-up ownership, progress watchdogs, and process
admission drain apply equally to the reserved pool. This does not cancel an
existing legacy owner of the same table.

## Generic and ordinary rollout tables

`MASTER_LOCAL_MERGE_TARGETS=table,generic:rows` executes the selected tables in
native master processes. `*` selects the explicitly owned, non-draining targets
in `MERGE_OWNED_TARGETS`; it does not enroll legacy tables into ownership.
Existing `ROLLOUT_APPEND_LOCAL` selection continues to select rollouts only.

Append-opted-in tables keep parallel append file preparation. Other rollouts
and generic tables use the existing bounded, ordered catch-up implementation,
including upserts and key-index maintenance. All local work shares the same
`ROLLOUT_APPEND_LOCAL_MEMORY_BYTES` budget and small metadata session. The
ordinary path uses `CATCHUP_MERGE_MAX_BYTES`, `CATCHUP_MERGE_MAX_GENERATIONS`,
`CATCHUP_PIPELINE_ENABLED`, and `CATCHUP_SLICE_SECS`. Each batch must fit the
configured shared budget; an indivisible generation can exceed the batch byte
cap, subject to the core reservation rules. This budget bounds merge Arrow
reservations, not total process RSS or all encoding/index scratch allocations.

Both paths publish continuation demand after a bounded slice. Metadata discovery
supports generic schemas without opening ingestion writers. Stable shard names
must include every writer to be serviced. No Kubernetes Pod creation or pool
inventory permission is required. To add execution capacity, scale the master
replicas; canonical claims distribute different tables, while the same table
retains one fenced publisher. Every replica has its own bounded buffer budget.

Activation still requires ownership migration: drain the target's old unfenced
maintenance writers, prove their admitted storage writes finished, then enable
owned maintenance consistently across writers, masters and helpers. Keep
existing consumers running for other tables until their own handoff is complete.
This flag neither imports private recovery ledgers nor turns unknown old
executions into safe takeover candidates.
