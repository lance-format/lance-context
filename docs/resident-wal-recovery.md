# Autonomous resident WAL demand

Resident masters already execute bounded append-only merges using the ordinary
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

With `ROLLOUT_APPEND_LOCAL=true`, `MERGE_WAL_INTERVAL_SECS>0`, and
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
