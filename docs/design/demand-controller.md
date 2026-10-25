# Leader-owned merge demand discovery

This document describes the discovery layer introduced in #565. The
[capacity-backed task dispatch extension](task-dispatch-controller.md) replaces
the per-executor queue scans mentioned below. With that extension the election
key is `demand-controller/leader-v2`, and the snapshot also includes capacity,
offers and canonical ownership (17 Range/Txn RPCs for the fixed fixture). The
seven-RPC measurements below apply to discovery alone.

Each master previously paged merge requests, execution inventory, and failures
once every 15 seconds. With N masters, discovery and advisory enqueue reads were
repeated N times even when the durable queue already contained the work.

The demand controller replaces that discovery loop. Execution still uses the
existing distributed task claims, concurrency budgets, maintenance rules and
storage fencing. It creates no Kubernetes objects or additional worker pool.

## Ownership and admission

Each scheduler registers its admission policy once under a process lease in
`demand-controller/members/<token>`. Keepalive renews the lease without rewriting
the policy. This registration is independent of the optional shadow planner.
The leader uses the union of these live policies: selecting a narrowly scoped
master must not hide work that another master can handle. Policies preserve the
existing owned/draining/automatic-merge and endpoint rules; they do not grant a
new executor permission to write a table.

Followers watch the exact `demand-controller/leader` key. They neither scan
inventory nor keep attempting the election while the leader exists. When the
key disappears, candidates compare-and-put it using their existing lease.

All discovery enqueue and request acknowledgement transactions compare the
leader's token as well as the source record revision. Stale dedupe cleanup also
compares the token. Losing the lease is not authority to remove a merge owner,
replay a storage write or reset a failure budget. Ordinary executor fencing and
recovery remain responsible for those actions.

## Snapshot and change processing

The leader lists the three inventory prefixes, policy membership, and active
queue/running records in pages
at one etcd revision. It starts no admission from a partial snapshot. It then
watches each range from that revision plus one on a shared stream.

Each watch has its own resume revision. Creation acknowledgements must not
advance a cursor: their header can be newer than events still being replayed.
A connection interruption resumes those cursors. Compacted history or an oversized replay response requires a
new consistent, paged snapshot. The code does not periodically relist a healthy watch.

One indexed timer per source key coalesces repeated updates. Updates replace the
observation without adding duplicate queue entries or postponing older work.
Queue/running records maintain an in-memory dedupe index. Already-active
recovery entries need no advisory etcd reads. Dependency-chain tasks do not
suppress independent recovery. Only demand records have timers; task and policy
records update their indexes on events. Policies are decoded once per update.
The 15-second reconciliation timer handles time-based failure eligibility and
requests retained during a running task; it reads the local cache, not every
etcd prefix. A reconcile has a bounded duration and leaves durable demand intact
on error. This timeout does not cancel an executing merge or judge its progress.

The cache has a 100,000-key and 64-MiB encoded key/value limit, plus bounded index
and decoded-record overhead. Capacity failure is an explicit error, never silent
truncation. Deployment must verify inventory fits these limits. It must also
verify cold-target admission latency under representative failure inventory.

## Validation and rollout

The isolated-etcd tests cover List/Watch setup gaps, reconnect replay, compaction,
policy union, stale-leader publication rejection, real process-task loss and
lease expiry. A fixed 256-request fixture checks 1, 10 and 50 participants. Its
measured discovery snapshot is seven Range/Txn RPCs (including active-task state) in each case, with no follower
discovery reads while the leader is healthy; the 50-participant quiet window
exceeds the old 15-second poll interval. Registration and failover election
traffic are explicitly outside that steady-state measurement. This is not a
measurement of total production RPCs or WAL throughput. A separate 64-active-
execution fixture verifies zero Range/Txn RPCs for repeated recovery reconciliation
and proves task completion removes suppression through Watch.

Mixed versions remain safe because the existing admission dedupe/revision
checks still apply, but old replicas continue their legacy scans. The full
reduction requires replacing those replicas. Validate ordinary writes, productive
merges, failure budgets, leader loss and accepted demand in staging before
rolling through production. Inventory discovery is only one source of etcd load:
the shared task-claim scans and their broad wakeup stream are unchanged by this
change, as are stats sweeps and execution progress reporting.

## Metrics

All labels remain bounded; none contain a table, endpoint or execution ID.

- `master_demand_controller_leader`: sum over live replicas should be one;
  short zero intervals during takeover are expected.
- `master_demand_controller_cache_keys` and `_cache_bytes`: max over live replicas;
  a former leader resets its gauges on exit.
- `master_demand_controller_list_rpcs_total` and `_snapshots_total`: fleet summed
  counter rates reveal unexpected relists.
- `master_demand_controller_watch_reconnects_total`: fleet summed rate.
- `master_demand_controller_reconcile_errors_total`: fleet summed rate;
  correlate with pending and admission latency, never treat errors as zero demand.

Compare these with total etcd Range/Txn rates, zero-WAL completions, positive WAL
commits, pending distribution, read refusals and read latency in the same window.
