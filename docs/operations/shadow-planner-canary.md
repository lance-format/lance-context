# Shadow planner canary

How to run the P1 shadow planner on one production master, what to watch,
and when to turn it off. The planner **observes and simulates**; it places no
tasks and removes no existing loop. Turning it on does not change merge
throughput. Design: `docs/design/unified-maintenance-scheduler.md`; data
model: `docs/design/scheduler-p1-data-model.md`; tracking: #333.

## What it does when enabled

On the master where `PLANNER_ENABLED=1`:

* publishes a leased `P/executors/<id>` heartbeat;
* campaigns for `P/leader`; if it wins, every `PLANNER_RECONCILE_SECS`
  (default 30) it reads every `demand-events/` record in byte‑bounded pages
  at one revision, folds them into `P/demand/<hex>`, scores merge demand,
  writes `Shadow` assignments against executor headroom, and compares itself
  with the real task queue;
* serves `GET /api/v1/scheduler/demand` and `GET /api/v1/scheduler/executors`.

Writers and scanners already publish demand events regardless of this flag
(#345, #351, #356, #357); the canary adds the consumer.

## Enable

1. Confirm every master in the fleet runs a build at or after #367. Older
   builds ignore the new key families, but the canary's comparison reads the
   task queue, which every build writes.
2. On **one** master set `PLANNER_ENABLED=1`. Leave `PLANNER_RECONCILE_SECS`
   at 30. Nothing else changes.
3. Within one lease TTL `planner_is_leader` should be 1 on that master and
   `GET /api/v1/scheduler/demand?limit=20` should return the top pending
   tables with `total_tables` near the registry size.

## Watch, in this order

| Signal | Healthy | Act if |
|---|---|---|
| Master RSS on the canary | flat after the first full pass | grows pass over pass — the demand cache is ~13k small records and should not; disable and report |
| etcd p99 latency, apply duration, DB size | no change from before | rises on the planner's cadence — the paged reads are too aggressive for this cluster; disable |
| `planner_reconcile_seconds` | completes every tick, seconds not minutes | climbs or `planner_reconcile_skipped_total` increments steadily — passes are overlapping |
| `planner_reconcile_errors_total`, `planner_shadow_errors_total` | zero or rare | sustained — read the warn logs; usually leadership lost during a write |
| `planner_is_leader` | exactly one master at 1 | flaps — lease keepalive is losing to something; disable |
| `planner_page_shrink_total` | occasional | continuous — records are larger than expected; fine, but note the shard distribution |
| Real merge progress (`master_merge_wal_generations_reclaimed_total`, `master_rollout_append_generations_reclaimed_total`) | unchanged | drops — the planner must not be the cause (it executes nothing); look elsewhere, but disable to rule it out |
| Pending WAL (`master_wal_backlog_generations`, `master_wal_backlog_largest_shard_generations`, with `master_wal_backlog_snapshot_available == 1`) | unchanged | worsens — same as above; unknown/missing snapshots do not prove health ([queries](../datadog-wal-backlog.md)) |
| Actual HTTP failures and read latency | unchanged | rises — investigate response counters and latency separately; backlog above a reference threshold is not a count of 503s |

## What the comparison means

`planner_shadow_comparison{outcome}` is per due table per pass. Each value
answers one question; none is a pass/fail gate.

| outcome | question | what a high value says |
|---|---|---|
| `discovery_miss` | due demand, reality has nothing queued or running | the real discovery loops are slower than the demand feed, or the demand feed is wrong — check a few tables by hand against `pending_wal_generations` |
| `queued_only` | due, queued, not running | admission or capacity is the bottleneck, not discovery |
| `running` | due and executing | agreement |
| `placement_gap` | due but the planner found no headroom | expected while legacy workers and catch‑up Jobs do not heartbeat as executors; **not** a parallelism measurement (reservations are deliberately pessimistic) |
| `phantom_task` | reality runs a task on a table with no scored demand | a lost demand event, or a tail the planner ignores — check the table's `demand/` record |

Record the first day's values before judging anything. The useful readings
after a few days: `discovery_miss` trending down toward the tail population,
`phantom_task` near zero, and `placement_gap` explainable by executors that do
not heartbeat yet.

## What this canary does not tell you

* Whether the planner could **schedule** merges correctly — it does not
  schedule.
* Real parallel capacity — every shadow reservation is the whole local budget
  because per‑generation sizes are not yet reported.
* Anything about compaction or index scheduling — only merge demand is scored.

## Disable

Unset `PLANNER_ENABLED` on the canary and restart it. Its lease expires
within `ETCD_LEASE_TTL_SECS`; the `leader`, `leader-token` and
`executors/<id>` keys disappear with it. `demand/`, `demand-events/` and
`assignments/` remain and are inert; nothing reads them. They can be left or
deleted under the etcd prefix with no effect on running masters.

## Widen

After the canary has run cleanly through at least one stats‑scan cycle on
every table (`STATS_SCAN_INTERVAL_SECS`, default 300 s, so hours in practice
for 13k tables) and the signals above are flat: enable on a second master.
Exactly one will lead; the other heartbeats as an executor. No further change
to behaviour. Do not proceed to per‑table `mode: planner` (P2) from canary
data alone; that step needs the planner to see legacy workers and catch‑up
Jobs as executors, per‑generation sizes, and a policy document per table,
none of which exist yet.
