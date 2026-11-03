# Unified scheduler — P1 data model

Status: proposal for the first implementation phase. Parent design:
`unified-maintenance-scheduler.md`; tracking: #333. This document fixes the
etcd records P1 introduces so that the shadow planner, executor heartbeats and
later phases build on one schema. Nothing here changes scheduling behaviour;
P1 only writes and reads these records beside the existing loops.

`P` = `ETCD_PREFIX`. `<hex>` = hex of target bytes. All values are JSON.
Every record carries `v: 1`; readers reject unknown major versions.

## 1. Demand: per-shard watermarks, never deltas

WAL generations are already numbered per shard as `u64` (`FlushedGeneration`
in `store_base.rs`; merged watermarks in `rollout_append::watermarks`). Demand
reuses that numbering so it is idempotent by construction.

### 1.1 Event record — written by writers and executors

```
P/demand-events/<hex>/<shard-uuid>
{
  "v": 1,
  "sealed_through": 1742,        // highest generation sealed (flushed) on this shard
  "sealed_bytes_through": 913_000_000,
  "merged_through": 1690,        // highest generation merged into base, if the writer knows it
  "flushed_at_ms": 1791480000000,
  "writer_epoch": 7,             // the shard writer's epoch; a lower epoch never overwrites a higher one
  "source": "writer" | "executor" | "scan"
}
```

Rules:
* **Generation numbers are monotonic per shard across writer epochs.** Lance
  keeps `current_generation` and `flushed_generations` when a new writer claims
  a shard (`ShardManifest { writer_epoch: next, ..base }`). The epoch is a
  writer fence and a tiebreak, never a numbering restart. A `merged_through`
  reported under epoch 1 therefore still counts against a `sealed_through`
  flushed under epoch 2 (merged 40, new writer flushes to 60 ⇒ 20 pending).
* `put` only; no deletes by writers. The stored record is the **per‑dimension
  join** of every event published so far: `sealed` and `merged` are each the
  max of `(value, epoch)` across stored and new, taken independently. A writer
  event with `merged_through: None` keeps the executor's merged mark; an
  executor event with an older `sealed_through` still advances `merged`.
  Nothing about one dimension ever lowers or replaces the other. The join is
  idempotent and commutative, so loss, duplication and reordering cannot
  corrupt the count, and a rebuild from stored records equals the in‑memory
  fold.
* Every publisher does read‑join‑CAS with a bounded retry on a lost race, so
  no publisher's progress is lost to another's. Executors publish **after**
  release, one shard per put, detached and bounded; the release transaction
  carries no demand keys (no cross‑key contention with writer flushes, no etcd
  txn‑size limit for many‑shard tables, no storage read on the release path).
* Source is irrelevant to ordering: a lower value from any source simply loses
  the max. A stale scan cannot lower anything. `merged_epoch` records the
  epoch a merged mark was reported under when it differs from the record's
  `writer_epoch`; provenance and tiebreak only.
* `oldest_pending_ms` is the flush time of the lowest unmerged generation. The
  stored record carries `sealed_times`: the lowest `SEALED_TIMES_CAP − 1 = 63`
  recorded generations plus the newest, min time per generation, bounded by
  cap **only** (pruning by `merged` would make the kept set depend on arrival
  order). Entries at or below `merged` are ignored on read. The age clock
  therefore **survives a rebuild from etcd** and a newer flush never resets
  it. The answer carries a bound: **Exact** when the oldest pending
  generation's own time is kept; **Upper** (the lowest kept time above it,
  at worst the newest sealed) once merging has consumed every kept entry on a
  shard that had more than the cap pending, or a long‑lived table whose
  early timestamps occupy the map. An Upper bound on the flush time is a
  **floor on the wait** (the generation flushed no later than that), so
  promotion for age uses the floor from either bound and is sound; score
  aging uses the exact time only, so a bound cannot inflate a score. Counts
  and bytes are exact in every order.
* Records carry `v`. A reader rejects an event it does not understand
  (`Ignored::UnknownVersion`) and a publisher **refuses to overwrite a stored
  record from a newer schema**; `v == 0` is read as version 1.

### 1.2 Table record — maintained by the planner

```
P/demand/<hex>
{
  "v": 1,
  "pending_generations": 52,       // Σ_shards max(0, sealed_through − merged_through)
  "pending_bytes": 221_000_000,
  "oldest_pending_ms": 1791470000000,
  "shards": { "<uuid>": { "sealed": 1742, "merged": 1690, "epoch": 7 }, ... },
  "fragment_count": 118, "small_fragment_bytes": 0,   // from scan
  "index_stale": true, "uncovered_fragments": 9,       // from scan / release
  "missing_fragments": false,                          // from repair / failure code
  "observed_revision": 48213,      // etcd revision of the newest folded input
  "updated_ms": 1791480005000
}
```

The planner folds `demand-events/` into this record; it is a cache, not a
source of truth, and may be rebuilt from events + scan at any time. Folding is
a pure function `fold(table, event) -> table` and is tested for commutativity
and idempotency (design invariant 5a).

### 1.2a Reads at scale

Production has ~13,400 tables and ~27,000 shard records; the shard count per
table is not yet measured and records with a full `sealed_times` map are
~1.5 KiB. A single prefix read of `demand-events/` exceeds the etcd client's
4 MiB message cap, and so does a fixed 512‑key page of large records. Every
planner prefix read (`demand-events/`, `demand/`, `executors/`,
`assignments/`, `queue/`, `running/`) is therefore paged **by bytes**: start
at 512 keys, halve the page on a response over 1 MiB or on a message‑size
error, grow back when pages are small, all **at one revision** so each pass
sees a consistent snapshot. Writes are chunked by **both** op count (etcd
`--max-txn-ops`, 100 per txn) and request bytes (`--max-request-bytes`,
768 KiB per txn). The reconcile runs detached from the leader loop so a slow
pass can never starve the lease keepalive; at most one pass is in flight and
skipped ticks are counted. Tested at 512 × 8 shards with full time maps and
at 100 × 20 shards (both over the respective caps when unbounded).

### 1.3 What writes what in P1

| Source | Writes | Change from today |
|---|---|---|
| Rollout server flush (`sweeper.rs::route_merge`) | `demand-events/<hex>/<shard>` **in addition to** `merge-requests/<hex>` | additive |
| Master append merge release (`rollout_append.rs`, `merge_execution.rs`) | `merged_through` per shard in the release txn | additive |
| Stats scanner (`scanner.rs`) | `demand-events` with `source: scan` for every shard it observes, carrying **both** the shard manifest's `sealed_through` and the base table's merged watermark for that shard, so a lost completion event is corrected on the next scan and an old table's first contact does not read generation 1001 as 1001 pending | additive; existing enqueue/retire paths untouched |
| Planner (new, leader only) | `demand/<hex>` | new |

No existing key is removed or re-purposed in P1.


### Completed advisory records

The active publication path retains `demand-events` only while there is
unmerged demand. Writer observations carry the conservative completed prefix
from the same durable shard manifest as the sealed watermark (the prefix is
before the oldest remaining SST, not highest generation minus SST count).
Executor observations use the full manifest seal and completed prefix and
share the scanner's batches of at most 32 exact-key reads.

After joining an observation with the current record, a known completed prefix
covering every known seal permits a revision-compared deletion of that one
advisory record. A concurrent seal rejects the delete and is re-read and
re-joined. An absent, fully completed record is not recreated. Unknown schemas
and unreadable records are not retired. These deletes never touch tasks,
execution ownership, claims, commit permits, or WAL payloads.

This changes persistence of *completed hints*, not the monotonic watermarks in
storage. A newly active shard publishes its own completed prefix along with its
new seal, so a leader rebuilding after retirement sees the new pending suffix.
A legacy publisher can recreate an overestimated hint during a mixed rollout;
a subsequent manifest observation reconciles it. Absence of a hint never
proves execution termination and cannot authorize takeover or payload deletion.
The old publication API remains monotonic for existing consumers.

Rollout validation must cover partial/non-prefix merges, an old publisher
racing retirement, a new writer after retirement, and snapshot/Watch recovery.
The writer acknowledgement cache includes completed-prefix changes and only
acknowledges successful publication; restart or eviction replays the observation.
`etcd_operation_attempts_total{operation="demand_retire"}` counts retirement CAS
calls (not guaranteed deletions); failed comparisons are successful RPCs.

## 2. Executors and reservations

### 2.1 Executor heartbeat — leased (`ETCD_LEASE_TTL_SECS`)

```
P/executors/<executor-id>
{
  "v": 1,
  "kind": "resident_master" | "worker" | "catchup_job" | "legacy_rpc",
  "instance": "<pod>/<uid>", "version": "0.6.4",
  "kinds": ["merge_wal", "compact", "index_id", "repair"],
  "slots_total": { "merge_wal": 2, "compact": 1, "index_id": 2, "repair": 2 },
  "bytes_total": 2147483648,
  "bytes_free_sample": 1900000000,   // diagnostic only; never used for placement
  "target_filter": { "owned": [...], "draining": [...], "append_local": [...] },
  "draining": false,
  "heartbeat_ms": 1791480005000
}
```

`executor-id` is stable per process (`admission.rs` already has
`executor_id`). Catch-up Jobs heartbeat for their lifetime and disappear with
their lease.

### 2.2 Assignment — the durable reservation (unleased)

```
P/assignments/<hex>/<unit-id>
{
  "v": 1,
  "unit": { "kind": "merge_wal", "target": "exp-3", "needs_write_turn": true, "preparing": false },
  "executor": "<executor-id>",
  "planner_token": "<leader token>",
  "reserved_slots": { "merge_wal": 1 },
  "reserved_bytes": 268435456,
  "class": 3, "score": 6.5, "reason": "merge_score",
  "created_ms": ..., "bind_deadline_ms": ...,
  "state": "assigned" | "bound" | "released",
  "execution_id": null | "<merge-execution id>"   // set at bind
}
```

Rules (review point 4):
* The planner's view of an executor's headroom is
  `bytes_total − Σ reserved_bytes` and `slots_total − Σ reserved_slots` over
  assignments in state `assigned` or `bound` on that executor — **never**
  `bytes_free_sample`. Unbound assignments count.
* A new leader rebuilds the reservation table from `assignments/` before its
  first cycle; nothing lives only in the old leader's memory.
* An assignment not bound by `bind_deadline_ms` is moved to `released` by the
  planner and its reservation returned; this is a backoff event, not a failure.
* `released` is written in the executor's merge-crate `release` txn (same txn
  as `merged_through`, §1.1), so reservation, ownership and demand change
  together. The planner garbage-collects `released` records after 1 h.
* Executors still enforce `MergeMemoryBudget` locally. The reservation
  prevents over-dispatch; the local budget prevents OOM when an estimate is
  wrong.

### 2.3 P1 scope for executors

P1 writes heartbeats from every master (all are executors regardless of
leadership). The planner scores `MergeWal` demand only (design §4.2: classes
Critical / Normal / Tail; class 1 oldest‑first), places against headroom, and
writes assignments **only in shadow mode**: state `shadow`, replaced wholesale
every pass, never read by any executor (a source‑level test enforces this), no
binding. Tails are not placed, matching the real sweeps.

Each pass then compares itself with the live system. **These are
observations, not a takeover gate.** The shadow planner sees demand and
capacity; the real queue also reflects ownership, eligibility, cooldowns,
and executors the planner cannot see (legacy workers, catch‑up Jobs). The
comparison therefore answers three precise questions, each its own gauge
`planner_shadow_comparison{outcome}`:

| outcome | meaning |
|---|---|
| `discovery_miss` | due (non‑tail) demand, and reality has the table **neither queued nor running** — the real discovery loops have not noticed it |
| `queued_only` | due demand that reality has queued but not running — discovered, waiting on admission or capacity |
| `running` | due demand that reality is executing |
| `placement_gap` | due demand the planner could not place for lack of headroom, **regardless of reality** — the planner's capacity model vs. the real pools |
| `phantom_task` | reality has the table queued or running with no scored demand — a lost demand event, or work the planner would not schedule |

Comparison is **per due table**, not per placement, so a table the planner
wanted but could not place is still compared on discovery. Reality is read
once per pass as two separate paged sets (queued, running); if that read
fails the pass records nothing rather than zeros
(`planner_shadow_comparison_skipped_total`).
`scheduler_shadow_disagreements_total{kind}` is `discovery_miss +
phantom_task`, kept for continuity; read the components. `discovery_miss`
falling to zero says the demand feed is complete; `phantom_task` at zero
says no demand events are lost; `placement_gap` is expected to stay
non‑zero until legacy workers and catch‑up Jobs heartbeat as executors.

Byte estimates: `ROLLOUT_APPEND_MAX_BYTES` is a per‑pass *target*, not a
ceiling — the append path reads a whole generation before checking it, so the
largest single generation sets the real floor and buffers cost ~2× decoded.
The planner does not know per‑generation sizes: a scan reports
bytes‑through‑sealed, from which only an **average** per generation follows,
and an average is not a maximum. Until per‑generation sizes are reported
(a later change to the scan), every shadow reservation is the executor's
**whole local budget**, which is the only honest answer for an unknown
maximum. That makes the shadow capacity model deliberately pessimistic —
it answers "could this executor safely take one unit" and **cannot** answer
"how many units in parallel"; do not read `placement_gap` as a parallelism
measurement. `planner_shadow_reservation_basis_total{basis}` and
`planner_shadow_average_generation_bytes` record what information each
reservation was based on so the gap to a size‑aware model is visible.

## 3. Leader

```
P/leader         (leased)  { "v": 1, "token": "<uuid>", "instance": "...", "since_ms": ... }
P/leader-token   (leased)  "<uuid>"      // bare token so writes can compare on it
```

Campaign with `Version(P/leader) == 0`, writing both keys under one lease in
one txn. Every planner write txn compares `P/leader-token == own token`, and
every reconcile reads it first even when there is nothing to write, so a
deposed leader notices on its next tick, not on its next change. Keepalive
every `TTL/3`; a failed or expired keepalive ends leadership and the loop
re-campaigns after a short pause. `PLANNER_ENABLED` (default off) gates all of
this; `PLANNER_RECONCILE_SECS` (default 30) sets the full-fold cadence. Followers watch `demand-events/`, `demand/`,
`executors/`, `assignments/`, `merge-executions/`, `target-locks/` to keep a
warm cache. The leader's cycle is triggered by those watches plus a 1 s tick.

## 4. Policy (read-only in P1)

```
P/policies/_default  and  P/policies/<hex>
{ "v": 1, "mode": "legacy", ... }     // see design §4.8
```

P1 ships the schema and `GET /api/v1/policies/{target}` returning the
effective merged policy. Nothing consults `mode` yet; it defaults to `legacy`
for every table. The per-table switch to `planner` is P2 and requires every
master in the fleet to run a policy-aware binary (review point 5).

## 5. Observability shipped with P1

* `GET /api/v1/scheduler/demand[?target=]` — `demand/` record, computed
  scores and class, and the planner's current verdict for the table
  (`would_place {executor, reason}` or `would_skip {reason}`), from the
  shadow cycle.
* `GET /api/v1/scheduler/executors` — heartbeats with computed headroom from
  reservations.
* Metrics: `scheduler_demand_pending_generations{target}` (top‑N only),
  `scheduler_shadow_placements_total{kind,executor_kind,reason}`,
  `scheduler_shadow_disagreements_total{kind}` — the planner would have placed
  something the real loops did not within 60 s, or vice versa. This is the
  number that has to go to ~0 before P2.

## 6. Key families added (and none removed)

`demand-events/`, `demand/`, `executors/`, `assignments/`, `leader`,
`policies/`. All are ignored by older binaries. Review point 5 holds: every
existing loop keeps running for every table throughout P1.

## 7. Tests P1 must land with

1. `fold` is idempotent and commutative over any permutation/duplication of
   events (property test, no etcd).
2. A scan with an equal epoch cannot lower `sealed_through`; a higher epoch can.
3. Headroom is computed from assignments, not samples: an executor whose
   `bytes_free_sample` says 2 GiB but with 2 GiB reserved gets no placement.
4. Leader failover rebuilds reservations and reaches the same placement
   decision as the old leader on the same inputs.
5. Shadow mode writes nothing an executor reads (grep‑level test on the
   executor code paths plus an etcd test that binds nothing).
6. Class 1 is served oldest‑first: with two class‑1 tables, a low‑score table
   promoted for age is placed before a high‑score critical merge that has
   waited less, given one feasible executor.
7. A late scan snapshot with an equal epoch and a lower `sealed_through`
   leaves the table record unchanged; only a strictly higher epoch may lower
   it (already covered by `scan_may_raise_but_not_lower_without_higher_epoch`
   in #344; keep it).
8. Backpressure shaping never raises the writer's un‑flushed row count or
   delays a flush past the configured interval; under `critical` the
   observable effect is a slower accept rate, not a larger memtable.
