# Maintenance scheduling runbook

Symptom-driven guide for the master's maintenance layer: WAL merge, compaction,
key-index build, repair and cold retirement. It names the exact endpoint,
metric or etcd key to look at and what the answer means. Design background:
`docs/design/unified-maintenance-scheduler.md`; defect list and roadmap:
`docs/design/scheduler-improvement-backlog.md`; tracking: issue #333.

`P` below is `ETCD_PREFIX` (default `/lance-context/master`). `<hex>` is the
lowercase hex of the target name's UTF-8 bytes; `generic:<name>` targets
include the prefix in the hex.

## 0. The one rule

**Progress decides whether a merge, compaction or index build is stuck.
Nothing else does.** A live lease, a heartbeat, a Running task state, a retry
countdown, a Ready pod or long elapsed time are not evidence either way. The
only authoritative signal is the execution's completed-work sequence
(`P/merge-progress/<execution id>`), and the only authoritative stall verdict
is the watchdog's: `MAINTENANCE_IDLE_TIMEOUT_SECS` (default 600 s) with no
completed step, confirmed again before cancellation. If you are about to
restart something because it "has been running a long time", stop and check
§3 first.

## 1. Where to look, in order

| Question | Look at |
|---|---|
| Is WAL piling up anywhere? | `master_wal_backlog_generations`, `master_wal_backlog_largest_store_generations`, `master_wal_backlog_stores_with_shard_over_4096`; first verify `master_wal_backlog_snapshot_available == 1` ([metric semantics and Datadog queries](../datadog-wal-backlog.md)); per table `GET /api/v1/experiments/{name}` → `pending_wal_generations`, `fragment_count` |
| Are tasks waiting instead of running? | `master_task_schedule_to_bind_seconds{kind}` (enqueue → claim). Rising with `master_task_pool_in_use{pool,state="running"}` at capacity = **capacity**. Rising with `state="claiming"` high = **the etcd claim itself is slow or blocked** (§4, admission scan). Rising with both low = **nothing eligible** (§4). |
| Is something queued at all? | `master_task_queue_depth`; `GET /api/v1/tasks/{id}` for a specific task |
| Is a running merge actually working? | `GET /api/v1/merge-progress?target=<name>` (§3) |
| Is compaction/index getting its commit turn? | `master_commit_turn_requests_total{kind,result}`: `served` vs `expired`; `master_commit_turn_wait_seconds`; `master_merge_yield_to_compaction_total` (§5) |
| Is a table being skipped on purpose? | `GET /api/v1/scheduler/cooldowns`, `GET /api/v1/scheduler/repairs`, failure ledger `P/merge-failures/<hex>/` (§4) |
| Did the last task fail, and how? | `GET /api/v1/tasks/{id}` → `error`; `master_tasks_total{kind,result}` |
| Is a catch-up Job stuck? | `master_catchup_*`; `P/catchup-records/<hex>`, `P/catchup-active/<hex>`; `kubectl get job -n $CATCHUP_NAMESPACE` |
| Is the stats view stale? | `master_wal_backlog_snapshot_age_seconds`, `master_wal_backlog_snapshot_available`, `master_scan_duration_seconds`, `master_stats_version`; `POST /api/v1/experiments/{name}/rescan` refreshes one table now |
| Is this replica draining? | `GET /api/v1/executor` |

## 2. Backlog is growing on a table

1. Confirm it is real: `GET /api/v1/experiments/{name}`. If `pending_wal_generations`
   is stale (compare `master_stats_version` age with `STATS_SCAN_INTERVAL_SECS`),
   `POST /api/v1/experiments/{name}/rescan` and re-read.
2. Is there a live task? `GET /api/v1/merge-progress?target={name}`:
   * `no_active_record` → nothing is scheduled. Go to §4 (why not).
   * `progress_unknown` → a legacy worker RPC holds the task with no telemetry.
     Check the worker's own logs/metrics; the master cannot see inside it.
   * `progress_details_unavailable` → an execution exists and has a progress
     `sequence` but no work report (older executor). Sample twice ≥ 60 s apart:
     a changing `sequence` is progress.
   * `reported` → read `work_report` (§3).
3. If a task is live and progressing: this is **capacity**, not a fault. Check
   `master_task_pool_in_use{pool="merge",state="running"}` and `{pool="resident_merge",state="running"}`; the
   executor's memory budget (`ROLLOUT_APPEND_LOCAL_MEMORY_BYTES`,
   `CATCHUP_MERGE_MEMORY_BYTES`); and whether the table is
   oversized-generation bound (`work_report` shows waiting-for-memory phases).
   Adding master replicas adds execution capacity only for tables in
   resident/owned mode; legacy tables are bounded by their workers.
4. If `pending_wal_generations` ≥ 256 the read path warns; ≥ 4096 reads return
   503. There is no proportional write-side throttle today (backlog item C7a).

## 3. Deciding whether a running execution is stuck

Use `GET /api/v1/merge-progress?target=<name>` and compare **two samples of the
same `execution.id`** taken ≥ 60 s apart.

* `progress.sequence` increased → **working**. Leave it alone, however long it
  has run. A merge on a hot table can legitimately run for hours as a chain of
  passes; compaction and index builds on large tables take many minutes with
  coarse progress; index trainers in Lance 9 report progress only through
  completed data/index file IO. Long is normal. Long **and** progressing is
  never a reason to act.
* `progress.sequence` unchanged for less than `MAINTENANCE_IDLE_TIMEOUT_SECS`
  → **undetermined**. Wait. Heartbeats and phase changes in `work_report` do
  not advance the sequence by design.
* unchanged for longer than the idle timeout → the watchdog will cancel the
  preparation, drain admitted commits, fence the manifest version and release.
  If it has not within ~2× the idle timeout, the executor process is likely
  gone without releasing: see §6.

What is **not** a stall signal: the task lease is alive; the pod is Ready; the
Job is Active; `needs_attention` is false; retry backoff is counting down; the
task has been Running for an hour. What is **not** a health signal:
`status: reported` (it means telemetry exists, not that work is advancing).

## 4. A table has demand but nothing is scheduled

Admission reasons, cheapest first:

1. **Cooldown** (Compact/IndexId/Repair only; MergeWal is never cooled down):
   `GET /api/v1/scheduler/cooldowns`. Produced after
   `TASK_COOLDOWN_AFTER_FAILURES`; clears on success.
2. **Failure ledger**: `etcdctl get --prefix P/merge-failures/<hex>/`. Each
   record has `class`, `consecutive_attempts`, `next_retry_ms`,
   `needs_attention`, `last_error`. Classes: `Retryable` (2 s ×3 then 30 s
   doubling to 900 s), `OwnershipUnresolved` (30 s doubling to 300 s),
   `DataOrConfiguration` and `Deadline` (3600 s, `needs_attention`). Errors
   tagged `[LC_STALE_PREPARATION]` are retryable by code regardless of
   wrapping. An hourly cooldown on a stale-preparation message from a binary
   older than #338 is reinterpreted on read; nothing to do.
3. **Ownership**: a write-turn holder exists. `P/target-locks/<hex>`,
   `P/merge-executions/<hex>`, `P/merge-claims/<hex>`, `P/catchup-active/<hex>`.
   `master_task_admission_blocked_total` counts these skips. One holder per
   table is correct; two tasks on one table is not.
4. **Fairness yield**: `P/compact-commit-ready/<hex>` equals
   `P/compact-preparations/<hex>` → merge is deliberately yielding one turn to a
   prepared compaction/index. `master_merge_yield_to_compaction_total`.
5. **Mode**: the table is in `MERGE_DRAIN_TARGETS` (maintenance refused), or not
   in `MERGE_OWNED_TARGETS` while all MergeWal paths require ownership, or
   resident-only pollers are configured but the table is not in
   `ROLLOUT_APPEND_TARGETS`. There is no endpoint that prints the effective
   mode yet (backlog A7); read the master's startup config log.
6. **Dependency**: `GET /api/v1/tasks/{id}` → `depends_on`; a Failed dependency
   fails the dependent.
7. **Discovery lag**: nothing enqueued because no sweep has seen the demand.
   `MERGE_WAL_INTERVAL_SECS` (600 s), `MERGE_WAL_TAIL_INTERVAL_SECS`,
   `ROLLOUT_APPEND_RECONCILE_INTERVAL_SECS`, `CATCHUP_INTERVAL_SECS`, and the
   stats scan all gate it. `POST /api/v1/tasks {"kind":"merge_wal","target":"<name>"}`
   enqueues directly and is safe: dedupe prevents a duplicate.

Manual enqueue is always safe. Deleting ownership keys by hand is **never**
safe while any process that might hold them is alive (§6).

## 5. Compaction or index never commits on a hot table

Symptom: `master_commit_turn_requests_total{result="expired"}` rising;
`master_index_phase_duration_seconds{phase="commit_wait"}` near
`INDEX_COMMIT_WAIT_SECS`/`COMPACTION_COMMIT_WAIT_SECS`; task errors
`maintenance commit ownership wait budget exhausted; reprepare`.

First: this is **not** a fault in the merger. A continuous merger on a hot table
that keeps winning the write turn is doing its job and is making progress (§3
tells you how to verify that). Never stop, drain or restart a progressing
merge to let compaction in; that trades a bounded fairness problem for WAL
backlog and a fenced recovery. The symptom is a scheduling‑fairness gap, and
the remedies are scheduling remedies:

* If `master_merge_yield_to_compaction_total` is **flat** while `expired` rises,
  some contender for that table does not understand the commit‑ready hint: a
  master or catch‑up executor older than #324, or a legacy worker self‑merge.
  Upgrade that binary at the next rollout. Until then, the preparation will keep
  expiring on that table; that is wasted IO, not data risk.
* If yield is rising and `expired` still rises, the merger wins the race after
  yielding once. Raise `COMPACTION_COMMIT_WAIT_SECS` / `INDEX_COMMIT_WAIT_SECS`
  for that kind as a stopgap so the preparer survives more than one turn; the
  design fix is `max_consecutive_turns` (design §4.2).
* If the table is hot enough that it never leaves the critical class, there
  is **no per‑table lever today** to defer its compaction/index without side
  effects, and two tempting ones are wrong:
  * `MERGE_DRAIN_TARGETS` stops merge as well as maintenance.
  * Removing the table from `COMPACTION_PREPARE_TARGETS` /
    `INDEX_PREPARE_TARGETS` does **not** pause the task. It switches that
    table back to the legacy path, which takes the table write lock *before*
    compacting or indexing — so the next Compact/IndexId task blocks WAL
    publication for the whole build. That is strictly worse than repeated
    preparation expiry.
  Fleet‑wide levers exist: `COMPACTION_INTERVAL_SECS` and `quiet_hours` on the
  compaction sweep, and raising `MIN_FRAGMENTS` so the sweep does not pick the
  table up. A manual `POST /api/v1/tasks {"kind":"compact","target":...}` in a
  quiet window still works. A per‑table `kinds_allowed` policy is design §4.8
  and does not exist yet.
* A preparation that expires is discarded and rebuilt from scratch. Repeated
  expiry on a large table is wasted IO, not a correctness problem.

## 6. An executor died holding ownership

Signs: `P/merge-executions/<hex>` with `phase` `Running` or `Uncertain`, its
progress sequence frozen past the idle timeout, no process with that
`instance` alive; or `P/target-locks/<hex>` = `merge-execution:<id>` with no
live claim lease.

Do not delete keys. Recovery is automatic and fenced:

1. Any owning master's retry loop (15 s) sees the execution with
   `maintenance` set and enqueues the matching kind for recovery; MergeWal
   recovers worker executions.
2. The recovering task freezes the record, fences manifest versions (120 s),
   finishes recovery, releases, and writes the failure ledger atomically.
3. `master_merge_storage_recoveries_total` increments.

If a record stays `Uncertain` for more than 30 min, the fence step is failing:
check storage credentials and `master_tasks_total{kind,result="failed"}` with the
task error. Escalate before touching etcd.

Leased keys (`claims/`, `merge-claims/`, `target-locks/` tokens,
`compact-preparations/`) expire with `ETCD_LEASE_TTL_SECS` when the process dies;
unleased keys (`merge-executions/`, `target-locks/` = `merge-execution:*`,
`catchup-*`) are deliberately durable and are what recovery reads.

## 7. Draining a replica or stopping a table

* Replica: `POST /api/v1/executor/drain` with the `executor_id` from
  `GET /api/v1/executor`; it stops admitting and finishes in-flight claims. See
  `docs/operations/master-drain.md`.
* Table: add it to `MERGE_DRAIN_TARGETS` on **every** replica and helper. Draining
  resolves existing owned executions and refuses new mutations; it does not
  cancel a legacy worker's in-flight self-merge.

## 8. Known gaps (do not debug these as faults)

* No endpoint prints a table's effective scheduling mode (A7).
* Queue order is time-ordered without priority: a 4000-generation backlog
  enqueued after a 2-generation tail waits behind it (C6).
* Legacy worker RPCs can hold merge slots indefinitely; the resident pool
  (`ROLLOUT_APPEND_LOCAL_TASK_CONCURRENCY`) is a reserved lane, not a fix (C3).
* Backpressure is a cliff at 4096 generations, not a slope (C7a).
* Five discovery loops with separate freshness windows; demand can be seen by
  one and not another for minutes (A4).

The unified scheduler work in #333 replaces these; until then this document is
the map.
