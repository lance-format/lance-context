# Explain a queued task

`GET /api/v1/tasks/{id}/admission` reports the current admission preconditions
for one task on the master that receives the request. Use a specific master
endpoint when comparing executors; the response includes `executor_id`.

This is an on-demand diagnostic. It does not claim or enqueue work, renew a
lease, recover an execution, remove a lock, or authorize replacement. It adds
no periodic polling. A lookup reads one task and then at most 73 exact keys in
one read-only etcd transaction; it never scans the fleet or reads WAL payloads.
Dependency lists longer than 64 return `dependency_limit` without evaluating
readiness. Avoid fleet-wide or per-second diagnostic loops.

The response separates local executor gates from shared task preconditions:

- `executor_blocker` is `executor_draining`, `wal_claims_disabled`, or null.
  These local observations are separate from the etcd snapshot.
- `diagnostic.snapshot_revision` identifies the common revision used for the
  queued task, its dependencies and ownership records.
- `diagnostic.reason` distinguishes `not_queued`, `dependency_waiting`,
  `dependency_failed`, `dependency_missing`, `wal_target_not_selected`,
  `admission_blocked`, and `admission_preconditions_satisfied`.
- A blocked ownership check includes the existing scheduler's `skip` reason
  (`ownership_blocked`, `yield_to_prepared_commit`, or
  `worker_execution_needs_merge`), the prospective `claim_shape`, and boolean
  ownership-presence fields. No owner tokens, storage locations, credentials
  or execution bodies are returned.

For example, an index task can have a completed compaction dependency and still
report `ownership_blocked`, with `claim_shape.preparing=true` and
`ownership.preparation=true`: another preparation already holds that table's
preparation slot. Presence alone does not prove an execution is stalled. Use
its actual progress and the normal fencing protocol before any replacement.

`task_changed_retry` or `queue_entry_missing_or_changed` means the task changed
between lookup and snapshot, or its queue entry disagrees with the canonical
record. Do not interpret that response as eligible; refresh the canonical task.
A missing task returns 404 and a malformed/oversized task ID returns 400.

`admission_preconditions_satisfied` does not measure free execution capacity,
prove FIFO traversal reached this task, or guarantee a claim will succeed. The
scheduler still reserves its local capacity and runs the normal claim CAS.
Compare `master_task_pool_in_use{pool,state}` and `/executor` claim progress
when investigating time spent scanning or waiting for capacity.
