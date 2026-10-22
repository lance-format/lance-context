# Retain legacy responses without holding scheduler slots

Legacy workers do not have request-correlated cancellation or storage fencing.
A merge POST that has been accepted cannot safely be retried merely because its
HTTP response is slow. Previously the master held its scheduler slot indefinitely;
a plain HTTP timeout would instead discard the one acknowledgement that could
resolve the durable execution.

The master now reserves a bounded response observer before durable admission and
before sending the POST. The observer owns the original HTTP future and a local
admission guard. After `LEGACY_MERGE_FOREGROUND_SECS` (default 30), the foreground
attempt yields its scheduler slot. This interval is **not a worker deadline or a
claim that progress has stopped**. The request continues, its durable target lock
remains, and another table can use the scheduler slot. The original attempt is
recorded as failed/pending, never as a completed merge; dependencies cannot advance
on that attempt. `master_merge_wal_response_yields_total` distinguishes this event
from actual worker response errors, whose counter advances only when a reply ends.

A success, 404, or proven pre-connect failure is persisted against the exact
execution with CAS, even after the original task/lease has ended. If etcd is
unavailable, the bounded observer retains the result and retries that CAS only.
It does not resend the merge. An ambiguous HTTP error retains the Running record.
Only a canonical reconciler with a current claim releases a terminal execution.

The execution stores the remaining serial fan-out, including previously skipped
endpoints. On late acknowledgement, the next task resumes that continuation, so
a slow first shard cannot repeatedly prevent later shards from running. Removed
configured endpoints invalidate that saved continuation and cause a new pass
under the current configured route. A known failed endpoint is still budgeted;
it is not silently counted as complete.

`LEGACY_MERGE_RESPONSE_CONCURRENCY` (default 32) limits **all** retained requests,
including those still in the foreground. Full capacity rejects new requests
before an execution record or HTTP POST is created. Reply payloads are capped at
64 KiB. Every observer remains in `/executor` as `legacy_merge_response`, with
`waiting_response` or `persisting_acknowledgement` phase, and process drain joins
it. Status/elapsed time are not storage-retirement evidence. The scheduler slots
and observer capacity are separate; permanently hung requests can still exhaust
the latter. This deliberately does not authorize unlimited requests.

## Compatibility and limits

- Existing requests already accepted by old master processes are not transferred
  to the new observer. This change does not retire production worker0/worker9 or
  justify clearing their locks or killing their processes.
- Old masters cannot CAS records containing the new continuation field after
  dropping that field during deserialization. Keep updated reconciliation
  capacity running until these records are gone; rollback can otherwise strand
  them conservatively.
- A process crash before persisting the HTTP reply still leaves an unresolved
  legacy record. Safe automatic takeover requires the existing fenced protocol
  and a verified migration of old writers; this patch does not manufacture that
  evidence.
- Worker fan-out stays serial for each target. No ConfigMap, Kubernetes RBAC,
  worker slot/memory setting, or separate execution pool is changed.

## Regression coverage

Real local etcd and HTTP fixtures verify that a one-slot scheduler finishes an
unrelated table while the original request remains live; late replies survive
foreground completion and retain exact ownership; full observation capacity and
drain create no unsent record; drain waits for acknowledgement persistence;
late completion resumes the next worker without reposting the first; and an
ambiguous HTTP 500 neither releases the record nor advances the fan-out.
