# Inspecting merge failures

`GET /api/v1/scheduler/merge-failures?after=<cursor>` reads the durable retry
ledger. It returns `failures` (at most 256 records) and `next` (an opaque cursor,
or null at the end). An invalid cursor returns HTTP 400. The endpoint only reads
etcd metadata; it never opens WAL payloads, probes workers, retries a merge, or
resets a failure budget. A ledger changing during pagination is not a snapshot.

Each record includes `target`, `endpoint`, `class`, `consecutive_attempts`,
`last_error`, `last_failure_ms`, `next_retry_ms`, and `needs_attention`.
Timestamps are Unix milliseconds. Errors are capped at 4096 characters in the
runtime ledger. `needs_attention` means automatic probes are insufficient to
resolve the incident; it does not authorize data deletion or bypass ownership.

Retry, storage-fencing, and rollout behavior are defined in
[merge-recovery.md](merge-recovery.md). This API adds observation only and does
not change those policies or provide an external notification integration.
