# Overlapping immutable encoding with upsert deletion

`CATCHUP_PARALLEL_UPSERT_IO_TARGETS=table_a,table_b` enables this experiment for
exact rollout table names on native master/catch-up executors. The default is
empty. Generic stores and append-only merge do not use this setting.

A prepared upsert batch already contains the deduplicated replacement rows.
Instead of waiting for old-key deletion before encoding those rows, the executor
encodes and uploads immutable files concurrently with deletion. It publishes
those files only after both operations succeed, under the existing table owner,
expected base version and commit fence. A schema or field-ID change rejects the
staged files. No second metadata committer is introduced.

This works with predicate deletion and with `CATCHUP_BULK_KEY_DELETE_TARGETS`.
It does not assume append-only input. Unrelated keys, replacement blobs, later
WAL generations and replay after failed publication retain their existing
semantics. A failed deletion still joins the admitted encoding operation; failed
encoding leaves the WAL recoverable. Unpublished files can remain until ordinary
storage cleanup. The delete and append remain separate transactions, as before.

The same decoded-byte reservation is held through both operations. There is no
third prefetched batch, but concurrent deletion and encoding can increase scratch
memory and CPU usage; the decoded-byte budget is not a process RSS bound. This
option does not increase merge slots or relax fencing. A draining executor stops
new batches and joins admitted work before persisting continuation and releasing
ownership.

Qualification must compare identical images, byte budgets and table/index
profiles with this switch off and on. Measure whole-pass committed generations,
`delete_keys`, `encode_upload`, peak process memory, failures and read latency.
`delete_keys` and `encode_upload` overlap within `append`: do not add their times
as if they were sequential. Encoding progress is not proof of committed WAL
reclamation. No production throughput gain is established by local correctness
tests or by successful immutable file upload alone.
