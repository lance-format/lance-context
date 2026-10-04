# Batched generic row lookup

`GenericStore::get_many(ids, columns)` reads up to 1,024 IDs with a single
LSM query per read attempt. The Rust `GenericStoreApi` trait exposes it on
embedded and remote stores. `ContextClient::get_rows` calls
`POST /api/v1/generic/{name}/get-rows`:

```json
{"ids": ["session-a:0", "session-a:1", "session-a:0"], "columns": ["id"]}
```

The response is `{"rows": [...]}`. Each found ID appears once, in the order
it was first requested; missing IDs are omitted. Empty input returns no rows.
The 1,024-ID limit counts duplicates. The HTTP endpoint retains Axum's default
request-body limit. There is no total response-byte limit: callers fetching
large blobs should use smaller batches and narrow projections.

Without `columns`, blob fields are omitted, matching `get`. An explicit
projection always includes `id`; `columns: ["id"]` (or `[]`) makes an
existence check avoid payload materialization. Deferred appends still require
`flush`; `seal_on_add` stores can read their writes on return. This API is a
read, not an atomic insert-if-absent or a uniqueness constraint for writers.

## Execution and correctness

Lance 9.0.0 routes a bare primary-key `IN` predicate to
`lookup_many_via_per_key` for disk-backed keys, so one API call can still
perform hundreds of serial reads. The batch API uses a small compatibility
adapter: an explicit zero offset selects Lance's general LSM scanner. It
pushes the whole `IN` predicate to each source, including indexed base and
flushed generations, and retains Lance's native newest-key suppression.
It also works without an index. No dependency versions are changed.

A plan regression test requires one scalar-index query for multiple IDs,
so dependency upgrades cannot silently restore the per-key path. The adapter
can be removed when Lance's native batch point lookup batches disk reads.
No limit is pushed down: older matches suppressed by newer generations must
not cause a short response. Reads use `StorageBase::read_consistent`, which
refreshes base/WAL views and retries invalidated merges instead of returning
partial results. Query cost still grows with pending WAL generations and
payload size; batching does not eliminate that read amplification.

Tests cover an indexed and unindexed base, multiple pending generations,
replacements within and across generations, missing/repeated IDs, escaped
IDs, projections and blobs, deferred visibility, base deletion vectors,
external merge refresh, and HTTP/client behavior. Generic stores currently
have no public row-delete/WAL-tombstone write API; the deletion test uses
Lance's public base-table delete API.

## Reproduction

```sh
cargo run -p lance-context-core --example generic_batch_get
cargo run -p lance-context-core --example generic_batch_get -- --pending-wal
# Read an existing private fixture without printing its contents:
cargo run -p lance-context-core --example generic_batch_get -- --uri /path/to/table
```

The default fixture is 2,048 rows with 2-KiB string payloads and a BTree ID
index, created in a temporary local directory. The second command leaves two
WAL generations unmerged. Each case has one warmup followed by five measured
reads; timing includes row materialization and excludes fixture creation and
correctness comparisons. All measured responses are checked against expected
IDs and values. The example logs only timing, byte counts and status. An
existing fixture is fully scanned to establish expected rows, so use a
bounded test fixture.
