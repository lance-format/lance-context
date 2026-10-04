# Batched generic row lookup

`GenericStore::get_many(ids, columns)` reads up to 1,024 IDs, normally with a single
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

## Local results, 2026-10-04

ARM64 shared host, unoptimized debug build, local filesystem, warm caches;
no concurrent build from this task during measurement. One warmup and five
measured calls per case. Times below include complete row materialization.
The private conversation fixture contains 721 rows, all merged into base;
its contents are not included. The synthetic fixture has 2,048 rows and two
pending generations, including overlapping replacements.

| Fixture | IDs | Legacy IN P50 (ms) | Batch P50 (ms) | Batch P95 (ms) | ID-only batch P50 (ms) |
|---|---:|---:|---:|---:|---:|
| Conversation, base | 1 | 11.94 | 10.18 | 10.41 | 8.65 |
| Conversation, base | 32 | 231.40 | 13.34 | 13.89 | 10.48 |
| Conversation, base | 512 | 3,703.85 | 42.35 | 42.89 | 18.87 |
| Synthetic, base + WAL | 1 | 21.85 | 18.12 | 19.43 | 14.61 |
| Synthetic, base + WAL | 32 | 523.33 | 20.00 | 20.45 | 18.45 |
| Synthetic, base + WAL | 512 | 7,892.15 | 49.42 | 50.51 | 40.83 |

The 512-row full responses were 831,557 bytes and 1,063,425 bytes respectively,
identical between legacy and batch lookup. Every measured response matched
expected rows; ID-only reads were checked separately. These are small,
warm-cache fixtures, not object-storage latency or ingestion-throughput
measurements; five samples do not characterize production tail latency.

Validation: 21 core generic-store tests and 10 generic HTTP route tests passed,
including a live HTTP/Rust-client round trip. Clippy passed for core, API,
client, server, and the facade with remote support (`--all-targets -D warnings`).
## Lance 9 WAL compatibility

Some WAL primary-key sidecars trigger Lance 9's
`RowAddrTreeMap::from_sorted_iter called with non-sorted input` error in its
batched membership probe, while native point lookup still reads the same rows.
Batch get retries that specific error through native point lookups against the
same captured base/WAL view, discarding any partial batch first. Other errors
still propagate. This compatibility path can have serial lookup latency; the
fast-path measurements above do not describe it.
