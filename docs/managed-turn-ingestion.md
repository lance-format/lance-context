# Managed ingestion with an application body schema

`lance-context-ingest` can provision a fresh managed body/annotation pair using
`TurnIngestionSpec`. The body keeps the application's Arrow schema, metadata,
original IDs and relationship columns. The specification binds ordered primary
key columns and a routing column to the table's durable contract. Key columns
must be non-null UTF8; the routing column must belong to the primary key.

For example, `primary_key = ["session_id", "id"]` and
`routing_column = "session_id"` distinguish the same ID in different sessions
while routing every occurrence of one business key to the same physical shard.
The internal annotation key is a canonical JSON tuple, obtained with
`TurnIngestionSpec::business_key` or `prepare`. It does not replace the original
body columns. Changing the specification requires a new table binding.

## Write and publish

Use the existing APIs with the `materializer` feature:

1. Construct `TurnIngestionSpec::new(body_schema, primary_key, routing_column)`.
2. Partition input by `PreparedTurns::bucket`, using the deployment's immutable
   layout. Preparation preserves duplicate rows and body buffers; reserve its
   identity workspace within the caller's existing memory admission budget.
3. Explicitly provision each fresh physical shard with
   `bootstrap::annotations::mixed::provision_turn_shard`.
4. Start its owned `annotations::runtime::Runtime::connect_turn_shard`. This
   reuses the bounded active/frozen staging, RF3 apply, batch publication and
   recovery path. The process configuration's optional `annotations.turns`
   accepts `{ "schema": <Arrow Schema>, "primary_key": [...],
   "routing_column": "..." }` and requires an explicit shard and body URI.
5. Build initial narrow annotations using the internal business keys and wrap
   the original body with `annotations::mixed::wire::creates_for`. Use
   `patches_for` for annotation changes. Bind the physical shard with
   `Shard::bind_batch`, then serialize with the ordinary Arrow IPC file writer.
6. With `http-client` enabled, send through
   `http::journal::DurableMutationClient`. Its `stage`, `append`, `resume`,
   `last_completed` and `shutdown` share the existing durable client journal and
   producer-lane state machine. Reopen with the same persisted producer-open
   tuples, journal directory, specification and binding. A different ingress
   endpoint is permitted.

A mutation succeeds after quorum persistence and ordered application. The owner
chooses the first committed create for each business key, including duplicates
from another producer or after restart. A repeated create preserves the first
body and current annotations. Different payloads for the same key are **not**
compared for conflicts, and there is no per-row inserted/duplicate result.

HTTP 422 carries a durable `MutationResult::Rejected`, not an ambiguous network
failure. The client journals that terminal result and advances its sequence.
Transport failures leave the original pending body and sequence intact; `resume`
resends that intent. Cancellation or an unknown local journal write fences the
lane until reopen. The local client journal is not itself a replicated volume.

For unresolved HTTP responses, downcast the error chain to
`http::journal::UnresolvedResponseStatus` for the status even if an intermediary
returned empty or malformed JSON. `AppendResponseError` additionally preserves
validated service labels when present. Status alone does not establish whether
a mutation committed or authorize replacing its durable pending intent.

## Read and source progress

Annotation point reads use the internal business key and the existing owner read
barrier. Forwarding ingresses route that key by its configured session component.
`Service::get_turn` returns the original body and separate current annotations;
a body's own `score` column is not overwritten by an annotation score.

For a single full-body table provisioned with `bootstrap::provision_body_shard`,
the owner and forwarding routers expose
`POST /api/v1/annotations/{table}/read-body`. Send a JSON point request such as
`{"turn":"[\"session-1\",\"turn-1\"]"}`. Construct that tuple with the table's
`TurnIngestionSpec::business_key`; forwarding uses its configured routing
column, so the component names and order belong to the application schema.

A found row returns HTTP 200 with an Arrow IPC file
(`application/vnd.apache.arrow.file`) containing one row in the original body
schema. The owner reads its compatible applied staging/Lance view, including
acknowledged rows awaiting publication, and checks its owner fence before
returning. It does not scan the WAL on the point-read path. A missing row returns
HTTP 204 with no body; memory pressure or an unavailable owner returns HTTP 503.
The application's authentication boundary still applies to these routes.

The owner admits read/encoding workspace and output backing before reading,
and retains that admission through response consumption. Forwarders stream the
owner's Arrow bytes without decoding or re-encoding them. A client can validate
the response schema, row count and exact requested key with
`http::annotations::full_body::decode`; the client must bound and admit its own
input/decode memory and retain that admission while using the result.

`Service::selected_body` and the HTTP `selected-body` route return the exact
managed body snapshot and covered WAL cut, scoped by cluster, layout, physical
table, incarnation, group and shard. `DurableMutationClient::selected_body`
validates that scope. Open the returned body URI at its **exact version**; managed
bodies may use detached Lance versions, so opening the URI's latest mainline
version does not select the same data.

The selected reference is not a reader lease or an external source checkpoint.
Applications must also durably account for all source outputs and associations
before advancing a source cursor. This API currently accepts turn creates and
annotation patches; arbitrary source-association or split-fence records are not
part of its wire contract. Even a batch containing only duplicate turns advances
the managed WAL cut without writing more bodies, but that does not cover omitted
application outputs.

Exclusive physical file reclamation is disabled for this generic body mode until
external reader retention is integrated. WAL source progress uses the existing
managed publication protocol. Long-running storage reclamation and a real
application's source checkpoint bridge require separate qualification.

## Executable integration checks

The Rust `tests::generic_turns` tests use actual RF3 services, Lance files and
HTTP ingresses. They cover composite-key deduplication, updates during a frozen
publication, exact body preservation, duplicate-only publication, reopen, lost
HTTP responses and durable rejections.

```sh
cargo test -p lance-context-ingest --features service,http-client --lib generic_turns
```

An application can supply `LANCE_CONTEXT_TURN_FIXTURE_DIR` containing four Arrow
IPC files named `request-0.arrow` through `request-3.arrow` with one batch each.
They must have a common schema with non-null UTF8 `session_id` and `id`, row counts
2/2/4/2, and six unique keys: the second request repeats the first; the third
repeats it and adds two keys; the fourth adds two keys in another session.
`LANCE_CONTEXT_TURN_OUTPUT_DIR` optionally retains successful publication files
and `selected-body.json` for independent reader compatibility checks.

These are correctness checks, not throughput or real-source acceptance results.

## Turn-ID existence index

Set `publication.turn_id_index` to maintain an index on the narrow table's
canonical key column `id`. `index_type` selects the native structure:

```json
{ "turn_id_index": { "min_rows": 65536, "index_type": "zonemap" } }
```

- `btree` (default, unchanged): stable-row-ID BTree segments.
- `zonemap`: per-fragment min/max zones over the canonical JSON key. A zone match
  is only a candidate; the existing exact `id IN (...)` filter still decides
  membership, so results are identical with or without the index.

A table never mixes the two: maintenance refuses to add a segment of the other
type, and changing `index_type` requires a fresh table binding.

ZoneMap answers in the physical-address domain and cannot be remapped. When a
small-file compaction rewrites covered fragments, the Rewrite commit withdraws
those fragments from the ZoneMap's coverage (native Lance, stable row IDs only),
reads of them fall back to an exact scan, and the next maintenance turn rebuilds
coverage for the uncovered fragments only. Segments whose coverage was fully
withdrawn are retired by that build. Compaction therefore never waits for, or is
blocked by, the ZoneMap. This exemption applies only to ZoneMap segments named
`lance_context_turn_id` on stable-row-ID tables; every other address-domain
index still requires an eager remap.

Build and merge use the existing bounds: at most 128 fragments and 1,048,576
rows per build, four-way tiered merges of similarly sized segments through
Lance's native `merge_existing_index_segments`, and the shared maintenance
workspace. Segment count is not otherwise capped; the tiered merge keeps it
logarithmic in the number of builds for similarly sized tails.

### When ZoneMap helps

Pruning depends on key order across fragments. Each publication batch is
written in canonical-key order, but separate publications overlap in key
range, so a lookup reads every fragment whose `[min, max]` range contains the
key. A synthetic cold-session profile (`turn_id_zonemap_cold_lookup_profile`,
ignored by default; 32 fragments x 16,384 rows; production predicate; native
`ExecutionSummaryCounts`):

| fragment layout | lookup window | no index | ZoneMap |
|---|---|---|---|
| key ranges disjoint per fragment | 20 keys in 4 sessions | 6.17 MB, 96 IOPS, 0.34 s | 0.40 MB, 12 IOPS, 0.05 s |
| key ranges disjoint per fragment | 64 keys spread over table | 6.13 MB, 96 IOPS, 0.35 s | 2.91 MB, 78 IOPS, 0.27 s |
| keys random across fragments | either | 6.8 MB, 96 IOPS, 0.33 s | same bytes and IOPS, 0.50 s |

When fragments' key ranges overlap, ZoneMap prunes nothing and adds index
load time. It is worth enabling only if the table's fragments have mostly
disjoint key ranges. Compaction concatenates fragments without sorting by key,
so it does not create that property. These are synthetic local measurements,
not real-traffic or capacity results. Per-key zone hit counts are not exposed
by native Lance and are not reported.
