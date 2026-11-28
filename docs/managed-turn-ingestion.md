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

## Replacing an executor that lost its journal

`http::journal::ProducerControlClient` uses a small scheduler-owned control
record. It reuses the WAL's existing generation fence and exact-tail retirement;
no separate producer registry is required.

1. Retain the original `ProducerOpen` tuple and an independently confirmed source
   checkpoint in durable scheduler state. After executor loss, choose and persist
   one replacement operation UUID.
2. Call `prepare_replacement` with that original tuple and operation. It recovers
   the signed predecessor credential. Durably save the **entire** returned
   `ReplacementIntent` before calling `execute`.
3. Execute that saved intent. The service fences the exact predecessor. It returns
   `WaitingForRetirement` until the predecessor's latest source record has been
   published and retired. Keep the same intent while waiting or after an unknown
   response. Do not rerun preparation after replacement might have succeeded:
   the original producer generation may already be gone.
4. `Opened` returns the successor `ProducerOpen` tuple and credential. Persist the
   tuple before starting its single executor with a fresh journal. Resume source
   **first-create** operations from the last fully confirmed checkpoint. Existing
   business keys preserve the original body and current annotations.

The managed `replace-producer` HTTP route accepts the signed predecessor and
replacement operation, routes by that credential's table/shard scope, and retains
the owner's publication fence. A competing operation or stale generation receives
409. The Rust client exposes a validated fence as `ProducerReplacementFenced`.
It does not automatically retry. Pending-retirement polls use quorum observations.

Repeating a successful intent returns the same live successor, even if it has
already written. This is **not** permission to restart its sequence at one with an
empty journal. If that successor executor also loses its journal, fence it with a
new replacement intent. Scheduler ownership must prevent two executors using one
generation. Replacement does not resolve old request results, make arbitrary
annotation patches safe to replay, or complete source-output obligations. Keep
those recovery records until the application's checkpoint protocol permits their
removal.

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

## Membership cache observations

The existing annotation observation includes `immutable_membership` cache
diagnostics. Its `misses` is the sum of `miss_disabled`, `miss_absent`,
`miss_wrong_key`, `miss_wrong_lineage` and `miss_future_cut`. These describe
resident probes, not completed ingestion, publication or source progress.
`miss_absent` cannot distinguish a new key, eviction, unsupported resident
lookup or a fresh owner namespace. Ordinary verified append publication preserves
the namespace and lineage. `warms` counts submitted insertions, which may be
immediately evicted; `refusals` counts backing admission or size failures, not
evictions.

`session_cache_capacity_bytes` and `session_cache_reported_entries` report the
existing shared Session backend, including other entry types and namespaces.
The latter is an approximate entry count whose zero value can also mean the
backend does not support a cheap count. Neither field measures membership-only
resident bytes. No eviction count is exposed by the current cache API. These
observations perform no cache enumeration, maintenance flush, storage read or
extra key lookup; each membership probe still updates one atomic counter.

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
withdrawn are retired by that build, so a rewrite needs no ZoneMap remap.
Compaction is still subject to its other checks (uniform coverage across the
task, stable row IDs, no other address index needing an eager remap), and the
rewrite and a concurrent index build still serialize through the publication
fence. The exemption uses the same typed predicate as native compaction
(`withdraws_zone_map_coverage`: decoded `ZoneMapIndexDetails`,
`index_version == 0`); every other address-domain index still requires an
eager remap.

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
disjoint key ranges. By default compaction concatenates fragments without
sorting by key, so it does not create that property. These are synthetic local
measurements, not real-traffic or capacity results. Per-key zone hit counts are
not exposed by native Lance and are not reported.

### Clustered compaction for the full-body dedup lookup

Sessions arrive interleaved, so every appended full-body fragment spans almost
the whole `session_id` range and no ZoneMap can prune it. Full-body Turns shards
enable both dedup indexes by default when the caller leaves them unset (and the
publication budget covers the index workspace): a Bloom filter on `id`
(`index_type: "bloomfilter"`, prunes per zone independent of layout) plus
write-time clustering (compaction `sort_by` = routing column then key, and the
routing ZoneMap). The equivalent explicit settings, with ZoneMap on `id`:

```json
{
  "compaction": { "target_file_bytes": 134217728, "min_fragments": 4,
                  "sort_by": ["session_id", "id"] },
  "turn_id_index": { "min_rows": 65536, "index_type": "zonemap",
                     "routing_zonemap": true }
}
```

- `compaction.sort_by` rewrites each bounded task's rows in that key order
  (then by stable row ID), using a vendored Lance `CompactionOptions::sort_by`
  (Lance 14 has no compaction sort key). Every row keeps its stable row ID and
  created/updated versions, carried as a permutation. The sort runs inside the
  task's admitted compaction workspace (pool = 2 x `target_file_bytes`) and
  spills to the temporary directory beyond it. A sorted rewrite always
  re-encodes and records no address moves, so it is used only on stable-row-ID
  tables without blob columns whose indices need no address remap (BTree
  address segments, system indices); other tables keep source order.
- `turn_id_index.routing_zonemap` keeps a ZoneMap named
  `lance_context_turn_routing` on the spec's routing column, with the same
  bounded build/merge policy as the turn-ID ZoneMap. A rewrite withdraws its
  coverage of the rewritten fragments; the next maintenance turn rebuilds it
  over the sorted output. Full-body shards accept `zonemap` or `bloomfilter` turn indexing, not BTree.
- `lookup_existing` emits the routing column's `IN (...)` first. Lance parses
  `IN` into a `SargableQuery::IsIn` that ZoneMap answers per zone, so fragments
  and zones outside the requested sessions are skipped before decode; the `id`
  membership and the exact composite check still follow.

`clustered_zonemap_lookup` (16 interleaved appends of 512 sessions, 256-row
zones) asserts the effect with the lookup's `fragments_scanned`/`rows_scanned`:
the unclustered table scans all 16 fragments and 8,192 rows with a ZoneMap; the
sorted, re-indexed table scans at most 4 fragments and 2,048 rows for a 4-session
window.

The win is limited to the compacted share of the table. Recent appended
fragments that compaction has not yet rewritten remain interleaved and are
scanned in full, as are rewritten fragments until their ZoneMap coverage is
rebuilt. Sorting is per task, so separate tasks' outputs still overlap; each
task's fragments are internally narrow, and a lookup scans about one fragment
per requested session per task.

## Bounded Arrow-file replay

Build the example inside this repository, using its root `Cargo.lock` and
`[patch]` entries. Cargo does not inherit these patches when another workspace
adds a git dependency on `lance-context-ingest`. An independently configured
consumer workspace must supply and qualify its own matching patches.

```sh
cargo build --locked -p lance-context-ingest --features service,http-client \
  --example managed_turn_replay
managed_turn_replay spec input.arrow session_id session_id id > turn-spec.json
managed_turn_replay validate replay.json input.arrow
managed_turn_replay append replay.json input.arrow
managed_turn_replay append-batched fresh-replay.json input.arrow
managed_turn_replay resume replay.json
managed_turn_replay verify replay.json input.arrow replay.log owner.json
managed_turn_replay slot owner.json GROUP SLOT
```

`replay.json` supplies `layout`, `binding`, `spec`, staging `limits`, `shard`,
the persisted producer `open` tuple, `endpoint`, `bearer_token_file` and
`journal_directory`. Generate `spec` from the actual input: schema metadata is
part of the binding, so a synthetic fixture does not establish a real input's
schema contract. Provision the matching fresh table pair before replay.
`validate` checks schema and business-key bounds without opening a journal or
contacting the service; its output contains aggregate counts and lengths only.
`slot` performs a read-only quorum query with the owner's existing trusted WAL
configuration. Slots are allocated per Raft group across all table bindings;
using the same slot number for two tables does not create independent producers.

This entry runs one producer lane on one already partitioned shard. It uses the
normal `DurableMutationClient`, accepts an IPC file of at most 128 MiB and bounds
each request to the configured limits (at most 8 MiB / 8192 rows). An oversized
individual turn is rejected. It shares a 1 GiB memory budget with its journal.
For wide schemas, the existing service-test configuration uses
`RUST_MIN_STACK=16777216`.

`append` requires a fresh journal directory. On an uncertain result, preserve
that directory and producer tuple; `resume` resolves its pending intent and
waits for its selected publication. It does not resume a source reader or infer
which later file rows to skip. Each successful request reports the input batch,
row interval, producer sequence and WAL index. Current replay output uses an
absolute input row start and count, including when a request spans source batches.
The final JSON reports the exact
selected body reference and covered cut. Preserve version numbers as `u64` or
raw JSON: detached versions can exceed JavaScript's exact integer range.

`verify` checks that the replay ACKs cover the entire input, then opens the exact
selected body version and validates its native transaction and authorized URI.
It compares schema/metadata and every value against the first input occurrence
of each business key, rejecting missing or duplicate published keys. This bounded
test expects a fresh destination containing only that input's unique keys. It
prints only counts and snapshot identities. Its expected input is limited to
128 MiB IPC, one million rows and 512 MiB of unique Arrow buffer allocations;
these limits are verifier bounds, not a native scanner RSS quota.

`append-batched` groups consecutive input batches toward half the configured
record byte limit, while respecting the row limit. It writes them into one IPC
file without concatenating body arrays. If actual encoded bytes exceed the
record limit, it splits the group with shared Arrow slices before sending.
It uses a fresh journal and one sequential producer lane, just like `append`.
Compare it with `append` on the same input and controlled cold/warm state;
duplicate-only replays do not measure throughput for newly inserted rows.

The reported times measure this sequential replay, including client journal
sync, encoding, RPC and publication waiting. They are not a sustained capacity
benchmark. The entry never advances an external source checkpoint, and its
candidate-row count is not a count of newly inserted unique turns.
