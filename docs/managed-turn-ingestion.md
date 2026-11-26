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
managed_turn_replay resume replay.json
```

`replay.json` supplies `layout`, `binding`, `spec`, staging `limits`, `shard`,
the persisted producer `open` tuple, `endpoint`, `bearer_token_file` and
`journal_directory`. Generate `spec` from the actual input: schema metadata is
part of the binding, so a synthetic fixture does not establish a real input's
schema contract. Provision the matching fresh table pair before replay.
`validate` checks schema and business-key bounds without opening a journal or
contacting the service; its output contains aggregate counts and lengths only.

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
row interval, producer sequence and WAL index. The final JSON reports the exact
selected body reference and covered cut. Preserve version numbers as `u64` or
raw JSON: detached versions can exceed JavaScript's exact integer range.

The reported times measure this sequential replay, including client journal
sync, encoding, RPC and publication waiting. They are not a sustained capacity
benchmark. The entry never advances an external source checkpoint, and its
candidate-row count is not a count of newly inserted unique turns.
