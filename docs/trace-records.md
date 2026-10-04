# Trace records

`TraceRecord` stores one conversation turn per row in one Lance dataset. It is
a built-in schema and typed adapter over `GenericStore`: it uses the existing
WAL, BTree ID index, merge recovery, batch lookup, and HTTP routes. There is no
second copy of the turns, separate reference table, or per-turn checkpoint store.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | required string | Stable storage identity, reused on retries |
| `session_id` | required string | Conversation grouping key |
| `turn_id` | required int64, nonnegative | Zero-based turn order in the observed conversation |
| `role` | required string | User, assistant, system, tool, or a source-specific role |
| `content` | nullable large string | Exact content of this turn; null differs from empty |
| `content_type` | required string | MIME type; the typed record defaults to `text/plain` |
| `source` | nullable string | Source dataset or system |
| `metadata` | nullable large string | JSON-encoded source-specific annotations |

The typed Rust `metadata` field is `Option<serde_json::Value>`; `to_row` and
`from_row` convert it to/from the physical JSON-text column. Whole-call token
arrays, prompts, and responses should not be copied into every turn's metadata.
No embeddings, model calls, or training labels are generated implicitly.

## Identity and reconstruction

The caller assigns `id`. For deduplicated observations, derive it deterministically
from the source/tenant scope, session ID, turn ID, role, content type, and exact
content. Use unambiguous serialization and preserve null versus empty content.
Turn IDs come from the input's order, not arrival time. Receiving turns 0–2 after
turns 0–4 must produce the same IDs for the shared prefix.

`(session_id, turn_id)` is **not unique**. Different content at the same position
uses different IDs, retaining both observations. A session without variants can
be reconstructed by sorting on `turn_id`. When there are branches or truncated
inputs, positions alone do not establish a global trajectory or identify which
variants appeared in a particular call. The application must preserve that
association explicitly if required; the schema does not invent branch links.
Source deletions do not automatically retract a turn shared by multiple calls.

## Embedded Rust

```rust,no_run
use lance_context_core::{
    trace_schema, GenericStore, GenericStoreOptions, TraceRecord, TraceStore,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let backend = GenericStore::open(
    "az://container/trace-turns.lance",
    trace_schema(),
    GenericStoreOptions { seal_on_add: true, ..Default::default() },
).await?;
let traces = TraceStore::new(backend)?;
let record = TraceRecord {
    id: "stable-content-derived-id".into(),
    session_id: "session-42".into(),
    turn_id: 0,
    role: "user".into(),
    content: Some("Hello".into()),
    content_type: "text/plain".into(),
    source: Some("model_calls".into()),
    metadata: None,
};
let ids = vec![record.id.clone()];
let present = traces.existing_ids(&ids).await?;
if present.is_empty() {
    traces.add(&[record]).await?;
}
let records = traces.get_many(&ids).await?;
let mut backend = traces.into_generic();
backend.cleanup_wal().await?;
backend.create_id_index().await?;
backend.close().await?;
# Ok(())
# }
```

`TraceStore::new` checks the persisted schema before typed access. The same
adapter accepts core `GenericStore`, the local/remote `lance_context::GenericStore`,
or `lance_context_client::RemoteGenericStore`. Their maintenance and filtered
projection APIs remain accessible through `as_generic` / `into_generic`.

## Batch deduplication and durability

`get_many` returns complete typed records. `existing_ids` uses the **same batch
lookup introduced for GenericStore**, projecting only `id`; it does not fetch
content or metadata. Both accept up to 1,024 input IDs (including duplicates),
return found IDs once in first-requested order, and omit missing IDs. Larger
requests must be split into bounded batches by the caller. Reads resolve the
base and flushed WAL together, including newest-ID resolution during merges.

An existence probe followed by `add` is not a cross-writer atomic insert-if-absent
operation. Coordinate writers when immutable conflict rejection is required.
`add` validates the complete typed batch before writing and inherits generic
newest-write-wins semantics. Reusing an ID with changed content replaces that
logical record; it does not preserve a variant or merge metadata/provenance.

`seal_on_add: true` makes the append visible before acknowledgement; otherwise,
call `flush` before depending on a previous write being visible to dedup probes.
Only advance the source checkpoint after durable destination acknowledgement.
The returned base version is not a per-append commit token or a complete source
snapshot. Checkpoints and source scheduling belong to the ingestion service.

## HTTP and remote Rust

Create a generic store with `schema: trace_schema()` and the required
`seal_on_add` setting, then use the existing endpoints:

- `POST /api/v1/generic/{name}/rows`: batch append encoded trace rows.
- `POST /api/v1/generic/{name}/get-rows`: batch lookup; use
  `{"ids":["a","b"],"columns":["id"]}` for deduplication probes.
- `POST /api/v1/generic/{name}/flush`: publish deferred appends.

Rust clients can wrap `RemoteGenericStore::connect_or_create` with
`TraceStore::new`, so typed encoding, batch reads, and ID-only probes are identical
for local and remote storage. Raw generic HTTP writes enforce the physical schema;
the additional nonempty-identity/nonnegative-turn checks are performed by the
typed adapter. Raw callers must apply the same semantic validation.
