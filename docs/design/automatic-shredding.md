# Automatic JSON shredding

Tracking issue: [#832](https://github.com/lance-format/lance-context/issues/832).

The `shredding` module selects scalar object paths automatically. Callers supply
JSON values and resource limits, not extraction paths. Its first increment is a
storage-independent codec; persistent document access and search are subsequent
increments tracked in the parent issue.

## Selection

`ShredLayout::evolve` profiles a bounded prefix of an incoming batch and appends
new fields to the existing layout. It never reassigns an existing column. A path
and scalar type qualify when their occurrence fraction or total sampled value
bytes meets the policy threshold. Candidates rank by bytes, then occurrences,
with deterministic path/type tie breaking. Only one scalar type is selected per
path; other types remain in the residual. Strings, signed/unsigned integers,
floating-point numbers, and booleans have separate physical types.

The default policy allows 32 columns and profiles at most 256 records, 100,000
JSON nodes, 4,096 path/type candidates, and 32 object levels. Candidate paths are
limited to 1,024 key bytes. A field qualifies at 50% occurrence or 64 KiB of
sampled scalar values. These are initial heuristics, not benchmark-derived
performance claims. Sampling is bounded rather than statistically random;
rare paths outside the sample may be discovered in a later batch. Once the
column budget is full, remaining values stay residual. Query-aware selection,
column replacement, and background re-shredding require later policy work.

## Semantics

`shred` consumes a document and moves matching scalar values into positional
slots. Those keys are removed from the residual; their object parents remain.
`reconstruct` restores the document using the same layout. Missing fields,
explicit nulls, mismatched types, arrays, and unselected values stay residual.
In particular, an extracted empty string is distinct from a missing field or
JSON null. Literal path components avoid ambiguity for keys containing dots,
slashes, tildes, and empty strings. Arrays are preserved as complete JSON values;
this increment does not extract wildcard paths inside arrays.

A reconstructed value is semantically equal to the input JSON value. This does
not promise original whitespace, object-key order, or duplicate-key text
preservation. Integers do not pass through floating-point conversion. Layouts
and policies are serializable, but a storage adapter must still version and
validate their persisted metadata. The decoder rejects duplicate paths,
incorrect value counts/types, absent object parents, and residual collisions.

An additive layout allows existing rows to take nulls in new physical columns:
their original values remain residual until an explicit rewrite. Queries must
therefore combine typed-column access with residual fallbacks, including old
rows and type mismatches. Merely querying newly added columns is insufficient.

The codec removes duplication between residual and extracted values. Indexes,
replication, WAL, and retained dataset versions still have their normal storage
costs. Whole-document reconstruction must read the extracted text too; storage
and latency claims need end-to-end measurements.

## Persistent document store

`DocumentStore::create(uri, policy, storage_options)` creates a versioned Lance
dataset; `open` reads the persisted policy and field mappings. `add` discovers
additional fields automatically, writes extracted values into typed columns,
and returns generated document IDs. An `id` key inside the JSON remains part of
the document and is independent of the store's generated ID.

```rust,no_run
use lance_context::{DocumentStore, shredding::ShredPolicy};
use serde_json::json;

# async fn example() -> Result<(), lance_context_core::LanceError> {
let mut store = DocumentStore::create("documents.lance", ShredPolicy::default(), None).await?;
let ids = store.add(vec![json!({"body": "some text", "metadata": {"source": "wiki"}})]).await?;
let document = store.get(&ids[0]).await?;
let body = store.get_path(&ids[0], &["body".into()]).await?;
# Ok(())
# }
```

The initial adapter is embedded and uses direct Lance commits with one writer
per dataset. It does not attach a MemWAL or expose an HTTP endpoint. Integrating
automatic schema evolution with resident WAL writers requires coordination that
the existing immutable-schema GenericStore does not provide.

Promotion adds nullable columns with their logical-path/type metadata in one
schema transaction. Older rows receive physical nulls while their logical values
remain in the residual. The next transaction appends the new documents atomically.
An interrupted append may leave unused columns, but cannot leave half a document.
Columns have deterministic path/type-derived names to avoid interpreting a stale
writer's data with a different logical mapping. Concurrent schema changes can
still conflict; this API does not claim coordinated multi-writer evolution.

`get_path` reads a typed column directly when populated, falling back to document
reconstruction for historical residual values, type mismatches, and container
paths. Explicit null and missing remain distinct. `checkout` pins both the layout
and documents to a read-only version; `refresh` restores the latest view. Field
selection, projection, and round-trip semantics are covered by real Lance tests.
