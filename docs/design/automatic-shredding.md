# Automatic JSON shredding

Tracking issue: [#832](https://github.com/lance-format/lance-context/issues/832).

The `shredding` module and `DocumentStore` select scalar object paths
automatically. Callers supply JSON values and resource limits, not extraction
paths. The embedded Rust and Python APIs persist the layout, reconstruct complete
documents, filter logical fields, and prepare native full-text indexes.

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

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
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

## Logical queries and text search

`filter_eq(path, value, limit)` returns matching generated IDs. A compatible
scalar predicate runs against its typed column, while rows with physical nulls
are checked through reconstructed JSON. The fallback covers values written
before promotion, type changes, containers, and explicit nulls. Missing is not
equal to JSON null. Results have no ordering guarantee.

`search(query, limit)` searches the automatically selected string paths and
returns IDs with native Lance full-text scores. It uses a lowercase simple
tokenizer without stemming or stop-word removal. Arrays, root scalar documents,
and paths outside the extraction budget are not full-text indexed in this
increment; they remain available through document reads and logical filtering.

The first search after layout growth automatically backfills historical values
and builds missing inverted indexes. `prepare_text_search` performs that work in
advance. Preparation can be expensive: it streams a rewrite of the dataset and
creates an index for each selected string path. Subsequent appends with the same
layout use Lance's indexed-plus-unindexed-tail search. Selection is automatic;
this initial indexing policy deliberately favors correctness over maintenance
cost. Query-aware index selection and asynchronous maintenance remain follow-ups.

Backfill retains generated document IDs and publishes an overwrite with zero
commit retries. A stale rewrite fails instead of rebasing over newer appends.
The backfill marker and data are published together; failed index creation can
be retried. Pinned snapshots can search if already prepared, but cannot trigger
backfill or index creation. Retained versions continue to hold their old bytes.


## Python API

```python
from lance_context import DocumentStore

store = DocumentStore.create("documents.lance")
ids = store.add([
    {"body": "otters swim upstream", "metadata": {"source": "wiki"}},
    {"body": "otters dive underwater", "metadata": {"source": "notes"}},
])

assert store.get(ids[0])["body"] == "otters swim upstream"
assert store.get_path(ids[0], ["metadata", "source"]) == "wiki"
hits = store.search("otters")
matching_ids = store.filter_eq(["metadata", "source"], "wiki")
reopened = DocumentStore.open("documents.lance")
```

Paths in reads and filters are literal key sequences, not extraction settings.
The Python API raises `KeyError` for a missing document/path and returns `None`
for an explicit JSON null. `add` accepts a sequence of documents, including root
arrays/scalars/null, and returns generated IDs. JSON nesting is limited to 127
container levels so accepted records remain readable. Residual floating-point
values use round-trip precision. Python integers must fit signed or unsigned
64 bits; larger integers, non-finite floats, and non-string object keys are
rejected before writing instead of silently changing their values.

Use `policy={"max_columns": 16}` or other `ShredPolicy` limits at creation to tune
selection without naming fields. The persisted policy is reused on open. The
first search can perform a substantial rewrite/index build; call
`prepare_text_search()` before serving latency-sensitive searches when needed.

## Reproducible performance comparison

The `automatic_shredding` Rust example compares three representations using the
same release executable and Lance dependency tree: complete JSON (`json`), JSON
plus duplicated `body` and `late_text` columns (`duplicated`), and the public
automatic `DocumentStore` (`shredded`). It rejects debug builds and existing
output directories. Run each arm in a separate process so peak RSS is independent:

```bash
cargo build --release -p lance-context-core --example automatic_shredding \
  --features lance-io-metrics
target/release/examples/automatic_shredding --mode json \
  --output /tmp/shred-json --rows 128 --body-bytes 262144 --samples 15
target/release/examples/automatic_shredding --mode duplicated \
  --output /tmp/shred-duplicated --rows 128 --body-bytes 262144 --samples 15
target/release/examples/automatic_shredding --mode shredded \
  --output /tmp/shred-automatic --rows 128 --body-bytes 262144 --samples 15
```

Use the same worker count, row count and body size for each arm. Repeat with fresh
directories, rotate arm order, and include small records and tens-of-MB records.
Save the source revision, compiler version, executable checksum and machine/resource
limits alongside results. Synthetic text uses a deterministic 4,096-token
vocabulary; it does not represent every application's compression or search costs.

Each directory contains `result.json` with individual samples and median/p95
latencies, requested object-store bytes, current and retained data bytes, index
bytes, and total directory bytes. Linux peak RSS covers the entire process,
including fixture generation, ingestion, validation and index construction;
it is not a per-operation allocation measurement. I/O counters measure logical
storage requests, not physical disk traffic. Reads run with warm OS caches;
the first read after reopen resets handle caches but does not flush the OS cache.

Correctness checks cover reconstruction, projected reads, residual filtering,
search, layout growth and reopen outside measured phases. No arm has an ID index.
The duplicated baseline materializes both text columns upfront, while automatic
shredding promotes the rare historical `late_text` only on a later write. Initial
index preparation and preparation after layout growth are measured separately
from steady-state search. JSON-only FTS is omitted because it has no equivalent
native text index. These are local embedded measurements, not cloud or service
capacity guarantees.

Both baseline arms store serialized JSON in a `LargeUtf8` column. This isolates
extraction and duplication costs using the same string encoding as the document
residual; it is not a benchmark of Lance's JSONB extension or Arrow Variant.
