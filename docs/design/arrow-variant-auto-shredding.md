# Arrow Variant automatic shredding

This implementation uses the canonical `arrow.parquet.variant` extension and
Apache Arrow's `parquet-variant-compute` kernels. The existing `DocumentStore`
(custom typed columns plus residual JSON) remains a separate baseline. Its
correctness and performance results are not evidence for the Variant path.

## Kernel and selection layer

`AutoVariantLayout::infer` applies bounded automatic field profiling; callers
provide resource limits rather than extraction paths. Its target is passed to
Arrow's `shred_variant`. The result contains standard binary `metadata` and
`value`, with extracted values in `typed_value` children. No second residual
codec or custom reconstruction metadata is introduced. Selection settings are
adapter policy; they do not change the standard Variant representation.

The first targets are object paths containing strings, signed integers, doubles
and booleans. Arrays and other types remain standard Variant residuals. A new
inference can select later text or a changed field type even after an earlier
layout reached its budget. Changing an existing dataset's layout requires an
atomic migration; inferring a new layout alone does not migrate stored data.

Arrow's shredding casts permit conversions that change JSON types or values.
The adapter validates the selected values before invoking the kernel. A row
with an incompatible selected value remains entirely in the root residual.
This preserves correctness but sacrifices typed extraction for the other fields
in that row. The original text is not copied into an additional storage column.

JSON input uses Arrow's Variant builder. This avoids Arrow 58's JSON-string
parser converting unsigned integers above i64 through f64: the typed builder
encodes them as standard scale-zero decimals. SQL null and Variant null remain
distinct. Reconstruction uses the upstream unshred kernel and Variant-to-JSON
conversion. Tests additionally accept arrays built by upstream `json_to_variant`.

Nested view arrays are converted to offset arrays in the adapter before writing
to the current Lance version. This milestone adds no vendored Lance changes.
The existing baseline encoder patch remains part of the shared dependency.

Arrow 58.3's full validation rejects legal empty object keys in an unsorted
dictionary because it requires strictly increasing offsets. A pinned
`parquet-variant` patch changes this comparison to nondecreasing offsets and
adds a regression test that still rejects decreasing offsets. The source is
`beinan/arrow-rs` at `3b82666c986414727c20b05d1cbb1c63a71a5877`; its manifest
uses the released Arrow dependencies to avoid introducing a second Arrow type
universe. No Arrow source is copied into lance-context. Remove the pin after
upgrading to an upstream release with the fix.

## Delivery boundaries

The initial PR delivers the kernel/selection adapter and standard-array storage
interoperability tests. Persistent automatic layout migration, document access,
index integration, Python bindings, and equivalent baseline/Variant benchmarks
are follow-ups tracked in #832. The kernel milestone alone does not establish a
complete automatic document store or a performance advantage.

## Persistent Variant document adapter

`VariantDocumentStore` keeps `id` and one canonical `document` Variant column.
The schema stores the selection policy and selected layout; Variant value
reconstruction uses only the standard representation. Every append samples its
incoming documents. Sampled string paths have priority over numeric paths; new
paths precede retained paths of the same kind. The bounded remainder retains
old paths. This is a deterministic recent-batch policy, not a global workload
optimizer. Conflicting ancestor/descendant selections are never combined.

When the selected physical layout changes, old Variant batches are unshredded
and reshredded with Arrow kernels, then committed together with the incoming
batch in one overwrite. No partial migration becomes visible. A stale rewrite
fails rather than rebasing over later writes. Ordinary appends avoid rewrites.
Both paths encode batches targeting 4 MiB/1024 rows, allowing one oversized row;
caller-owned inputs, IDs and Lance buffers are outside this target. Rewrites
cost O(existing data), and retained snapshots keep their old data files.

The adapter disables cross-batch column buffering through Lance's existing
writer options. Otherwise many historical nulls followed by one large text
value can become an uncompressed null-padded constant page. Keeping the bounded
batches separate lets the existing large-singleton compression path apply.
This adds no Lance source change, but smaller pages can add metadata overhead;
the equivalent benchmarks include that cost.

A BTree index on ID is maintained after a durable data commit. Failure of that
maintenance is logged without reporting the committed append as failed;
`prepare_point_reads` explicitly retries and reports indexing errors. Whole reads
use upstream unshred/JSON conversion. Path reads project ancestor residuals and
the requested typed subtree. Physical projection resolves literal keys to field
IDs, including dots, quotes and empty keys. Arrow 58's direct shredded path
traversal can miss an ancestor residual when a typed child exists but is null
for that row. The adapter therefore unshreds the projected representation with
Arrow before invoking its `variant_get` kernel. Unrelated extracted text is not
part of that projection. This adds reconstruction work for the selected fields;
it does not require a Lance patch or a custom Variant decoder.

## Text search and Python

Text indexes target standard `document.typed_value...typed_value` children.
Preparation builds missing indexes without a data backfill. Native matches rank
first; remaining result slots use the same OR tokenizer over standard residual
values, with score zero and unspecified order. This fallback reads root and
nested residual fields, excluding text covered by native indexes. The current
Lance nested-index traversal supports plain ASCII identifier components here;
typed strings under other keys use the fallback scan too, preserving coverage
without copying their text into a separate column. It can scan all residuals
on rare/no-hit queries and does not provide global BM25 ranking across both sets.
Equality uses Arrow's path kernel over projected columns; scalar predicate
pushdown remains a performance limitation.

Python exports `VariantDocumentStore` alongside the baseline `DocumentStore`.
Their ordinary add/get/get_path/filter_eq/search and snapshot interfaces align.
The Variant wrapper delegates to the separate native Variant implementation;
it does not adapt the baseline's JSON-residual codec.
