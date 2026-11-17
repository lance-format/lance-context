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
