# Lance 14.0.3 compatibility patches

This directory starts from the official published `lance-encoding 14.0.3` crate.
Archive SHA256: `4d569708f3ccb1a0aae3d236c92a79086a7e7ffc5c705dcf7871d4e3971625d0`.
Upstream commit: `c7dc014f1e722ad7dcc9a731991e4f33bb8852bb`.
The original Cargo manifest, VCS provenance, and Apache 2.0 license are retained.

Cargo.toml uses sibling paths for the patched crates and excludes Cargo's reserved
provenance filenames during sdist repackaging. Standalone regression manifests
select the same object_store revision as the workspace.

The existing scoped child-task propagation is retained in src/decoder.rs and
array_encoding/{logical/{list,primitive},physical/{binary,dictionary,packed_struct}}.rs.
It uses the task group documented in ../lance-core/BACKPORT.md and preserves
existing cancellation and handle behavior. The remaining upstream source,
including the 14.0.2 sliced-validity fix, is unchanged except for the changes
described below.

`encodings/logical/primitive/structural_encoder.rs` lets a single variable-width
value larger than 32 KiB fall through constant-page selection to the existing
compression strategy. A constant page stores its scalar raw; for a singleton
there is no repetition to eliminate. This prevents DocumentStore backfill from
expanding compressed 20 MiB text into uncompressed pages when streaming bounded
batches. Existing multi-value constants (including null-padded large values),
small scalars, and the on-disk format are unchanged. Regressions cover compressed
UTF-8/binary singletons with both offset widths and DocumentStore rewrite/reopen.

## Constant-page seeking

Backport Lance PR [#10339](https://github.com/lance-format/lance/pull/10339),
commit `29affc511d4ce86819a3f7b6a9a7bd88f584f33f`, without code changes:
`constant.rs` and its three expanded constant-layout regression cases in
`structural_encoder/tests.rs`.

Scalar pages without repetition levels jump directly to the requested row.
Contiguous output ranges are handled as level slices, and decoded arrays avoid
encoder-only statistics. List pages still walk repetition levels to locate rows.
This changes reading only; persisted bytes, writer behavior, and query semantics
are unchanged.
