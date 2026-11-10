# Lance 14.0.2 compatibility patches

This directory starts from the official published `lance-encoding 14.0.2` crate.
Archive SHA256: `acc812903e6daaa4ab2a925350a96ab04d06a83deeceac0f026355a4dccbfd0a`.
Upstream commit: `40f15ed1b71b66950961b175829fdf17a763f108`.
The original Cargo manifest, VCS provenance, and Apache 2.0 license are retained.

Cargo.toml uses sibling paths for the patched crates and excludes Cargo's reserved
provenance filenames during sdist repackaging. Standalone regression manifests
select the same object_store revision as the workspace.

The existing scoped child-task propagation is retained in src/decoder.rs and
array_encoding/{logical/{list,primitive},physical/{binary,dictionary,packed_struct}}.rs.
It uses the task group documented in ../lance-core/BACKPORT.md and preserves
existing cancellation and handle behavior. The remaining upstream source,
including the 14.0.2 sliced-validity fix, is unchanged.
