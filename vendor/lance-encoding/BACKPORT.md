# Lance 13 owned preparation task propagation

This directory is the published `lance-encoding 13.0.0` crate, archive SHA256
`3c04a91e545b505b5e6ec98854c7eadc56689cbc072f138dd26805ef0095e81a`, upstream commit
`0923349baa19fd9af776b7d4626dc758dc610c80`. Original workspace manifest and
Cargo VCS provenance are retained; the upstream Apache 2.0 license is included.
No version or file-format upgrade is introduced.

The runtime helper is documented in `../lance-core/BACKPORT.md`. Only these
Rust source files differ from the published archive:

- `src/decoder.rs`: scope batch decoder and scheduler children.
- `src/array_encoding/logical/list.rs` and `primitive.rs`: scope existing legacy encoding and indirect-scheduling children.
- `src/array_encoding/physical/binary.rs`, `dictionary.rs`, and `packed_struct.rs`: scope existing decode children.

`Cargo.toml` excludes Cargo's reserved provenance filename on repackaging and
uses explicit sibling paths for the patched dependencies, so source archives
rebuild the same code. Maturin regenerated the standalone `Cargo.lock` to
resolve those local paths; the root workspace lock keeps the same dependency
versions. Other upstream source files are byte-identical.

These helpers are active only inside an owned preparation task group. Their
default behavior and returned task-handle types are preserved. No shared global
service is joined as if it belonged to one request. The regression tests live
in context core's `staging_cancellation_tests.rs`; they exercise real Lance
scanners, failure while a later fragment is outstanding, caller cancellation,
CPU children, shared request tokens, and physical scheduler IO cleanup. Actual
production RSS/throughput still require representative fixed-image validation.

This is a local compatibility patch pending upstream review. Remove it when a
compatible published release provides equivalent lifetime tracking and the same
regressions pass against that release.
