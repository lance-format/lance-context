# Lance 14.0.3 compatibility patches

This directory starts from the official published `lance-core 14.0.3` crate.
Archive SHA256: `3ddcb680ecedb6b6d89d84ba1d272d0b43f64392b693f37d8bf49f25d130bd3d`.
Upstream commit: `c7dc014f1e722ad7dcc9a731991e4f33bb8852bb`.
The original Cargo manifest, VCS provenance, and Apache 2.0 license are retained.

Cargo.toml uses sibling paths for the patched crates and excludes Cargo's reserved
provenance filenames during sdist repackaging. Standalone regression manifests
select the same object_store revision as the workspace.

The existing opt-in task group in src/utils/tokio.rs is retained. It joins actual
CPU, async, and ordinary blocking children, with tokens for requests in shared
services. Scope membership ends after underlying work and its captures are
destroyed, including cancellation before first poll. Unscoped behavior, pool
selection, and task-handle semantics remain unchanged. The owning join future
must be driven to completion. This is lifetime accounting, not a hard RSS bound.

Call-site patches in lance, lance-encoding, and lance-io are required together.
Context's staging cancellation and merge-budget regressions exercise them.
Remove these patches when a published release provides equivalent ownership
and the same regressions pass.

`NativeMemoryPool::try_reserve_cache` distinguishes optional cache retention
from required decoder/index allocations. Its default delegates to the existing
`try_reserve`; hosts may override it to decline warming without reclaiming live
backing. The allocation-kind wrapper forwards this distinction. Context uses
the hook only for disposable immutable membership markers; required processing
continues to use the original admission methods.
