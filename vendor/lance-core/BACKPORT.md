# Lance 14.0.2 compatibility patches

This directory starts from the official published `lance-core 14.0.2` crate.
Archive SHA256: `874fd0494c0b3357a258c0614bbaff3d1654382613ba4cf76a758c399eed7f15`.
Upstream commit: `40f15ed1b71b66950961b175829fdf17a763f108`.
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
