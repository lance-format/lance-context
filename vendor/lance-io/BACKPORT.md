# Lance 14.0.2 compatibility patches

This directory starts from the official published `lance-io 14.0.2` crate.
Archive SHA256: `f5bbdcf364df034775ebbe50f01248e724146b647ee88276f2e204528e1c6621`.
Upstream commit: `40f15ed1b71b66950961b175829fdf17a763f108`.
The original Cargo manifest, VCS provenance, and Apache 2.0 license are retained.

Cargo.toml uses sibling paths for the patched crates and excludes Cargo's reserved
provenance filenames during sdist repackaging. Standalone regression manifests
select the same object_store revision as the workspace.

The multipart completion fix from lance-format/lance#9913 is now upstream and
is no longer backported. The upstream AimdHttpConnector and its new delete
behavior are retained. The local 36-part Azure completion regression remains;
it checks identical block-list retries, retained options, no part re-upload,
and the original error after exhaustion.

The existing throttle-classification correction remains: retry-exhaustion metadata
alone is not a throttle signal. Only explicit provider/status signals restart
the throttle layer. Context's SDK error-boundary regressions cover this.

The workspace and standalone regression graph retain object_store 0.14.2 at
1149230d06b87d2b458a1b6a886e0b8224498338. That revision includes the Azure Latest
block-list response-loss fix and the owner-scoped completion hook. Completion
retries retain the original serialized request and do not replay outer
publication or upload parts again.

Existing task lifetime changes remain in scheduler.rs, scheduler/lite.rs,
object_writer.rs, local.rs, object_store.rs, and object_store/providers/local.rs.
See ../lance-core/BACKPORT.md: physical IO/upload work retains ownership until
it actually completes or its state is destroyed.

Remove local compatibility patches only after equivalent upstream behavior
passes the context cancellation, multipart, and storage-error regressions.
