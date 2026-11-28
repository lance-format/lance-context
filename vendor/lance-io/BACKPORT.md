# Lance 14.0.3 compatibility patches

This directory starts from the official published `lance-io 14.0.3` crate.
Archive SHA256: `6a2ca7e47930a87fecba7b6ddfaa3c3d9caa20e06be44a5cfe0558fbb5c25927`.
Upstream commit: `c7dc014f1e722ad7dcc9a731991e4f33bb8852bb`.
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
6e5253a3a9c0565c2b5b07ea304023d23422f8a1. That revision includes the Azure Latest
block-list response-loss fix and the owner-scoped completion hook. Completion
retries retain the original serialized request and do not replay outer
publication or upload parts again.

The SDK also sends a singleton deletion through native Azure DELETE, retaining
the existing credential, endpoint, retry and error-classification machinery.
Multi-object requests still use Blob Batch. Context uses this path to retry only
unresolved files in an already sealed failed Azure collection page, under its
existing concurrency, fencing, memory and publication-pressure limits. The SDK
does not retry a failed batch automatically as individual requests.

Existing task lifetime changes remain in scheduler.rs, scheduler/lite.rs,
object_writer.rs, local.rs, object_store.rs, and object_store/providers/local.rs.
See ../lance-core/BACKPORT.md: physical IO/upload work retains ownership until
it actually completes or its state is destroyed.

HTTP metric connectors initialize throttle and retryable-response counters at
zero for all counted status codes before use. This gives collectors a baseline
for the first response without resetting counters when a connector is recreated.

Remove local compatibility patches only after equivalent upstream behavior
passes the context cancellation, multipart, and storage-error regressions.
