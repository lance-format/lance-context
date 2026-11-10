# Lance 14.0.2 compatibility patches

This directory starts from the official published `lance 14.0.2` crate.
Archive SHA256: `6895913cb87c0ca8140d13a3c980ceb5b6c67d95f5e96093a3e157347021c61b`.
Upstream commit: `40f15ed1b71b66950961b175829fdf17a763f108`.
The original Cargo manifest, VCS provenance, and Apache 2.0 license are retained.

Cargo.toml uses sibling paths for the patched crates and excludes Cargo's reserved
provenance filenames during sdist repackaging. Standalone regression manifests
select the same object_store revision as the workspace.

The existing scoped read-task propagation is retained in dataset/fragment.rs
and io/exec/{scan,take,pushdown_scan,filtered_read,utils}.rs. BlobSource moved
to dataset/blob/file.rs upstream; its per-request ownership token moved with it.
Shared queues remain independent of any one caller. See ../lance-core/BACKPORT.md.

Existing correctness patches are retained as well:

- index/scalar_logical.rs restricts each index segment's results to its committed
  fragment coverage before combining them (#706).
- dataset/mem_wal/manifest.rs uses a successor probe for a current version hint,
  and validates conditional updates against each CAS attempt.
- dataset/mem_wal/{write,observer,test_util}.rs and memtable/flush.rs retain failed
  generations and their prepared publication, reconcile uncertain publication
  against exact attempted versions, and never advance replay past a failed
  predecessor (#824). The finite-attempt/idle-retry behavior is preserved.
- memtable/batch_store.rs and wal.rs preserve each store's WAL receipt through
  coalesced flushes (#827). A later writer-global position is not its receipt.
- dataset/rowids.rs admits the commit handler's manifest-record identity into
  the bounded row-ID mapping cache when a physical ETag is unavailable. The
  cache key keeps logical identity and physical ETag in separate fields, along
  with version and deletion mode. With neither token, mappings remain private.
  This avoids rebuilding managed-table mappings without adding object-store
  HEAD requests or changing persisted data. Recreate/deletion regressions and
  a real RF3 publication test cover generation isolation and repeated reuse.

The 14.0.2 index plugin/schema API is retained: each queued generation carries
its own index specifications. The close cancellation token remains effective
during retry backoff. Context calls the renamed force_freeze_active API.
Upstream's new already-published shortcut is not used: a retired generation
alone is insufficient evidence of this prepared publication's selected cut.
The corresponding test reuses the prepared publication and checks its receipt.
The upstream stale-covered-position replay test explicitly installs its old
cursor after writer shutdown: the retained per-store receipt now closes the
original writer-bookkeeping window. It also checks the correct normal cursor.

These patches remain necessary for context's existing flush-retry, replay-boundary,
publication-loss, scalar-index, and task-lifetime regressions. They do not add a
new persisted format. Remove only when upstream semantics and tests cover them.
