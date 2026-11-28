# Lance 14.0.3 compatibility patches

This directory starts from the official published `lance 14.0.3` crate.
Archive SHA256: `f2d220b0b8bcadec407aa4677dbfbcafee2fa69e56421d1ce07f156ad8c08503`.
Upstream commit: `c7dc014f1e722ad7dcc9a731991e4f33bb8852bb`.
The original Cargo manifest, VCS provenance, and Apache 2.0 license are retained.
The upstream 14.0.3 legacy Blob compatibility test uses the released v12 fixture
in `test_data/v12.0.0/blobs/legacy_fragment.lance` at the repository root.

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
- mem_wal/{observer,write,wal}.rs expose optional put-phase observations and the
  last observed durability/index cursors of unfinished visibility waits. Observer
  callbacks default to no-ops; no admission, scheduling, persistence or ACK
  semantics change. Concurrent puts have independent observation tokens, and
  dropped futures are reported as abandoned rather than successful. This is
  diagnostic instrumentation, not a write-timeout fix.

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
