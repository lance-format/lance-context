# Upgrading lance-context to Lance 14.0.2

Rust Lance packages and optional Python `pylance` use 14.0.2. Python's matching
`lance-namespace` range is 0.13.x. Arrow remains at 58, DataFusion at 54, and
the Rust object_store SDK at 0.14.2. The SDK's existing Azure completion patch
is retained.

The vendored packages start from the official 14.0.2 archives. Each package's
`BACKPORT.md` records its archive checksum, upstream commit, and remaining local
changes. The multipart completion correction from Lance #9913 is now upstream;
its duplicate backport is removed. The existing error-classification and task
lifetime corrections remain necessary.

The MemWAL API renames `force_seal_active` to `force_freeze_active`. Its new
index-plugin API keeps index specifications with each frozen generation. The
context integration retains that generation-specific state across failed flush
attempts, along with the exact WAL receipt and prepared publication. An absent
retired generation alone does not confirm the outcome of an uncertain publish.
Existing scalar-index coverage masks are also retained.

Lance 14's column rewrite now preserves join-key columns itself. Annotation
publication validates the original ID column and index coverage without the
Lance 13 file-mapping repair. Only labels, status, and score may be rewritten;
fragment membership and unrelated physical columns must remain unchanged.

Staging budgets and publication batch sizes are unchanged, and the experimental
Lance 2.3 file format is not selected. Indexed maintenance now persists the
native rewrite and its remapped index files using existing descriptor formats
4/5, including for immutable ingestion tables. All catalog voters and publishers
must support this maintenance policy before it is used; older binaries reject
that path. This does not establish a mixed-version rolling-upgrade guarantee.
The preparation instructions in
[the Lance 13 upgrade notes](lance-13-upgrade.md) still apply to generations
written before field-ID preservation.

Lance 14 does change the native BTree index row domain: new version-1 segments
store physical row addresses, while Lance 13 version-0 segments store row IDs.
Stable dataset row IDs therefore do not by themselves preserve a new BTree's
coverage after compaction. A rewrite must carry the native index remap before
publication; otherwise Lance drops the replaced fragments from index coverage.
Likewise, an incremental index build must not combine old row-ID segments with
new row-address segments under the same logical index.

The upgrade is not ready to deploy until managed compaction retains those
remaps through prepared publication and the existing-index transition is
validated. The current compaction regression intentionally continues requiring
ID index coverage on the replacement fragment. Falling back to a scan is not
an acceptable substitute for that coverage guarantee.

Validation must cover the existing real Lance flush failure and replay-boundary
regressions, concurrent annotation publication, indexed reads, task cancellation,
SDK completion retries, and Python source-distribution rebuilds. Performance
results from Lance 13 remain results for that version until a new binary is
measured with the same workload.

Related ingestion work: [#378](https://github.com/lance-format/lance-context/issues/378).
