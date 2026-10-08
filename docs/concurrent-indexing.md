# Index preparation alongside WAL merge

`INDEX_PREPARE_TARGETS` selects rollout tables (comma-separated exact names or
`*`) whose IndexId tasks build files outside the table write lock. An empty
list keeps legacy indexing; generic and draining targets keep their existing
behavior. Both configured index types, BTree and ZoneMap, use this path.

The task takes a lease-backed preparation claim and calls Lance's
`execute_uncommitted` against a fixed snapshot. Handles opened in preparation
have a pinned authorizer that rejects manifest commits. Append merge can keep
publishing while the index scans keys and writes immutable files. Existing
general task concurrency and Lance memory limits still bound indexing; this
change does not add another indexing pool or increase parallelism.

Once files are ready, the task requests the next writer turn, then promotes its
exact task/preparation token into the existing table writer protocol. It never
cancels the active writer. `INDEX_COMMIT_WAIT_SECS` (default 120) bounds how long
prepared files retain a local task slot waiting for ownership. Expiry fails that
attempt and leaves the other writer untouched. Lease loss invalidates promotion;
a recovered task builds fresh files. Abandoned files remain unpublished.

Index and compact share the existing `compact-preparations` and
`compact-commit-ready` wire keys deliberately. There is one preparation slot per
table, independent of merge. Older masters already honor this commit request,
so a rolling upgrade preserves scheduling fairness without a queue migration.
Old full-lock maintenance may run during new preparation; validation still
rejects stale output. Persistent dedicated-owner guards retain their existing
opt-in requirements (`MAINTENANCE_CATCHUP_TARGETS`).

Publication opens a fresh handle inside the existing maintenance execution
scope, retaining owned-target fencing and ambiguous-commit recovery. Before
committing it checks the dataset URI/schema, every original source fragment,
and the same-name index metadata. A delete, rewrite, schema change, or concurrent
index replacement invalidates preparation. The CreateIndex transaction retains
the original snapshot read version, allowing Lance to check intervening commits
as well. Newly appended fragments remain unindexed and query planning scans them;
index publication must preserve merge watermarks and newly appended rows.

These exact stale-preparation validation failures use the ordinary bounded retry
budget, even when Lance wraps them as `Invalid user input`. The first two failed
attempts wait two seconds; repeated failures back off and eventually require
attention. Existing records incorrectly classified as data/configuration errors
are interpreted with the corrected policy on read, retaining their original
failure time and attempt count. This does not clear durable records or change
ownership. Arbitrary invalid input, corruption, and permission errors retain
their longer cooldown. During a rolling upgrade, older masters may still honor
the previously stored cooldown until the table is handled by a new master.

The progress watchdog counts completed data/index reads and upload parts,
including IO from child tasks. Response headers, metadata polling and failed IO
do not count. Lance 9 does not report fine-grained progress from its basic BTree
or ZoneMap trainer, so training callbacks alone are insufficient. Streaming
reads remain streaming and use no new payload buffer. Local/uring readers bypass
ObjectStore wrappers, so preparation also samples their handle-specific completed
byte counters. The preparation-specific store parameters isolate those counters
from other tasks using the same session. Storage commit progress
continues to use the existing execution scope. The
`master_index_phase_duration_seconds` histogram separates `prepare`, `commit_wait`,
and `commit`. Logs include task/target identities and phase transitions without
adding table labels to metrics.

Validation covers both index types and append commit orders, indexed/uncovered
reads, WAL watermark retention, stale source/schema/index rejection, forbidden
publication, etcd admission during an active merge, exclusive publication, and
replacement after preparation lease loss. Production rollout should compare
these phase durations and pending generations while ordinary jobs continue.
