# Immutable application WAL merges

Applications storing immutable events can opt a `PreparedMerge` into
`use_append_only()` before `StorageBase::commit_prepared_merge`. The default
remains keyed replacement. No rollout schema or particular primary-key name is
required, and the application keeps its existing scheduler and worker routes.

The append mode reuses the rollout file publisher: new fragments and per-shard
generation watermarks commit together. It performs no base-key lookup or delete
join. Existing preparation bounds, schema validation, stale-generation checks,
ownership guards and WAL retirement still apply. A definitive version conflict
can reuse staged files if schema and shard coverage remain unchanged. An unknown
commit result returns to recovery rather than blindly retrying publication.

This makes **merge retries** idempotent, not arbitrary **ingest retries**. Sending
the same event again in a later generation or another shard can create another
physical base row. Readers must deduplicate identical immutable events by their
full identity before applying changes, evaluating lifecycle state, or paginating
logical event histories. A B-tree is not a uniqueness constraint; base-only scans
and ordinary compaction do not supply this application guarantee. Updates and
tombstones must continue using keyed replacement.

The existing preparation reducer may remove repeated keys within one bounded
merge. Callers must still handle duplicates across merge passes and shards.
Allowing duplicates also increases physical storage, index and read work. Keep
source-byte/read budgets independent of the number of logical events returned.

The regressions in `store_base/append_merge/tests.rs` cover explicit selection,
unchanged keyed behavior, stale work, a concurrent later flush, conflicting shard
commits, and recovery after base publication succeeds but WAL retirement fails.
