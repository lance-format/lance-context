# Compaction preparation alongside WAL merge

A large compaction can spend many minutes reading and encoding old fragments.
Holding the table writer lock during that work prevents WAL publication even
when the merge only appends new fragments. `COMPACTION_PREPARE_TARGETS` selects
rollout targets (comma-separated names, or `*`) for two-phase compaction. The
default empty list preserves the previous path; generic stores and draining
targets do not use preparation.

1. Claim the durable Compact task and a leased `compact-preparations/<target>`
   key. Do not acquire the shared table lock, merge claim, or execution record.
   The normal task lease/recovery protocol remains unchanged.
2. Read a fixed snapshot and write immutable, unpublished rewrite files. A
   pinned commit authorizer rejects any manifest publication during this phase.
   Actual storage/processing progress drives the idle watchdog; claim loss
   cancels preparation. Neither cancellation nor process loss changes live data.
3. Once files are ready, acquire the normal table lock and merge-claim key using
   one transaction that verifies the original task/preparation tokens and all
   current execution, table, and dedicated-owner predicates. A lost response
   may only be adopted after verifying this task's unique token.
4. Reconcile prior maintenance, then open a fresh handle in the existing
   maintenance execution scope. Owned targets retain version fencing and
   ambiguous-commit recovery. Unowned targets retain their existing commit
   protocol. Validate source fragments and schema, then call Lance's
   `commit_compaction` with the original RewriteResult read versions. Fragment
   reservations, index remapping and final publication all occur in this phase.
5. Finish the task and release its own keys. A failed preparation must never
   delete another task's merge claim or execution record.
6. After durable completion, refresh statistics best-effort using a fresh
   handle outside publication authority. Waiting for `stats-writer` does not
   retain table ownership. The task keeps its local capacity permits until
   refresh ends, so slow statistics cannot spawn unbounded background work.
   A crash here can omit the statistics update; the next scan refreshes table
   counts, but the best-effort compaction counter can miss that compaction.

An append-only merge does not alter the selected source fragments and can
complete during preparation. An update, delete, overlapping rewrite, or schema
change may invalidate the prepared output; stale files must not resurrect old
rows. Lance retains its intervening-transaction conflict checks in addition to
the source-fragment preflight. WAL merged-generation watermarks remain part of
the latest manifest and must survive either commit order.

Compaction capacity is reserved before claiming, together with the existing
general task budget. Index/repair and WAL polling remain independent. The
`master_compaction_phase_duration_seconds` histogram separates `prepare`,
`commit_wait`, `commit`, and `stats_refresh`; task/target identities appear only
in logs. Task completion does not wait for the best-effort statistics refresh.

Prepared output is deliberately not a new durable publication protocol. A
process lost before commit may leave unreferenced immutable files and its task
is requeued after lease expiry. Files are never considered committed because
encoding completed. A newly selected dedicated owner still excludes compaction
publication. Preparation does not override maintenance dependencies or guards.
`COMPACTION_COMMIT_WAIT_SECS` (default 120) bounds how long ready output retains
local capacity while waiting for a writer. Exhaustion fails only this compact
attempt, releases its preparation claim, and leaves the progressing writer
untouched. A later compact replans; no prepared output is treated as committed.
This admission budget is separate from the progress-based execution watchdog.

Deployment can mix old and new masters: queue/task records retain their schema,
and an expired preparation claim can be safely retried through either path.
Disabling the option affects newly claimed tasks; a live preparation can finish
through the same fenced publication path. Do not manually delete ownership or
restart a progressing compaction to toggle this option.
