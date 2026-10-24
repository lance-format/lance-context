# Maintenance beside persistent native catch-up owners

For a master assigned only to Compact, IndexId and Repair, set
`MASTER_CLAIM_WAL_MERGES=false`. The default is `true`. This excludes queued
MergeWal work from its general, dedicated merge and resident merge pollers;
it leaves that work queued for other masters. It does not stop enqueue sweeps,
cancel accepted operations or change ownership/fencing. Set enqueue intervals
separately when the executor should not generate new periodic work.

`MERGE_WAL_INTERVAL_SECS=0` only disables periodic enqueueing, not claiming.
`MERGE_WAL_CONCURRENCY=0` shares the general pool and also does not disable
claiming. Changing the claim setting requires a new process; retiring an older
process still requires its supported drain and storage-quiescence evidence.

`MAINTENANCE_CATCHUP_TARGETS` is an explicit comma-separated list of rollout
targets. It defaults to empty and does not support `*`. A target must also be
owned by the merge protocol, non-generic, and not draining. Enable it only
when every dedicated publisher for that target uses the common task claim,
merge claim, target lock, and fenced Catchup execution protocol.

For opted-in targets, a persistent `catchup-active` identity no longer
permanently excludes Compact and IndexId. Immutable compaction preparation
may overlap an active native merge; publication and index changes still take
the same exclusive writer ownership. Each admission CAS compares the exact
observed dedicated identity. Compaction promotion also requires the identity
recorded at preparation admission; a replacement owner invalidates promotion.
Neither maintenance completion nor cleanup removes the publisher identity.
Ordinary MergeWal and Repair do not receive this exception.

With `COMPACTION_PREPARE_TARGETS` enabled, compact file preparation can overlap
native progress, then wait for the existing pass to release its claim. The
commit wait remains bounded by `COMPACTION_COMMIT_WAIT_SECS`; it does not cancel
the native writer. Without preparation, Compact takes exclusive ownership
before rewriting. IndexId always takes exclusive ownership. Old/unverified
publishers, generic targets and draining targets retain the original guards.

Validate against a persistent native supervisor before production enablement:
keep ingestion active; enqueue Compact and IndexId; observe successful
maintenance publications and subsequent native drainage without clearing the
active identity. Verify old/new WAL watermarks, all test payloads, index query
results and memory bounds. A test with ordinary merges alone does not cover
this ownership lifecycle. Gate a canary on current native executable/image and
supervisor identity, then verify a maintained table continues draining.

This enables eligibility, not a guarantee that maintenance wins every writer
race. A continuously reclaiming publisher may still exhaust a compact's bounded
publication wait. Measure that contention and keep the existing merge/drain
recovery running; do not reset claims or stop progressing writers to force a
maintenance slot.

## Validation-only Repair

`REPAIR_VALIDATE_ONLY_TARGETS` is a separate comma-separated list of exact
rollout table names, empty by default. It does not accept `*`, generic or
datagen targets. An enrolled target must be owned and non-draining. Verify
that every publisher uses the fenced claim/target-lock protocol described
above before enabling this option; an old Pod being absent or a catch-up
record being missing is not that proof.

For these targets, Repair checks current data and deletion file references
with sequential metadata HEAD requests. Each successful reference advances
`work.references_validated` and the existing progress sequence; waiting or
transport errors do not. It succeeds without changing the base manifest only
when all references exist and the base version remains unchanged. This checks
reference existence, not payload integrity or every index file. Missing files,
transport errors or a concurrent version change fail the task explicitly;
restore missing objects or resolve the reported storage error before retrying.
The normal failure budgets remain in effect. No fragments, WAL objects or
catch-up identities are deleted, and unrelated failure budgets are not reset.

The setting also permits this Repair to claim beside a retained catch-up
identity. It still takes exclusive task/merge/target ownership, compares the
exact observed catch-up value in the final etcd transaction, and cannot
prepare concurrently with a writer. A lost process lease preserves persistent
execution ownership; the replacement reconciles and fences it before work.
The original Repair reaches Done through normal task completion, after which
its dependent tasks may run. Nothing manufactures successful task records.

Validation-only behavior remains selected if the catch-up marker later
disappears. Disabling the option restores the pre-existing Repair behavior,
which can delete fragments naming missing files, so do not disable it merely
to retry a failed validation. This option does not retire the historical
catch-up owner or prove all of its successors have stopped writing.
