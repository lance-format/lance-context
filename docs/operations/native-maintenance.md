# Maintenance beside persistent native catch-up owners

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
