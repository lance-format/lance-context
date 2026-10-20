# Durable worker merge admission (integration in progress)

Worker count-triggered and timer-triggered merges previously wrote without a
persistent execution record. A master could observe a hung request but could
not establish what that writer might still commit. This path records those
executions before storage work and reuses the owned executor's useful-progress
watchdog, conditional commit permits, cancellation and storage fencing.

## Behavior

`MERGE_FENCE_LEGACY=true` enables the protocol for new rollout and generic worker
merges, independently of the static native-ownership list. It requires a stable
`MERGE_EXECUTOR_ENDPOINT`, private etcd configuration, and positive
`ROLLOUT_MERGE_CONCURRENCY`. Master fanout remains serial. Ordinary HTTP merges,
count triggers and cleanup timers use the same process slots and executor.

A background trigger tries a slot before spawning; it never waits for capacity
or etcd on the flush sweep. Per-table admission is coalesced. The detached
executor exists before admission IO, so a disconnected HTTP caller cannot
abandon a writer. Its slot remains occupied until its storage leaves drain or
are fenced. An ambiguous terminal record is retained for master recovery.

Admission atomically persists the execution, exclusive owner and coalesced
request. It respects canonical claims, catchup ownership, prepared-commit
priority, admission closure and persisted failure budgets. A terminal worker
cannot delete an owner being transferred to a waiting master claim.

The master inventories worker records as well as maintenance records, including
records whose original worker is no longer configured. It observes actual
execution progress before cancellation, reconciles ambiguous commits through
the existing storage barrier, and never treats an empty endpoint list as a
successful catchup. New-table demand can enqueue fenced worker fanout without
adding a static owned target.

Worker HTTP compaction is refused in this mode; tracked master maintenance must
handle it. This prevents an untracked worker rewrite from bypassing merge
ownership.

## Persisted protocol and startup

The private-etcd `worker-merge-admission-mode` key is persistent and unleased.
Once activated, disabling the initiation flag on a restarted participant does
not restore legacy writes. An unknown newer marker is rejected, including when
activation is requested. The master resolves this mode at startup. Workers
resolve it independently of startup/flush; unresolved metadata defers maintenance
and cannot authorize a legacy fallback. Participants without etcd and without
the opt-in keep standalone legacy behavior.

A worker that has resolved legacy mode retains that mode for its incarnation.
Consequently, initial activation requires a coordinated transition covering
all old worker/master instances and external helpers. A marker, a Pod restart,
or an empty execution inventory does not prove that historical writers are
quiescent. Removing the shared etcd namespace/configuration or deploying an old
binary is not a supported rollback after activation.

## Remaining native handoff requirements

This implementation does not certify the initial transition, dynamically add
native ownership, or authorize native writes solely because admission closed.
Before production activation, finish and validate:

- Complete old-writer coverage, including helpers and master compact/index/repair.
- Durable native enrollment after recorded writers finish or are storage-fenced.
- Automatic new-hot-table selection and bounded continuous native catchup.
- Combined staging of worker admission, tracked master maintenance and native
  execution, followed by measured production handoff.

Keep this integration disabled until those requirements are implemented. The
current production helpers remain necessary until the handoff is demonstrated.
