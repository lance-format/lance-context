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
or etcd on the flush sweep. Due demand is also published through a separate
four-request metadata bound, so occupied execution slots do not hide a hot table
from the master. Per-table admission and demand are coalesced. The detached
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
adding a static owned target. Until automatic native routing is enabled, an
existing append wildcard applies only to explicitly owned tables; new targets
continue through fenced worker fanout instead of entering an ineligible local
execution path.

Worker HTTP compaction is refused in this mode; tracked master maintenance must
handle it. This prevents an untracked worker rewrite from bypassing merge
ownership.

Concurrent merge progress can invalidate several read snapshots in well under
a second. Consistent reads now distinguish that progress from repeated missing
data at the same base version: up to 16 snapshot attempts are allowed, while
three missing-file failures without a new base commit still stop the request.
No additional attempt starts after the two-second retry budget is consumed;
that budget does not cancel an otherwise valid long scan. Reads never return
partial results or discard required WAL generations to satisfy this bound.
Rollout point reads apply that recovery to their WAL fallback while preserving
the cheap cached-base hit and WAL-only semantics. Generic filtered/point reads
keep their cached-view behavior on success or ordinary miss, but recover an
actual missing-file race through a fresh consistent snapshot. Pinned reads do
not advance; invalid filters are not retried as missing storage.

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

## Automatic native routing (validation in progress)

After the initial legacy-writer transition is verified, explicitly enabling
`MASTER_LOCAL_MERGE_AUTO=true` with `MASTER_LOCAL_MERGE_TARGETS=*` routes new
rollout/generic demand through resident masters without changing a table
allowlist. Worker protocol activation also enables fenced master maintenance.
Automatic routing requires that persisted protocol, stable shard IDs, positive
merge capacity and the existing shared memory budget. It creates no Pods or
second queue. Draining targets and datagen tables remain excluded.

The canonical task claim excludes new background admission. A progressing
background executor is reconciled to completion before native catchup starts;
uncertain storage writes retain their fence until recovery succeeds. Native
catchup uses bounded passes and publishes continuation demand before releasing
the claim, so another master can resume it. Existing explicit append-protocol
targets retain that optimization; newly selected targets use native sealed-WAL
catchup, including generic updates.

Neither automatic routing nor the activation marker proves old writers are
quiescent. Before enabling this in production, finish and validate:

- Complete old-writer coverage, including helpers and pre-upgrade master work.
- Combined staging of worker admission, tracked maintenance and automatic
  resident execution with ongoing ingestion, progress stalls and restart.
- Measured production handoff under the actual traffic and memory limits.

The source paths and regression tests are implemented but final validation is
still running. Keep automatic routing disabled until the initial transition
is proven. Existing production helpers remain necessary until the handoff is
demonstrated.
