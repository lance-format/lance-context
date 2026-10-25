# Capacity-backed task dispatch (implementation under validation)

The demand informer discovers merge work once per fleet. The next layer replaces
per-master queue scans with exact task offers from a leased controller.

A scheduler reserves its existing local pool permit before publishing a leased
capacity ticket. Tickets describe kind and target filters and every existing
maintenance admission policy. General and compaction work advertise combined
shared capacity; compaction is advertised only while its additional permit is
reserved. Each pool retains one lease and exact-key Watch across successful
claims. Acceptance atomically removes the capacity ticket, offer and reservation;
the next locally reserved slot re-advertises capacity on the same lease. A busy
pool has no ticket even while its session stays alive. Re-advertisement compares
ticket/offer absence and refreshes the complete kind/target policy; the receiver
ignores buffered Watch events older than that advertisement. Faults, rejected
offers and drain close the capacity session, and the next attempt opens a new
exact snapshot/Watch without a fleet scan. Executors wait on the exact offer key. No controller means waiting,
not fallback scans. Capacity release wakes local scheduling promptly.

The controller keeps queue, running tasks, ownership and capacity in its bounded
List/Watch cache. Placement traverses finite 64-task by 64-ticket slices, including
heterogeneous capacity beyond the first page. Pagination advances over raw
records, so malformed or already-occupied tickets cannot hide later capacity. Publishing an offer compares the
leader token, ticket lease/value, absent task reservation and queue revision.
The offer itself grants no storage ownership. The receiving executor uses the
original exact task admission/claim transaction, with ticket/offer/reservation
comparisons added. Accepted claims retain their own leases and storage fencing.
Losing leadership or the capacity lease does not cancel productive execution.
When a previously yielded legacy HTTP response arrives, it first joins its
pool's fair semaphore queue, then notifies that pool's idle receiver to withdraw
its capacity reservation. Otherwise a receiver could hold the last slot while
waiting for a task that depends on the returning execution finishing. Responses
still in flight do not signal, and unrelated pools keep their capacity. A resume
guard clears the notification state on reacquisition or cancellation. A claim
already in flight is joined rather than canceled.
Only one unaccepted offer per target is published at a time. Consumption removes
that restriction, so preparation sharing still permits compaction and merge
to execute concurrently under their existing canonical admission rules.

Native replicas do not run startup global recovery or completion-triggered
history scans. The leader observes orphaned running tasks and uses the original
requeue CAS, additionally fenced by leadership. History pruning runs on one
leader every five minutes, separately from placement, with a revision-consistent
paged snapshot and explicit memory bounds. Followers do not publish stale queue
depth. Existing standalone TaskStore clients retain their legacy behavior.

Dependency evaluations are retained per queued dependency chain and removed
with that queue entry. Each placement step reads at most 64 IDs; longer chains
yield between chunks and finish at one etcd revision. Related task events mark
the evaluation for refresh; unrelated events and older watch replay do not cause
new reads. No global dependency-cache eviction or oversized-chain fallback is
needed. Retained task history is not loaded into the informer. Missing/failed
dependencies reach canonical failure propagation even when another dependency
is waiting. The executor also batches its authoritative dependency reads by 64.

The new election key is `demand-controller/leader-v2`: a discovery-only old leader
cannot strand new capacity tickets during rolling migration. Old replicas still
perform old scans and may win a canonical claim; new receivers must reject their
stale offers safely. The old and new discovery cohorts may temporarily overlap,
so total load reduction requires finishing the replica replacement.

## Validation and remaining release gates

Isolated real-etcd tests cover exact/duplicate claims, expired offers, stale
leaders, abrupt leader loss, an actual waiting receiver, productive-owner
preservation and orphan recovery, heterogeneous tickets beyond one page, and
blocked-candidate RPC amplification. With 50 tickets, 96 blocked tasks and one
runnable tail, placement uses one offer transaction and repeated blocked passes
use zero Range/Txn calls. This excludes ticket registration, lease traffic,
canonical claims, startup snapshot and task completion. It is not production
throughput evidence.

A real-etcd receiver test claims 16 successive tasks while changing kind filters:
it creates exactly one capacity lease and one Watch stream, plus the 16
independent canonical task leases. After every acceptance all dispatch keys are
absent, so a full pool cannot receive another offer. Canceled Watch and revoked
capacity-lease tests reopen the receiver and preserve running tasks.

Actual scheduler tests verify that a one-slot shared pool completes indexing
while the compaction budget is occupied, and that resident merge memory pressure
does not stop the general pool. A 10,001-dependency fixture verifies bounded
chunks, no repeated reads on unrelated events, and eventual refresh on relevant
progress. Mid-snapshot dependency changes trigger a new finite sweep.

Startup initialization survives an actual transport outage during final lock
cleanup; genuine initialization errors still fail. The old guard stops renewal
and expires. This avoids replaying already completed initialization.

Real-etcd tests verify concurrent prepared compaction and merge claims after
pending-offer consumption. They inject response loss after the actual claim
transaction: a committed original token is recovered; a foreign owner remains
untouched. Ticket cleanup does not revoke an accepted task lease. A stale leader
cannot prune history, and dependencies already present in the history snapshot
are retained.

Before activation, measure cold-target admission and cache byte bounds against
representative immutable-image traffic. The pre-existing history-pruning race
with concurrently created dependencies is not solved by centralization: a new
chain can reference terminal history selected for deletion after the pruning
snapshot. Missing dependencies fail explicitly; no task may bypass them. This
requires a separate atomic reference/retention protocol before claiming history
retention is race-free. Production etcd version remains unchanged.

## Dispatch observability

`master_task_pool_in_use{pool,state="waiting_for_offer"}` counts capacity that
is advertised or waiting, without table ownership. `state="claiming"` counts
actual assigned-task admission; `state="running"` counts
execution. All pool/state labels have fixed scheduler values.
`master_controller_offer_wait_seconds{pool}` records time until an offer is
received, including idle periods. It is not task queue latency. The existing
claim-phase duration excludes that wait. Use `master_task_schedule_to_bind_seconds`
for a task's actual enqueue-to-claim latency.

## End-to-end load qualification

An initial two-master/two-writer comparison against the discovery-only parent
reduced Range+Txn calls only 4.75% (10,199 to 9,714), while LeaseGrant calls rose
from 366 to 639 and Watch stream starts rose from 24 to 272. That implementation
created a capacity session for each task and was not a demonstrated fleet-load
fix. The persistent pool sessions above address that measured overhead; repeat
the same traffic with all unary RPCs and stream starts reported separately before
claiming an end-to-end improvement. Stream starts are not message counts or an
estimate of total etcd CPU. The fixture is small local traffic, not production
throughput evidence, and one earlier run had a 7.35-second list-read outlier.

A CI failure exposed the last-slot response/dependency cycle above. The regression
test now deterministically waits for the idle reservation before releasing the
old response: it fails before the fix and completes afterward using the original
timeout. A separate test checks pool isolation and canceled-resumer cleanup.
`master_controller_idle_reservation_yields_total{pool}` counts reservations
withdrawn for ready executions. The receiver RPC-count test runs in the existing
exclusive etcd step, so concurrent suites cannot contaminate global counters.
