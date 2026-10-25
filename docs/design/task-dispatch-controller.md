# Capacity-backed task dispatch (implementation under validation)

The demand informer discovers merge work once per fleet. The next layer replaces
per-master queue scans with exact task offers from a leased controller.

A scheduler reserves its existing local pool permit before publishing a leased
capacity ticket. Tickets describe kind and target filters and every existing
maintenance admission policy. General and compaction work advertise combined
shared capacity; compaction is advertised only while its additional permit is
reserved. Executors wait on the exact offer key. No controller means waiting,
not fallback scans. Capacity release wakes local scheduling promptly.

The controller keeps queue, running tasks, ownership and capacity in its bounded
List/Watch cache. Placement traverses finite 64-task by 64-ticket slices, including
heterogeneous capacity beyond the first page. Publishing an offer compares the
leader token, ticket lease/value, absent task reservation and queue revision.
The offer itself grants no storage ownership. The receiving executor uses the
original exact task admission/claim transaction, with ticket/offer/reservation
comparisons added. Accepted claims retain their own leases and storage fencing.
Losing leadership or the capacity lease does not cancel productive execution.

Native replicas do not run startup global recovery or completion-triggered
history scans. The leader observes orphaned running tasks and uses the original
requeue CAS, additionally fenced by leadership. History pruning runs on one
leader every five minutes, separately from placement, with a revision-consistent
paged snapshot and explicit memory bounds. Followers do not publish stale queue
depth. Existing standalone TaskStore clients retain their legacy behavior.

Dependency states are fetched by exact ID in batches of at most 64 and updated
by a task-event watch. Retained task history is not loaded into the informer.
Point-read revisions prevent older watch replay from regressing dependency state.
Missing/failed dependencies are sent to canonical failure propagation even when
another dependency is waiting. The dependency cache is capped at 10,000 IDs.

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

Before activation, validate actual shared-pool saturation, preparation sharing,
claim-response ambiguity, sustained dependency churn/cache eviction and very
large dependency chains; measure cold-target admission and cache byte bounds
against representative traffic. Chains above the dependency cache cap currently
fall back to exact executor dependency checks and need a bounded rejection
strategy before production. Review history retention races and stale-leader
reporting before release. Production etcd version remains unchanged.
