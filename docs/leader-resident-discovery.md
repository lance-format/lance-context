# Leader-owned resident WAL discovery

The durable task store remains authoritative. Queue entries, task identity,
claims, execution epochs and commit fencing are not replaced by an in-memory
queue. The demand controller already reconstructs its view with a paginated
single-revision snapshot and resumes each prefix Watch from its own cursor.
Executors receive targeted offers and acquire task ownership with transactions.

Resident WAL discovery now runs within that elected controller's lifetime.
Previously every resident-capable master read the shared discovery gate every
five seconds, even though only one could reserve a metadata-probe batch. With
100 such masters, idle gate checks alone generated about 20 Range RPCs/second.
The new loop has one gate reader, independent of replica count. This estimate
covers this discovery path only, not total etcd traffic or timeout reduction.

## Policy and recovery

* Existing leased membership records advertise resident selection, storage root,
  shard set and probe interval. The leader derives their union from its existing
  snapshot/Watch; there is no additional membership scanner or watcher.
* Equivalent policies are deduplicated. Explicit targets are discoverable even
  when the leader itself cannot execute resident merges. Wildcard discovery
  restores known table names from the persisted stats dataset even when the
  leader's scanner is disabled. A name-only projection is bounded to 100,000
  names / 8 MiB of name bytes, cached for five minutes and kept for one root
  at a time. Observed stats counts are never evidence of actual pending WAL.
* The existing persisted per-scope cursor and shared fleet gate remain compatible
  with older replicas. The cursor-reservation transaction now also compares the
  leader token. Losing leadership cannot advance the cursor.
* Metadata probes run concurrently with controller dispatch. They are bounded
  reads, with no new storage-write authority. Discovered demand is persisted
  before task publication, and publication checks the leader token.
* Leader loss cancels discovery without aborting executor jobs. A successor
  rebuilds membership from etcd and continues from the durable cursor. A crash
  after reservation may defer a page until the next sweep; it cannot erase WAL.
* A read that finishes after leadership loss can leave an advisory demand hint.
  The old publisher cannot enqueue/acknowledge it with a stale leader token.
  The new leader can reconcile the retained hint.

## Rollout and limits

Upgrade the elected demand controller before removing the old resident loops
from other replicas. An older leader ignores the new resident membership field;
it cannot perform the new discovery responsibility for upgraded followers.
Existing queued tasks, writer demand and fencing continue to use their existing
wire formats. Old resident loops and the new leader share the same batch gate.
After rollout, verify that all participating masters run the new controller.

This change does not replace external recovery-script scans, multiplex
execution-scoped Watch streams, or establish that etcd timeouts are resolved.
Those paths need separate attribution and measured changes. In particular,
commit/fencing checks must not be weakened to reduce request counts.

Validation includes policy-union updates, stale-leader cursor rejection and
successor continuation, plus the existing real-etcd snapshot/Watch gap,
compaction/relist, leader-crash recovery and 1/10/50-follower RPC tests. Tests
run against an isolated loopback etcd, never production.
