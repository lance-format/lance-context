# WAL reader lifetime and retirement

This candidate changes WAL reclamation so a merge cannot delete files still used
by a registered reader. It is not yet qualified for production activation.

## Behavior

Readers register under `<table>/_wal_readers_v1` **before** capturing WAL metadata.
Protection covers stream consumption and payload reads. Current-view reads capture
a consistent base/WAL pair; explicit pinned versions remain base-only. Generic
`Fragments` keeps its cached base semantics and `Wal` keeps its pending-generation
semantics. A successful read must still hold a valid registration when it returns.

Each registration contains a random nonce. Readers renew every 5 seconds and fail
if renewal is overdue for 15 seconds. Collectors use monotonic observation of an
unchanged nonce for 45 seconds; they do not trust wall-clock expiry timestamps.
Ambiguous registration writes never authorize an unprotected read. Cancellation
attempts bounded cleanup; abandoned registrations can expire during collection.
The protocol requires strongly consistent storage listing, including visibility
of surviving registrations while other entries change across listing pages.

Merges persist intents under `<table>/_wal_retirements_v1` before the fenced shard
manifest drain. Recovery rechecks the current shard manifest, waits for registered
readers, then deletes only validated retired generation directories. Incomplete
listing, metadata errors and cancelled collection retain the intent for retry.
Logical merges do not wait for readers or for physical collection.

Collection is process-wide and bounded to 4 workers, 256 tables and 1,024 admitted
tickets. Saturation leaves durable intents for rediscovery. The master consumes
the existing registry catalog independently of the stats scan, so cold tables
with zero pending WAL can still have retirement work discovered. This does not
establish a fleet-wide HTTP request limit; backend and concurrency load must be
measured separately.

## Configuration and upgrade

`ROCKETKEEP_WAL_LIFECYCLE` accepts only:

- `retain` (also the default): register readers and record retirement intents;
  continue logical merge, but do not discover or physically collect retired WAL.
- `collect`: retain reader protection and enable bounded retirement collection.

Store and append-coordinator handles capture the mode when opened. Changing the
environment is **not** a live cancellation or safety switch.

1. Qualify real storage behavior, failure handling and metadata load in an
   isolated staging storage prefix. Exercise all read APIs and concurrent merges.
2. Roll qualified code in `retain` mode to every reader and deletion executor:
   workers, all master variants, dedicated helpers and direct library clients.
   Keep write, merge, compaction and indexing work executing. Monitor retained
   object growth. Old binaries may still delete files during this transition.
3. Prove that all old in-flight readers/deleters have finished and no Job,
   CronJob, helper controller or external client can recreate an old executor.
   Pending zero, Pod Ready and a fixed wait are not sufficient evidence.
4. Enable `collect` on selected new executors only after that fleet barrier.
   Verify real physical reclamation, cold-table recovery, request volume, memory,
   merge throughput and read latency before expanding.
5. Never roll back to unregistered readers while any collect executor survives.
   Move collectors to `retain`, join admitted collection work, and reestablish
   the barrier first; prefer a protected-reader rollback image.

## Qualification status

Local tests cover protected reads spanning merge and actual deletion, old handles
reading after physical reclamation, consistent base/WAL capture, cancellation,
retirement recovery, bounded scheduling and retain/collect activation. Store
regressions passed after updating a legacy GC fixture to explicitly exercise an
unregistered reader and enabled physical collection. These are local results,
not proof of real Azure behavior, acceptable metadata traffic, or mixed-version
safety. Those gates remain required before production deployment.
