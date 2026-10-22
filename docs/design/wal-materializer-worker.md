# Materializer scheduling over durable WAL publication

The optional `lance-context-ingest/worker` feature runs bounded publication tasks
over authorized `Assignment` values. Each assignment supplies an existing ingress
binding and its resolved Lance storage. Multiple processes may receive overlapping
assignments and share the same etcd claim prefix. The assignments are currently a
fixed set supplied by the application; dynamic namespace discovery and bootstrap
are separate integration work.

`Claims` uses unique, expiring etcd leases to distribute tables. A local
keys-only hint avoids granting and revoking a lease on every busy-table poll;
a stale occupied hint delays an attempt, while a stale empty hint must still
pass the linearizable create-if-absent transaction. Claim validation before WAL
acquisition remains linearizable. Each successful
claim records a new owner/request identity and the complete binding fingerprint.
Renewal and scheduling RPCs have explicit deadlines. A lost claim transaction
response leaves an expiring lease; it never authorizes deleting or stealing an
existing claim. Releasing a claim joins its renewal task and revokes only its own
lease. A cancelled owner stops renewing and allows TTL expiry.

The lease is not the publication fence. Before acquisition the worker reads the
WAL publisher epoch, then checks the claim with a linearizable etcd read, and
finally submits an epoch-conditional WAL acquisition. Reversing those first two
steps would let a paused old owner observe a successor's newer epoch and steal it
back. An old authority cannot prepare, select or confirm the successor's manifests.
Already prepared files and pending publication state survive a change of owner.

`Worker::start` bounds active table jobs, publication rounds per claim, scanning,
and polling. Assignments share a supplied memory budget and the ingresses' WAL
connection pool. Workers release claims after the configured publication quantum
or when no work is found. Busy/failed claims wait for a later poll; they do not spin
in an immediate retry loop. Fixed assignment polling can produce idle acquisition
traffic and is not yet an active-table discovery index or a scaling result.

`Worker::shutdown` stops new scheduling and joins in-flight publication and lease
cleanup. Cancelling a shutdown waiter does not cancel its owning supervisor or
release memory while file work still runs. A task panic drains the other jobs
before reporting failure. `observation` reports local job, claim, publication and
error counts plus supervisor/drain state; it does not claim quorum availability.
Deployment termination grace must cover admitted blob work and lease cleanup.

Readers can use `materializer::read_dataset` with the authorized ingress/storage
binding without acquiring a publisher epoch. It opens only the manifest selected
by the WAL catalog and checks the rollout schema.

The new etcd-backed tests revoke a claim after real Lance file preparation, allow
a successor to acquire, attempt to reacquire through the old claim, verify old
authority rejection and exact prepared-file reuse, then run two automatic workers
over the same table and read every resulting row through an independent reader.
A second test abandons a claim and requires actual TTL expiry before takeover.
Both etcd checks passed, along with the five HTTP/ingress/materializer regressions
and worker feature Clippy checks. They are correctness checks, not a process-kill
or sustained throughput benchmark.

Tracking: #378.
