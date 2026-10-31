# Automatic service for low-count WAL tails

A count-only merge trigger can leave one to seven sealed generations indefinitely
when ingestion stops below the default eight-generation threshold. The optional
master tail sweep periodically enqueues this work through the existing durable
scheduler. It does not use table names or table modification timestamps as a
proxy for the actual age of a WAL generation.

Set `MERGE_WAL_TAIL_INTERVAL_SECS=3600` on ordinary merge-scheduling masters to
revisit low-count tails after each rotation. The default is zero (disabled).
`MERGE_WAL_INTERVAL_SECS` must also be nonzero and workers must be configured;
maintenance-only replicas remain disabled. This does not enable the Kubernetes
catch-up controller or change any table's merge protocol.

The sweep uses the existing in-memory scalar stats snapshot, with
`MERGE_WAL_TAIL_STATS_MAX_AGE_SECS=900` by default. It never opens a table, scans
payloads or lists WAL objects. Missing, future or stale observations are excluded.
Only positive counts below `MERGE_WAL_MIN_GENERATIONS` enter this rotation; hot
work keeps the normal sweep. An ingestion/write/read hot path gains no new I/O.

An etcd CAS reserves at most `MERGE_WAL_TAIL_BATCH_SIZE=16` candidates per
shared 30-second batch, hard capped at 64. The default unrestricted policy
shares one fleet-wide rotation; each explicitly selected target scope has its
own rotation and batch bound. The durable name cursor prevents
busy early targets from hiding later targets and survives master replacement.
The interval applies after the last page of a rotation; it is not an exact age
measurement or a guaranteed completion deadline for every table. Existing task
cooldowns, active tasks, draining targets and dedicated catch-up owners are
respected. No lock, owner, attempt record or failure ledger is deleted or reset.

Page reservation precedes enqueue. A lost response or process crash may defer a
page to the next rotation, but never causes immediate replay of an uncertain
mutation or one replica's workload to be multiplied by every other replica.
An enqueue error is isolated to its target; other candidates continue. A new
rotation evaluates current fresh demand through normal scheduler deduplication.
`master_wal_tail_enqueue_requests_total` counts accepted enqueue requests, not
completed generations or necessarily newly created tasks during concurrent
normal scheduling.

This sweep covers low-count ordinary-worker WAL tails. It deliberately leaves
externally registered persistent publisher lifecycle to its current owner; those
tables need the separate desired-coverage migration before external babysitting
can be retired. Bounded rolling metadata coverage remains required.

## Selecting automatic admission targets

`MERGE_WAL_ADMISSION_TARGETS=a,generic:b` restricts this master's count and tail
admission thresholds to those exact targets. The default empty list preserves
existing behavior. The selection applies to aggregate watermarks, early scan
admission, periodic sweeps, and the stats-based tail rotation. Scans still observe
other tables. Scoped tail rotations have separate durable cursors; scoped periodic
policies elect separate owners. Target order and duplicates do not create new
rotation or periodic-owner identities.

This setting is independent of `MERGE_OWNED_TARGETS` and the durable legacy
fencing mode. It does not change execution ownership, manual tasks, or recovery
of already-published merge requests. Other masters' admission policies continue
to apply, so this is not a fleet-wide deny list.

During a mixed-version rollout, scoped thresholds are advertised separately and
the legacy threshold fields are disabled for that member. Older aggregate
leaders therefore ignore its scoped thresholds instead of applying them globally;
their other policies and the member's execution capacity remain available.
Members with a nonempty scope and automatic merging enabled also elect one
independent scoped watermark controller per etcd prefix. This controller admits
the union of live scoped policies without replacing the global dispatch leader
or waiting for persisted statistics. Identical and different scopes share this
one additional informer, rather than starting a fleet watch per table or policy.
It uses the same durable admission frontiers and task deduplication as the global
controller, with publication fenced by its own election lease. All-table
outstanding work still counts against its observed admission allowance. This is
the existing soft queue bound, not an atomic credit limit across independent
producers. Its live cache gauges use `master_scoped_aggregate_wal_*` names so a
global controller in the same process cannot overwrite them. Admission/error
counters include both controllers.

Verify actual member policies, selected/unselected behavior, concurrent global
admission, and scoped-owner failover in an isolated, already-fenced environment
before enabling the new image in production. Older images without the scoped
controller still require a new global leader for watermark admission; their
scoped periodic sweep alone depends on persisted statistics freshness.
