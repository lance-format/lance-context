# Native master catch-up jobs

The master can provision bounded Kubernetes Jobs for tables whose scanned WAL
count exceeds a threshold. The same admission rules serve an automatic scan and
`POST /api/v1/catchup`, so an agent can request recovery without shell access,
custom Pod specs, arbitrary commands, or a bypass for ownership and resource
limits. The controller lives inside the master process, independently of its
ordinary scheduler pools. No developer workspace or external supervisor is needed
for this catch-up path.

This feature is disabled by default. It requires owned merge targets. Deploy
compatible masters, including auxiliary maintenance masters, before activation:
the native executor adds a `catchup` maintenance discriminator to the existing
execution fence. Older readers reject that discriminator. This does not convert
legacy writers into fenced writers; existing per-table rollout/drain requirements
still apply. It does not enable owned targets or restart production workers.

## Configure and activate

1. Build the native master binary into a tested image and pin its digest in a
   copy of `deploy/catchup/pod-template.example.json`. Mount this PodSpec JSON into
   every master. Provide storage and etcd credentials through the reviewed
   template's Secret references/mounts. Controller-generated environment values
   override the target, data URI, etcd endpoints/prefix and execution budgets.
   Image, command, credentials and resources never come from API callers.
2. Bind the namespace-scoped Role in `deploy/catchup/rbac.yaml` to the masters'
   ServiceAccount. It permits creating/getting/patching Jobs and listing Pods; it grants
   no Pod deletion, exec, or deployment modification. The executor Pod does not
   receive the Kubernetes service account token.
3. Set `CATCHUP_POD_TEMPLATE` to the mounted JSON path,
   `CATCHUP_NAMESPACE` to the deployment namespace, `CATCHUP_SHARDS` to the exact
   stable worker identities used to derive WAL shards (for example
   `worker-0,worker-1`), and `CATCHUP_ENABLED=true` on compatible masters.
   Shard identities must include historical writers whose WAL should be drained.
4. Keep the same template, namespace, shard list, Job limit and execution budgets
   on every master. The first controller persists this policy in etcd; mismatched
   replicas fail closed. To change it, disable admission on every master, keep
   reconciliation configured until all reservations finish, then remove only the
   `/catchup-policy` key before starting the new shared configuration.

Defaults: four Jobs cluster-wide, 256 pending generations, a 30-second scan
interval, and stats no older than 900 seconds. A Job has one container, one table,
ordered shard commits, no Kubernetes retries, a 1800-second soft admission slice, a
30-second termination grace period, and a 24-hour terminal retention period.
The Job cannot preempt other Pods. CPU/memory requests must equal explicit
limits. The sample reserves 2 CPUs and 8 GiB per Job (8 CPUs/32 GiB total at the
default fleet cap); size these against available capacity before enabling.

The native process uses a 128 MiB shared Lance cache, a single committer,
64 generations (`CATCHUP_MERGE_MAX_GENERATIONS`) and 64 MiB per batch, and a
1 GiB merge-buffer budget. Generation count and byte limits both apply; zero
count disables only the count cap. Small generations can now share a commit
instead of repeating index maintenance every eight generations.

`CATCHUP_PIPELINE_ENABLED=true` (default) overlaps the next shard's preparation
with the current shard's commit. At most two batches are preparing/retained,
sharing the same memory budget. Commit and drain order remains shard order;
no ordinary worker fan-out is made concurrent. A budget that only admits one
batch falls back naturally to serial reads/commits. Set the flag to `false` and
the generation cap to `8` for a comparison with the previous executor. Both
settings are passed explicitly to new Jobs. These
are buffering bounds, not a total RSS guarantee: one indivisible generation and
Lance working memory can exceed the buffer bound. Kubernetes limits provide the
final process bound. Validate real large generations before increasing concurrency.
A slice visits configured shards up to 16 times, stopping early when a
whole pass reclaims nothing or `CATCHUP_SLICE_SECS` ends. An already admitted
operation may finish beyond that slice while making progress. No Job or Pod
runtime deadline is installed at creation. Each
Job rotates the starting shard using a durable attempt number, including after
failures, so a slow first shard cannot always hide the rest. Further fresh
pressure can request another slice.

## Agent/API contract

These are internal admin endpoints with the same network/authentication boundary
as `/api/v1/tasks`. Do not expose them publicly; use the deployment's authenticated
admin ingress or an authenticated proxy for agents. This change adds no public
unauthenticated agent gateway.

```http
POST /api/v1/catchup
Content-Type: application/json

{"target":"hot_table","reason":"WAL pressure observed during reads","dry_run":true}
```

A dry run evaluates eligibility, active reservations, table locks and capacity.
Remove `dry_run` or set it to `false` to reserve a Job. Repeating the request
coalesces with the active Job and does not reset its failure budget. Generic
stores use scheduler target names such as `generic:events`.

The response contains `target`, `decision`, and optional `job`. `reserved` means
admitted, not started or completed. Other decisions describe missing/stale stats,
a threshold not crossed, a non-owned target, a current execution needing recovery,
a table still busy, cooldown, capacity exhaustion, or an existing Job. Unknown
request fields (including attempts to supply a Pod spec or force flag) are rejected.

```http
GET /api/v1/catchup?target=hot_table
```

Status exposes the current/latest Job, admission pending count and reason, active
reservation, start/finish timestamps, outcome, consecutive failures, next retry
and `needs_attention`, the last observed execution/sequence and observation times,
plus whether termination was requested. A missing target record returns JSON `null`. For failures,
inspect this status and the referenced Job logs; do not clear ownership keys or
blindly recreate Pods.

An agent or skill should first read status, issue a dry run, request a Job only
when eligible, and poll status at a bounded interval. It must treat
`awaiting_execution_recovery` as recovery in progress, not permission to run an
unfenced writer. All resource and retry rules apply identically to agents and
automatic requests. Write/read pressure reports can use this endpoint; this PR
does not add synchronous metadata scans or HTTP calls to those data paths.

## Failure and ownership behavior

Admission atomically reserves a per-table key and one persistent fleet slot in
etcd. A native scheduler task already holding a target lock prevents provisioning;
there is no fleet of empty waiters. Ordinary merge claim transactions exclude
reserved targets, while the native executor must present the exact Job identity
to claim that target's queue entry. Other tables and master jobs continue normally.

A lost master or Kubernetes create response does not release a slot: another
master reconciles the same deterministic Job name. Failed Jobs retain capacity
until every associated Pod is terminal. A failed/unknown Kubernetes lookup
retains the reservation and reports an error. A Kubernetes outage or a Pod stuck
terminating may therefore require cluster repair; claiming completion would allow
unbounded overlapping Pods. Controllers do not remove those reservations blindly.

The native executor merges only sealed generations without claiming the live
writer's epoch. It uses the same durable target ownership, admission revocation,
progress timeout and storage version barrier as other owned maintenance. Losing
a task lease or terminating its Pod is not enough to authorize new storage writes;
unresolved execution fencing must still succeed. The native executor times out a
busy initial task claim after 30 seconds. No-progress work uses the existing
`MAINTENANCE_IDLE_TIMEOUT_SECS` (600 by default). The master independently
observes completed work, not heartbeats. Continuous no progress for that interval
plus a separate 60-second confirmation triggers recovery; the startup window
before an execution appears is `CATCHUP_STARTUP_TIMEOUT_SECS` (1800 by default).
A new execution receives a fresh observation window. Failed samples or a gap
longer than three controller intervals (at least 60 seconds) invalidate the old
window; an observation outage does not prove a stall.

For a running execution, a final etcd transaction compares both ownership and the
observed progress sequence before revoking commit admission. A concurrent advance
wins over cancellation. Only then does the master patch the exact Job's deadline
to terminate a non-cooperative process, with UID/resource-version preconditions.
It persists the termination request for retry across master/API failures. This
is a reaction to confirmed lack of progress, not a runtime limit on healthy work.
The existing storage barrier must still fence previously admitted commits before
replacement writes; Job termination alone is insufficient. Confirmed Job UIDs
cannot be silently replaced or recreated after disappearance.

Completed nonempty WAL batches and successful manifest commits count as progress;
conditional commit conflicts and unchanged publications do not. A pending count
may rise under ingestion despite useful merge work, so it is not the stall signal.
Some Lance-internal phases, including a single index build or long storage call,
do not yet expose intermediate checkpoints. Validate the idle threshold against
these phases on real large tables; completed-step telemetry cannot establish
whether an opaque call is still advancing internally.

Failure counts survive master and Job replacement. Failed Jobs back off from
120 seconds to one hour, with `needs_attention` after three failures; the existing
merge failure ledger remains in force too. A successful Job waits at least a
minute and requires a newer stats observation before readmission, preventing an
old high count from creating repeated empty Jobs. There is no unlimited immediate
retry loop. Controller errors are visible in status, logs and
`master_catchup_*` counters/gauges. The controller examines a rotating window of
up to 128 eligible candidates so a busy group of large tables cannot indefinitely
hide the remainder.

Disable new admission with `CATCHUP_ENABLED=false` while retaining the template
and policy settings until existing reservations finish. This leaves reconciliation
running. Removing the entire controller configuration before Jobs finish strands
reservations and is not the shutdown procedure.

This is a catch-up controller, not a replacement for every external watchdog rule.
It does not evict ordinary workers or diagnose arbitrary OS-level deadlocks. Its
progress-based termination handles dedicated processes; ordinary owned-task recovery still
uses the existing worker identity/progress detection and cancellation protocol.

## Validation before production activation

The tests cover multi-master CAS admission, global capacity, wrong-Job claims,
ordinary-table progress, durable backoff, policy disagreement, an ambiguous Job
creation response, delayed Pod termination, and a native multi-shard merge with
live ingestion handles, progress beyond the old runtime ceiling, repeated unchanged
heartbeats, sampling gaps, stale-progress revocation races, and Job identity checks. Run the etcd-backed tests with `ETCD_TEST_ENDPOINTS` and
`--include-ignored`; the repository CI runs ignored etcd tests separately.

In staging, use isolated storage and an etcd prefix. Verify real large-generation
RSS and throughput, continuous writes with result conservation, survival beyond
the slice while advancing, confirmed no-progress termination, master replacement after reservation/creation, storage failure during
a commit, and complete fencing before retry. Merely seeing a Pod become Ready
is not a throughput or recovery test. These changes do not themselves deploy or
activate the production controller.

Owned worker merges and local owned maintenance also stop using a total execution
time ceiling in this change. Their existing idle watchdogs, ownership checks,
queue limits, cancellation and manifest-drain barriers remain. The legacy timeout
configuration/wire fields remain for compatibility; older master/worker binaries
may still enforce them during a mixed-version rollout. Deploy compatible versions
throughout before relying on the absence of a total execution ceiling.
