# Periodic WAL admission refresh

Persisted statistics can continue to report pending generations after a merge
has drained them. Once the completed task releases its deduplication key, the
next periodic sweep can create another merge from the same stale observation.

Before creating a periodic WAL task with no active task pointer, the master
now inspects fresh WAL shard manifests. A completed observation with no sealed
SSTs skips that candidate for this sweep. Any remaining generation retains the
ordinary enqueue path, even if it is below the original stats threshold. The
persisted count is not rewritten: a partial pass must not lose its tail.

This check does not apply to writer requests, aggregate frontier admissions,
execution recovery, or failure recovery. It never clears an execution, changes
ownership, authorizes a commit, or permits an unfenced replacement. There is no
negative cache between sweeps or leaders. A new seal concurrent with the probe
is handled by the existing writer/frontier paths and subsequent fresh sweeps;
this advisory observation is not an atomic snapshot or an ownership fence.

## Storage and scheduling bounds

The probe opens an object-store client for the canonical main-branch URI. It
does not open the base dataset, list its versions, count rows, open generation
datasets, or instantiate writers. It enumerates persisted `_mem_wal` shard
directories, including historical writers absent from current configuration,
then refreshes one shard manifest at a time with a forward HEAD batch of one.

Cloud directory enumeration uses native delimiter pagination, requesting at
most 64 children per page, at most eight logical page calls and 256 entries.
Continuation tokens are preserved, including on short or empty pages. Remote
full-directory and recursive listing fallbacks are denied. Unsupported
providers, incomplete enumeration, missing shard manifests and absent shards
produce an unknown observation and retain ordinary admission. Unix local files
use a bounded immediate-directory walk; symlinks are unknown.

Each probe permits at most 512 logical object-store operations, objects up to
1 MiB, and a cumulative 4 MiB of requested object-body ranges. These are **not**
a bound on physical HTTP requests, total response bytes, or process memory:
provider retries, listing response framing, HTTP buffering and decoding are
separate. The page size is sent to the backend before listing; returned page
validation alone is not presented as a pre-allocation memory bound.

A sweep spends at most 64 probes and 10 seconds of cumulative probe time,
with a three-second deadline for each probe. Exhaustion or any storage error
retains ordinary admission. Consequently the change reduces verified empty
requeues but does not guarantee elimination of all stale-stat work when probes
cannot finish. It introduces no etcd operations or persisted keys.

## Observation

`master_periodic_wal_refresh_total{outcome}` counts `empty`, `pending`,
`unknown`, and `budget`. These labels are finite and have no table identifiers.
`master_periodic_wal_refresh_duration_seconds{outcome}` records the complete
probe duration, including timeouts (a skipped budget entry makes no sample).
Successful empty checks log the target. Storage failures are debug logs.
Enabling `lance_context_core::wal_presence=debug` records per-probe logical
operations, listing pages, requested body bytes and elapsed time without adding
high-cardinality metric labels. Cancelled probes have a duration/unknown sample
but no completed-probe debug event.
Compare this with `master_task_admissions_total{kind,origin,outcome}` to
separate periodic admissions from writer and recovery work.

## Release evidence required

Local tests exercise actual bounded merges, historical shards, new seals,
unchanged stale stats across two masters, recovery and writer admission,
missing or oversized metadata and directories, read-only access, and native
Azure client HTTP page-size/token behavior against a loopback fixture.

Before deployment, use an isolated Azure staging URI with 20 historical
shards and actual merges to measure probe latency/operation counts, repeat
sweeps after drain, and new writes/recovery after drain. Do not infer Azure
latency from local tests or infer fleet etcd reduction from fewer empty tasks
alone. Production acceptance also needs a comparable workload interval showing
reduced Range/Txn pressure without lost productive merges or read regressions.
