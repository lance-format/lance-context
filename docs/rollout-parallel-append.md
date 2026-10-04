# Parallel rollout WAL append

Rollout rows are immutable and their IDs are never reused for different records.
This opt-in path moves WAL reading, encoding and immutable file uploads to workers,
then publishes several workers' files in one base-table transaction on the master.
Generic/context stores retain their existing keyed merge behavior. Staging requires
Lance V2 files, whose dictionary encodings are local to each file.

## Configuration

Deploy the compatible worker binary first. Select **owned, non-draining rollout
names** with `ROLLOUT_APPEND_TARGETS=table1,table2` on the master. `*` selects all
owned rollout targets; generic targets are always excluded. The empty default
keeps the existing path. Every staging worker must advertise append protocol 1 and owned-merge
configuration for the target. All ingestion writers must already honor the same
owned-target configuration; the table claim cannot exclude legacy, uncoordinated
writers. There is no fallback to an older
manifest-publishing worker RPC after a capability or staging failure.

| Setting | Default | Bound |
| --- | --- | --- |
| `ROLLOUT_APPEND_CONCURRENCY` | 4 | 1–32 concurrent staging RPCs per table |
| `ROLLOUT_APPEND_MAX_GENERATIONS` | 64 | 1–256 generations per shard per pass |
| `ROLLOUT_APPEND_MAX_BYTES` | 64 MiB | 1–256 MiB before starting another generation |

Worker merge slots and the existing process-wide `MergeMemoryBudget` are shared
with other merge work. A worker refuses staging if that budget is disabled or the
requested byte limit exceeds its configured limit. Generations remain indivisible;
an oversized first generation retains the existing exclusive-growth behavior.
Reservations account for twice buffered Arrow bytes to accommodate filtering.
Encoding and Lance caches also consume memory, so these settings are not an RSS cap.

The master enumerates shard directories using metadata only, and also visits
advertised stable worker identities and additional `CATCHUP_SHARDS`. Unavailable
workers are skipped during staging admission; healthy workers can process their
WAL. An explicit incompatible capability response rejects the new protocol. Any compatible
worker can stage immutable generations from any shard; it does not claim its epoch.
Dedicated catch-up executors receive these settings through the generated Job
and perform parallel staging with their own CPU and shared catch-up memory budget;
they do not send their data work back to ordinary workers. A speculative read
which cannot grow its reservation retries once alone after other readers finish.
Both normal master merges and catch-up executions use the same publisher under
the existing maintenance ownership protocol. The ordinary master never reads WAL
payloads locally, including when no remote worker is available.

## Commit and recovery

1. Read bounded shard prefixes and capture the base version and schema.
2. Workers stage files with `InsertBuilder::execute_uncommitted`, returning only
   fragment descriptors. Staging cannot publish base/shard manifests.
3. Publish a bounded group when it fills or two seconds after its first result.
   A slow worker does not hold completed groups indefinitely. The two-second
   interval governs batching, not execution expiry.
4. Use Lance's `Operation::Update` with **no removed/updated fragments**: its new
   fragments and `merged_generations` watermarks are committed atomically. This
   performs no row update, key-index maintenance, or target payload scan.
5. Drain only the WAL generations covered by the committed watermark, preserving
   concurrent flushes and the live writer epoch. Delete drained directories
   asynchronously, as with the previous merge implementation.

An uncertain commit response is resolved by reopening the base and checking its
watermarks. Fully committed results become drain-only retries. Partially stale
prefixes must be replanned; appending their whole files would duplicate rows.
Unexpected schema changes or a changed WAL prefix reject the staged result.

The final commit handler pins the exact validated next version. Merely setting
Lance's retry count to zero is insufficient: it still attempts transaction rebase
before its first write. Ownership authorization and the existing shielded manifest
write scope remain active around the conditional storage write. Master failure or
revocation uses the existing recovery fence before another publisher takes over.

A staging failure gets one retry on another worker. Old staging can leave orphan
files but cannot publish them. Persistent failures return to the existing durable
merge failure/backoff policy. HTTP progress is NDJSON: unchanged liveness messages
do not reset the master idle watchdog. Completed reads and increasing Lance
write statistics do; upload-buffer progress is not evidence of a committed table.

## Existing tables and rollback

At first use, persist a per-shard cutover boundary in table metadata. The historical
WAL prefix may contain rows from a legacy append whose subsequent drain failed.
While staging this prefix, probe the base for IDs in chunks of 1024, projecting
**only `id`**, and omit already-published immutable rows. This costs key lookups
until the historical prefix is consumed; it does not read old payloads or delete
base rows. Generations beyond the cutover do not require this migration lookup.

Merge retries are idempotent through the atomic generation watermark. This is not
an arbitrary upsert or ingest-deduplication API: replaying the same ID into different
new generations/shards after cutover violates rollout's no-reappend contract.
Do not enable the path for workloads requiring that behavior or row updates.

Disabling the master setting returns to legacy keyed merge. Updated legacy
preparation also reconciles committed watermarks before reading WAL, and its
append records a new atomic watermark after cutover. A staged commit interrupted
before drain can safely finish through that path, and disabling/re-enabling the
setting does not lose the fallback publisher's commit evidence. Rolling back to
an older binary without this compatibility code requires resolving pending legacy
writes before re-enabling staged append. Changes to
ownership/drain configuration still follow the existing ownership rollout rules.

Staged files are not reader-visible until their manifest is committed. Files from
cancelled/failed stages remain subject to normal unverified-file retention; they
must not be eagerly deleted while another execution may still reference them.
No separate orphan-file vacuum or changes to compaction scheduling are introduced.
New fragments may be uncovered by the existing ID index until normal index
maintenance runs; point queries must retain their existing uncovered-fragment scan.

## Verification

Tests exercise parallel file preparation and a single combined commit, live writes
arriving between stage and publication, duplicate results, partial stale prefixes,
byte bounds, legacy append-without-drain migration, restart after base publication,
and fallback through the existing merge implementation. Master HTTP tests require
two workers to reach a barrier together, catching accidental serial fan-out.

A local ARM64 debug-build benchmark (Lance 9.0.0, one process, local filesystem)
used four shards × eight generations × sixteen 8-KiB payload rows, about 4 MiB per
fixture. Two fresh fixtures per variant, with the second order reversed, produced:

| Merge path | Mean time | Generations/s | Base version increments |
| --- | --- | --- | --- |
| Existing serial keyed merge | 1.141 s | 28.0 | 11 |
| Four concurrent stages + combined append | 0.724 s | 44.2 | 2 |

Both variants consume the same 32 generations and verify all 512 final IDs and
payloads, with no duplicate base rows. The append path's two versions are the
one-time cutover metadata commit and one combined data/watermark commit.
This is approximately 1.58× throughput for this small synthetic workload, not an
Azure or multi-Pod production throughput measurement. Large-blob and highly
fragmented tables still need deployment-environment measurements.

Run the opt-in benchmark with:

```bash
ROLLOUT_APPEND_BENCH=1 cargo test -p lance-context-core --lib \
  benchmark_rollout_parallel_append -- --ignored --nocapture
```

`ROLLOUT_APPEND_BENCH_ROOT` optionally selects a scratch object-store prefix. Each
fixture creates a new UUID subdirectory; it never scans an existing table.
