# Batch cold annotation lookups

The initial release experiment in [annotation-batch-load.md](annotation-batch-load.md)
stalled near 34 updated rows/second for both 32-row and 128-row requests. Every
cold turn created a separate Lance scanner while `pending_batch` held the staging
mutex. Increasing request size mostly increased the time readers and publication
waited for that mutex.

Application now processes lookup windows of at most 2048 mutations. It collects
only keys missing from pending, active and frozen values, and reads those keys
with one Lance `IN` filter against the same base snapshot. Only `id`, `labels`,
`status` and `score` are projected. The temporary map borrows request keys and
holds Arrow row slices sharing the returned buffers. It does not preconstruct a
second map of owned annotation strings. Values are decoded when their mutations
are applied, in source order; later mutations of the same turn use pending state.
There is no clean-value cache after the batch.

This retains the existing serialized apply and publication lock. The whole WAL
batch is installed only after every window succeeds. Missing rows and invalid
mutations still reject the whole batch; capacity pressure does not advance the
applied cursor. A duplicate base key or storage failure stops application without
exposing partial values. One extra result row detects duplicate keys without
retaining an arbitrarily large corrupt result set.

At most 2048 requested keys and their projected row views are retained for one
lookup window. Staged values retain the existing byte/entry charges. Arrow
scanner buffers and Lance I/O workspace remain separate from staging charges;
row limits and shared slices do not establish a total RSS bound. This change
adds no WAL format, request identity, read barrier, or publication protocol.

Query windows are independent of scanner vectors, which remain 128 rows. The
original implementation used that vector size as the key-window size too: a
2048-row cold mutation batch could issue 16 sequential Lance queries while
holding the staging lock. A window now issues one query for those keys. Larger
accepted batches still use bounded windows; repeated keys and rejection row
numbers retain their original whole-batch order. The temporary projected row
views can retain more Arrow buffers than before, so this trades bounded lookup
workspace for fewer query starts. It does not increase the durable staging limits.

## Original 128-key release comparison

The same one-host experiment, budgets, 4,096-turn working set and two writers
produced these 30-second results. Each update batch is followed by cross-ingress
reads. Latency bounds are the larger per-writer p99 buckets; the original large
batches had too few ACKs for a stable tail estimate.

| Rows per batch | Before updates/s | After updates/s | After ACK p99 bound | After read p99 bound | After drain |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.35 | 70.07 | 47 ms | 27 ms | 360 ms |
| 32 | 33.57 | 2,052.07 | 40 ms | 22 ms | 1,062 ms |
| 128 | 34.08 | 8,589.27 | 36 ms | 23 ms | 695 ms |

All three runs passed every post-ACK read, verified all final rows, drained,
and had zero admission refusals. The 128-row run acknowledged 257,792 updates;
its largest sampled staging charge was 2,491,136 bytes and its active source
window reached the configured 64-record limit. The table retained one fragment.
Its largest logged staging observation wait was 1,525 microseconds, compared
with 3,795,380 microseconds before the change.

This comparison measures the combined lookup change: queries are shared across
keys, the scanner batch size increases from two rows to 128, and the key column
is projected. Faster application also revisits this finite working set more
often while values remain staged. These are achieved annotation-update rates
with final drain, not pure cold-scan throughput, body ingestion bandwidth, or
multi-host capacity. At the time of this experiment, the ordinary point-read
miss still used the existing single-key scanner with a two-row scan batch.

An additional five-second run used 65,536 turns and 128-row batches with the
same budgets. It acknowledged 32,000 updates to 32,000 distinct keys, so every
update started without a staged value for that turn. It achieved 6,374.54
updates/s, with ACK/read p99 bucket bounds of 49/21 ms and 554 ms final drain.
All 65,536 final rows were verified, no requests were refused, and the largest
sampled staging charge was 4,102,272 bytes. This is a first pass over cold staging
keys; fixture creation had already touched local storage, so it does not measure
cold OS caches or remote object-store latency.

A 120-second run with the original 4,096-turn working set and 128-row batches
completed 1,010,176 updates at 8,417.45 updates/s. ACK/read p99 bucket bounds were
37/21 ms, all 15,784 post-ACK reads succeeded, and no requests were refused.
Materialization advanced throughout the sampled run; final drain took 697 ms
and all 4,096 final rows matched. The largest sampled staging charge was
2,563,328 bytes (4,864 entries), below the 16 MiB / 16,384-entry limits. The
source window stayed within 64 records and the table retained one fragment.
This is two minutes of annotation-only load, not mixed-ingestion or deployment
qualification.

```bash
ANNOTATION_LOAD_SECONDS=5 ANNOTATION_LOAD_TURNS=65536 \
ANNOTATION_LOAD_BATCH_ROWS=128 cargo test --release -p lance-context-ingest \
  --features service,http-client --lib \
  tests::annotation_http::batches::annotation_http_batch_load_preserves_values_and_drains \
  -- --ignored --exact --nocapture
```

## Point-read scan batches

A point read now uses the same 128-row scan batch size while retaining its
two-match result limit for duplicate detection. It still projects only labels,
status and score against the captured base snapshot. The result limit and scan
batch size serve different purposes: vectorizing a scan two rows at a time
made an unindexed miss from staging unnecessarily expensive.

A separate ten-second release experiment measured sequential point reads with
empty staging against fixed local Lance 13 files, no ID index, the full rollout
schema and 16 KiB bodies. Labels/status schema setup completed before timing.
Every returned annotation was checked, and staging retained zero values.

| Table rows | Before reads/s | After reads/s | Before p50 | After p50 | After p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 50.20 | 624.32 | 18.720 ms | 1.571 ms | 2.089 ms |
| 65,536 | 3.45 | 148.23 | 284.634 ms | 6.404 ms | 8.672 ms |

Repeating the old executable after the candidate on the same files yielded
52.59 and 3.42 reads/s respectively. These are staging misses, not cold OS-cache
measurements. The original 65,536-row run had only 35 samples, so it does not
support a reliable tail-latency estimate. A larger scan batch does not replace
an ID index or establish large-table lookup capacity.

The separate 30-second RF3 process experiment with one owner, two ingresses,
two writer loops, 128-row updates and durable client journals changed from
3235.55 to 3251.57 updates/s: no meaningful aggregate throughput gain. Both
runs passed all post-ACK reads, complete 4,096-row verification and source
reclamation. The point-read scan improvement is not a body-ingestion or WAL
throughput result.

Refs #378.

## 2048-key query windows

A separate-process indexed comparison used 65,536 turns, 2048-row mutation
batches, one RF3 group, one owner, two ingresses and two independent client
writers on one host. Both arms used the same saved client and WAL binaries,
128-row scan vectors, one-second publication aggregation, 16 MiB / 16,384-entry
staging, 32 MiB upload/apply budgets and a 64 MiB publication reserve. Only the
owner/ingress binary changed the query-key window; ingress forwarding does not
perform these Lance lookups. Initial test-table files were synced before each
load; WAL durability was unchanged.

| Query keys | Duration | Updates/s | ACK p99 bound | Point-read p99 bound |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 30 s | 27,234 | 272 ms | 34 ms |
| 2048 | 30 s | 34,390 | 308 ms | 14 ms |
| 2048 | 120 s | 33,549 | 299 ms | 15 ms |

All runs passed complete value/index checks for 65,536 rows, source reclamation
and final empty staging/source observations, with zero refusals or read errors.
The 120-second run acknowledged 4,026,368 updates and checked 3932 post-ACK reads
plus 64 final point samples. This demonstrates sustained drain for this bounded
annotation workload, not body-ingest bandwidth, a production capacity limit or a
300-billion-turn lookup qualification. The shared local disk still recorded
occasional 100–200 ms WAL write/poll stalls; improved point reads did not eliminate
ACK tail latency. Projected lookup workspace is larger even though staging limits
and row-vector size remain unchanged.

A sequential cloud comparison used three WAL replicas on separate hosts and
Lance on blob storage, with the same 65,536-turn indexed table, two writers and
budgets above. The table was reused, so cache state and accumulated content are
not controlled as they would be in an isolated repeated A/B experiment.

| Query keys | Publication wait | Duration | Updates/s | ACK p99 bound | Point-read p99 bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 s | 120 s | 9,238 | 628 ms | 267 ms |
| 128 | 3 s | 120 s | 6,859 | 1063 ms | 353 ms |
| 2048 | 1 s | 120 s | 20,977 | 477 ms | 382 ms |

The longer publication wait reduced throughput and reached staging pressure;
it is not recommended by this experiment. The larger key window improved
throughput but worsened cloud point-read p99. Application still holds the staging
mutex during cold lookup, so this is not a claim of improved read tail latency.

The 2048-key cloud run acknowledged 2,519,040 updates, checked 2460 post-ACK
reads and 64 final samples, and verified all 65,536 values, the original ID-index
identity/fragment coverage, source GC and final empty staging. It reached the
16,384-entry limit, with 46 backpressured observations and zero client refusals;
waiting under pressure still contributes to latency. Peak observed staging
charge was 8,798,208 bytes and owner RSS was about 165 MiB. No OOM or CPU
throttling was observed; two synchronous WAL writes took about 104 and 154 ms.

A following fault phase acknowledged 4096 updates while publication was delayed,
then killed the owner and all three WAL processes with SIGKILL. Reopening the
original persistent directories first observed the old materialized cut, then
all 4096 replayed entries before publication. Recovery passed point reads, full
value/index checks, source GC and final drain. This qualifies process termination
with retained storage, not power failure or loss of all replica disks.
