# Batch cold annotation lookups

The initial release experiment in [annotation-batch-load.md](annotation-batch-load.md)
stalled near 34 updated rows/second for both 32-row and 128-row requests. Every
cold turn created a separate Lance scanner while `pending_batch` held the staging
mutex. Increasing request size mostly increased the time readers and publication
waited for that mutex.

Application now processes lookup windows of at most 128 mutations. It collects
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

At most 128 requested keys and their projected row views are retained for one
lookup window. Staged values retain the existing byte/entry charges. Arrow
scanner buffers and Lance I/O workspace remain separate from staging charges;
row limits and shared slices do not establish a total RSS bound. This change
adds no WAL format, request identity, read barrier, or publication protocol.

## Release comparison

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
