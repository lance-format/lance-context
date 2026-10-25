# Qualify annotation batches and drain

The opt-in Rust experiment starts from the full rollout schema, adds the narrow
annotation columns through the explicit handoff, and drives one owner through
two forwarding HTTP ingresses. Three real persistent WAL replicas use TLS RPCs
on one host. The replicas, owner, ingresses and two writer tasks run in one test
process with a four-thread Tokio runtime. This does not qualify a multi-host
deployment or maximum ingestion throughput.

```bash
ANNOTATION_LOAD_SECONDS=30 ANNOTATION_LOAD_BATCH_ROWS=32 \
cargo test --release -p lance-context-ingest --features service,http-client --lib \
  tests::annotation_http::batches::annotation_http_batch_load_preserves_values_and_drains \
  -- --ignored --exact --nocapture
```

Use optimized builds for performance comparisons; output identifies debug versus
release builds. Compare batch rows 1, 32 and 128 with all other settings fixed. The default table
has 4,096 turns with 16 KiB inline bodies, existing scores, and no ID index. Body
creation is outside the measured interval; updates and reads project narrow
annotations. Each writer owns a disjoint half of the IDs. Setting
`ANNOTATION_LOAD_WORKING_SET` below `ANNOTATION_LOAD_TURNS` repeats a smaller set
of turns; both default to 4,096. The working set must be divisible by twice the
batch size. This controls key reuse, not an assertion about physical cache hits.

The fixed protocol admits at most 512 rows per record, independently of the
chosen batch size. Staging allows 16 MiB and 16,384 entries. Each active source window
allows 64 records and 8 MiB, alongside at most one frozen window, with a 4 MiB publication threshold and one-second
maximum batch wait. Upload, apply and publication have independent 16/16/64 MiB
budgets. These are admission charges; allocator RSS and Lance internal workspace
are not inferred from them.

Every successful batch is followed by cross-ingress reads of its first and last
turn (one read for a single-row batch). The two producers do not overwrite each
other. After all writers finish, the experiment requires staging/source drain,
then scans every final narrow row against the expected latest successful update,
including untouched rows, scores, labels and duplicate/missing IDs. Unknown
mutation outcomes or read failures fail the experiment. Only an explicit
`not_submitted` response resubmits the original producer/sequence/bytes.

Output reports the actual roles and budgets, per-writer ACK/read latency bounds
in one-millisecond buckets, definite refusal counts/fractions, updated rows,
acknowledged Arrow body bytes, distinct updated turns, observation wait,
sampled active/frozen/source backlog, and final drain time. The elapsed writer
interval includes post-ACK reads and completion of the last admitted request;
reported updates/second is the achieved throughput of that closed-loop workload.
Keep failed runs and their partial results, and do not present them as qualified
capacity. Compare publication progress over time as well as final drain.

## Initial baseline

Three consecutive 30-second release runs against product revision `3b05de4`
used the defaults above. Rates include cross-ingress reads. Latency columns are
the larger of the two writers' p99 bucket bounds; with only four ACKs per writer
in the 128-row run, that bound is the maximum sample, not a stable tail estimate.

| Rows per batch | Updates/s | ACK p99 bound | Read p99 bound | Final drain | Sampled retained entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.35 | 89 ms | 76 ms | 78 ms | 33 |
| 32 | 33.57 | 1,905 ms | 1,041 ms | 1,015 ms | 128 |
| 128 | 34.08 | 7,589 ms | 3,782 ms | 37 ms | 384 |

All runs had zero admission refusals, passed every post-ACK read, verified all
4,096 final rows, and drained. The table retained one fragment. These small
runs demonstrate correctness and a serious performance problem, not capacity.
Increasing the batch size from 32 to 128 barely improved throughput. The largest
sampled staging observation wait grew from about 1.00 to 3.80 seconds. Inspection
shows `pending_batch` holds the state mutex while making a separate Lance scan
for each cold turn. A bounded batch lookup is the next comparison; this baseline
alone does not assign an exact fraction of total latency to that code path.

Refs #378.
