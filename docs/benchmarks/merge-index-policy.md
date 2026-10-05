# Configurable WAL merge key index

`ROLLOUT_KEY_INDEX_TYPE=btree|zonemap` selects the base-table key index for
rollout, generic and datagen stores opened by workers. Masters use the same
setting for scheduled `IndexId` and propagate it to native catch-up Jobs.
The default remains `btree`. Core callers can set `key_index_type` in their
store options. This does not change ContextStore's separate ID-index setting.

Use the same setting on every writer/maintenance process for a table. An
existing `id_idx` of a different type is replaced during explicit index
maintenance, not just by opening the table. Roll out consistent configuration
before admitting maintenance with the new policy; otherwise mixed policies
can repeatedly replace the index. Selecting ZoneMap does not remove other
user-created indexes.

## Why merges use predicate deletion

With Lance 9.0.0, `merge_insert` requires an exact-answer index for its indexed
lookup path. ZoneMap does not qualify. A key-only source with
`WhenMatched::Delete` still produces a target scan projecting payload columns
on its non-indexed path. Both a Rust execution-plan probe and an Azure storage
benchmark reproduced this. Changing only `IndexType::BTree` to `ZoneMap`
therefore reintroduces payload scan amplification.

Both BTree and ZoneMap policies use `DeleteBuilder::from_expr` with typed ID literals,
allowing the predicate scanner to prune zones and read only keys. Expressions
are bounded to 1,024 IDs per delete; every delete commits before the prepared
rows are appended. The original WAL remains until the append and manifest
drain succeed. A retry deletes the same keys and appends their newest values
again. Existing table ownership, commit fencing, WAL byte limits and the shared
merge reservation are unchanged. Many small keys can require multiple delete
commits; this tradeoff must be measured for the intended workload.

WAL merges neither create nor extend the key index. Explicit `IndexId`
maintenance still manages coverage. This removes competing `CreateIndex`
transactions from parallel shard merges; uncovered fragments require key scans,
so measure scan costs and keep scheduled maintenance running. The historical
`index` phase metric is no longer emitted by the WAL merge path.

## Reproduction

Use a new isolated prefix for every invocation; the script writes new datasets
and must never point at a production table. Azure authentication uses the
normal Lance environment/workload identity. No production records are copied.

```
BENCH_ROOT=abfss://CONTAINER@ACCOUNT.dfs.core.windows.net/index-study \
BENCH_RUN=run1 BENCH_CASE=large_payload_ordered \
BENCH_KIND=zonemap_predicate BENCH_BATCH_ROWS=1024 \
python docs/benchmarks/merge_index.py
```

Compare `BENCH_KIND=btree BENCH_MAINTAIN_INDEX=1` (the previous eager-index
merge) with `BENCH_KIND=btree_predicate BENCH_MAINTAIN_INDEX=0` (the current
BTree path). `zonemap_predicate` with maintenance disabled exercises the current
ZoneMap path; set it to 1 to reproduce the earlier ZoneMap measurements below.
`zonemap` and `none` reproduce the old non-indexed join as controls. Each
invocation runs in a fresh process, generates deterministic synthetic data,
measures Lance read bytes and I/O counts plus sampled process RSS, and tests
0%, 50% and 100% duplicate IDs. It then retries the last batch and verifies all
IDs/revisions plus one payload. It does not simulate a mid-commit process kill.
Use 128 rows for `many_ids_random` / `many_ids_ordered` (262,144 base rows,
256-byte payloads), and 1,024 for `large_payload_random` /
`large_payload_ordered` (4,096 base rows, 64-KiB payloads; 64-MiB merge batches).

These measure index/delete/append costs on real object storage, not complete
WAL read/dedup/drain or production table throughput. Row distributions and
sizes are synthetic, not a sample of current production traffic. Small-table
results do not establish a fleet-wide default. Report storage failures and
exclude incomplete fixtures from successful-run aggregates.

## Azure results, 2026-10-03

Lance 9.0.0; one isolated AMD64 Pod, 4 CPU limit / 8 GiB memory limit;
sequential fresh processes for each fixture. The first four rows below each
contain six batches (two fixtures, each with 0%, 50%, 100% overlap); the
fragmented row contains three batches. Numbers cover index maintenance +
delete + append, and exclude initial index creation and base-table loading.

| Synthetic base / batch | BTree mean / median seconds | ZoneMap predicate mean / median seconds | BTree / ZoneMap mean read MiB |
|---|---:|---:|---:|
| 262k unordered IDs, 8 fragments / 128 rows | 1.213 / 1.309 | 0.687 / 0.681 | 4.930 / 2.292 |
| 262k ordered IDs, 8 fragments / 128 rows | 1.031 / 1.065 | 0.707 / 0.720 | 2.892 / 0.143 |
| 4k unordered IDs, 256 MiB payload, 8 fragments / 64 MiB | 1.128 / 1.116 | 1.011 / 1.051 | 0.303 / 0.460 |
| 4k ordered IDs, 256 MiB payload, 8 fragments / 64 MiB | 2.344 / 1.073 | 1.071 / 1.040 | 0.303 / 0.139 |
| 262k unordered IDs, 2,048 fragments / 128 rows | 1.102 / 1.266 | 1.354 / 1.681 | 5.208 / 57.068 |

The large-payload ordered BTree mean includes a successful slow append; its
median shows why the mean alone must not be used to claim a 2x improvement.
In the fragmented case, predicate projection avoids materializing payloads,
but small object reads/prefetch and weak zone pruning still amplify physical
I/O. ZoneMap is not uniformly faster. Sampled peak process RSS was 278–625 MiB
for the new ZoneMap path and 343–619 MiB for BTree across these fixtures.

A separate control using ZoneMap with the original `merge_insert` path read
263.27 MiB on average per delete against the 256 MiB payload fixture. The new
predicate path read 0.06 MiB per delete in the corresponding 128-row control.
Index selection without the delete-path change is therefore insufficient.

Every completed fixture passed ID/revision conservation, same-batch retry,
and payload spot checks; the Pod recorded zero OOM events. One ordered
large-payload ZoneMap fixture failed on a 30-second Azure manifest PUT timeout
during retry append. It is excluded from successful aggregates and recorded
in `merge-index-results.json`; a new-prefix rerun completed. Raw logs are
retained in the task's `/tmp/lc-index-study` directory.

These results support an opt-in ZoneMap policy for suitable layouts, while
retaining BTree as the default. A workload with thousands of overlapping
fragments needs further compaction/layout work or its own representative
benchmark before switching. To reproduce the fragmented case, use
`BENCH_CASE=fragmented BENCH_FRAGMENT_ROWS=128 BENCH_BATCH_ROWS=128`.


## Deferred index maintenance, Azure results, 2026-10-05

One isolated AMD64 Pod with 4 CPU / 8 GiB limits, fresh processes and datasets,
Lance 9.0.0, synthetic unordered IDs. Each row sums three batches (0%, 50%,
100% overlap), excluding initial loading/index creation and the replay check.
All eight fixtures passed complete ID/revision comparison and payload spot
checks after replay; cgroup max/oom/oom_kill counters remained zero.

| Base / batch | Index and delete policy | Three batches (s) | Read MiB | Peak process MiB |
|---|---|---:|---:|---:|
| 262k IDs / 128 rows | BTree eager index + merge_insert | 3.093 | 14.767 | 339.5 |
| 262k IDs / 128 rows | BTree deferred index + predicate | 2.656 | 6.288 | 308.7 |
| 262k IDs / 128 rows | ZoneMap eager index + predicate | 2.177 | 6.876 | 271.4 |
| 262k IDs / 128 rows | ZoneMap deferred index + predicate | 2.210 | 6.899 | 270.5 |
| 256 MiB payload / 64 MiB | BTree eager index + merge_insert | 3.416 | 0.909 | 585.7 |
| 256 MiB payload / 64 MiB | BTree deferred index + predicate | 2.747 | 0.474 | 653.3 |
| 256 MiB payload / 64 MiB | ZoneMap eager index + predicate | 2.962 | 1.380 | 586.8 |
| 256 MiB payload / 64 MiB | ZoneMap deferred index + predicate | 2.913 | 1.435 | 586.8 |

BTree elapsed time fell 14% and 20% in this small run; ZoneMap differences
were small. BTree's large-payload peak RSS increased by about 68 MiB.
These serial component measurements do not quantify fleet throughput or
concurrent conflict reduction. The separate Rust regression runs four shard
merges concurrently for both index types, verifies all updated payload bytes,
and checks physical row counts, empty replay and unchanged index identity.
Long-running tables with large uncovered key ranges still need index
maintenance; this run does not bound their scan amplification.
