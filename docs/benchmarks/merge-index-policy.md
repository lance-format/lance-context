# Configurable WAL merge key index

`ROLLOUT_KEY_INDEX_TYPE=btree|zonemap` selects the base-table key index for
rollout, generic and datagen stores opened by workers. Masters use the same
setting for scheduled `IndexId` and propagate it to native catch-up Jobs.
The default remains `btree`. Core callers can set `key_index_type` in their
store options. This does not change ContextStore's separate ID-index setting.

Use the same setting on every writer/maintenance process for a table. An
existing `id_idx` of a different type is replaced during merge/index
maintenance, not just by opening the table. Roll out consistent configuration
before admitting maintenance with the new policy; otherwise mixed policies
can repeatedly replace the index. Selecting ZoneMap does not remove other
user-created indexes.

## Why index selection also changes the delete path

With Lance 9.0.0, `merge_insert` requires an exact-answer index for its indexed
lookup path. ZoneMap does not qualify. A key-only source with
`WhenMatched::Delete` still produces a target scan projecting payload columns
on its non-indexed path. Both a Rust execution-plan probe and an Azure storage
benchmark reproduced this. Changing only `IndexType::BTree` to `ZoneMap`
therefore reintroduces payload scan amplification.

The ZoneMap policy uses `DeleteBuilder::from_expr` with typed ID literals,
allowing the predicate scanner to prune zones and read only keys. Expressions
are bounded to 1,024 IDs per delete; every delete commits before the prepared
rows are appended. The original WAL remains until the append and manifest
drain succeed. A retry deletes the same keys and appends their newest values
again. Existing table ownership, commit fencing, WAL byte limits and the shared
merge reservation are unchanged. Many small keys can require multiple delete
commits; this tradeoff must be measured for the intended workload.

BTree keeps the existing merge path and index-extension behavior. An `index`
phase metric now separately measures index preparation inside the encompassing
`append` phase; do not sum nested phase durations as independent work.

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

Compare `BENCH_KIND=btree` (existing merge) with `zonemap_predicate` (new path).
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
