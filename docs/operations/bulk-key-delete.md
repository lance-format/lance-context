# Opt-in bulk key deletion for rollout upserts

`CATCHUP_BULK_KEY_DELETE_TARGETS=table_a,table_b` selects exact rollout table names
on a native master or catch-up executor. It defaults to an empty list. Generic
stores and the append-only merge implementation do not use this switch.

For selected upsert merges, deletion joins the deduplicated WAL keys against the
base key column once, with insertion disabled and index lookup disabled. It does
not read or rewrite existing blob columns. The existing serial append, maintenance
fence, watermark checks, byte reservations and surgical WAL drain still apply.
A conflict fails the pass through the existing recovery path; it does not bypass
fencing or retry a stale publication. Replaying after a failed append retains the
same delete-then-append semantics as predicate deletion.

Qualification must use the actual table/index profile. In a local Lance 13 study
with 1,179,648 rows, 4,608 fragments, 1,024 keys and 128-byte synthetic payloads:

| Index coverage | Predicate delete median | Bulk delete median |
| --- | --- | --- |
| BTree, all fragments | 0.221 s | 0.645 s |
| BTree, 25% of fragments | 5.301 s | 0.732 s |
| ZoneMap, all fragments | 7.504 s | 0.559 s |

Each cell is the median of two alternating-order runs. Both methods preserved
all unrelated keys and payloads and tolerated repeated deletion. These are local
synthetic deletion-stage measurements, not Azure or total merge speedups. Complete
BTree coverage benefits from targeted lookups and regresses under this switch;
no global default or automatic policy is inferred from this experiment.

Before rollout, validate actual Rust runtime upserts/blob readback and fencing,
then compare delete duration, whole-pass throughput and process memory on the
selected table. Retain a pinned image and the same byte budget for comparison.

`GET /api/v1/experiments/{name}/key-index-coverage` reads only the base manifest
and index metadata for one rollout table. It reports the base version, active
fragment count and index kind/coverage, unioning deltas with the same name.
Missing coverage metadata stays unknown. Fragment-reuse mappings also produce
unknown coverage: the diagnostic deliberately does not open those potentially
large mapping files or legacy vector-index files to derive coverage. It never opens rows or WAL payloads,
updates stats, or takes a writer lock. One lookup per master is admitted at a
time; concurrent lookups return 429, and a lookup has a 10-second read deadline.
This endpoint adds no polling. Check this evidence before selecting production
tables; a recently completed index task alone does not prove current coverage.
Coverage describes manifest membership, not whether the query planner can use
an index version or whether that index is selective for the current keys.
