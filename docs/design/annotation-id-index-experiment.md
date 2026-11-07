# Native turn ID index integration experiment

Refs #378. This candidate is an opt-in implementation under validation, not a
throughput or production-readiness claim. It does not change the WAL payload
format or Lance OSS.

The annotation owner can schedule `turn_id_index` maintenance on the immutable
`id` column of its narrow table. It uses native BTree files and the existing
format-3 descriptors, empty-source plan, fixed target version and fenced
publication/recovery path. Installation preserves the materialized cut, request
results, body reference, active updates and captured old readers. Session index
scheduling remains separate and cannot be enabled on this owner.

The initial policy builds at most 1,048,576 rows from at most 128 fragments.
It starts when the uncovered tail reaches `min_rows` (default 65,536), or 128
fragments. Four segments in the same factor-of-four row-size class may merge,
with at most 16,777,216 input rows. It never selects all historical segments just
because a global segment count was reached. The admitted 512 MiB workspace is
an estimate, not an allocator/RSS guarantee.

Narrow compaction selects groups totaling at most 1,048,576 rows while this option
is enabled. This keeps future fragments eligible for indexing without creating
an ineffective split disguised as compaction. Enabling the option requires
stable row IDs, no deletions, existing fragments within that row bound, and an
annotation entry budget at most that bound. Existing oversized fragments need a
separate migration before this option can be enabled.

The owner derives index/compaction preference from the selected durable version.
A pending maintenance plan resolves before another plan is chosen. With enough
existing publication capacity for both the 512 MiB index estimate and an ordinary
publication, one ID build or merge can run before taking a catalog plan. Data
publications continue while its native I/O runs. Other maintenance waits for that
one job; smaller budgets retain serial maintenance. No additional budget is created.

Before selecting the result, the owner verifies that each covered fragment still
has exactly the original ID file, column mapping and physical row layout. Narrow
annotation rewrites preserve these inputs; newly appended fragments remain outside
the result's coverage. Replaced fragments, changed ID columns, source-index
replacement or overlapping coverage fail validation. Only after this proof is the
ID-only metadata certified for the current snapshot and recorded through the
existing exact-version format-3 preparation and fenced publication. Status indexes
remain on their existing path because their values can change.

Collection continues for data files, protecting the one job's original snapshot.
Index-page collection waits until this job resolves, including files uploaded
before their response completes. No other index job is admitted meanwhile. A crash
before the plan leaves unselected files; a successor reconstructs current values
from its ordinary Lance/WAL checkpoint and may rebuild the index. After preparation
is durable, the existing exact-result recovery applies. Native writers retain
their reservation if the caller is cancelled and cannot publish a manifest themselves.

Qualification must include real owner create/patch traffic, native build and
merge transactions, compaction, exact request-result recovery and lost Prepare
replies. This candidate also extends the existing closed-page collector to
`_indices/<uuid>/page_lookup.lance` and `page_data.lance` in the narrow table.
It protects raw index UUIDs from retained manifests and locally held read views,
including native versions hidden from query planning. Index-root decoding has
separate manifest-size admission and shares the existing weighted session cache.
Unfamiliar index layouts and body-table indexes keep their previous retention.
No index candidate is classified during pending preparation/publication.

Long-running storage bounds, cloud throughput, cold-read cost, capped-segment
growth at large shard sizes, and multi-shard scaling remain unproven. Keep the
option disabled by default; this candidate alone does not close those tasks.
