# Bounded status index maintenance

Annotation owners can opt into `publication.status_index` to build native BTree
segments on the selected narrow table. It defaults to disabled. All owners that
can take over the table must understand status index publication before enabling
it. This adds no WAL record or index file format.

Each build covers at most 1,048,576 rows and 128 uncovered fragments. Even a small
idle tail is eligible, without waiting for another update. Index work shares the
existing maintenance budget and fenced publication path with ID indexing and
small-file compaction. Preference rotates from the selected version, so owner
replacement does not reset scheduling. New updates continue through active
staging and do not wait for an index to become current.

When a data batch is ready, a status-only build below `min_rows` (65,536 by
default) yields to that batch. A tail with 128 uncovered fragments remains
eligible. Native index merges, ID index work, and body or narrow compaction can
still take a maintenance turn; this threshold does not delay an already retained
plan. The deferral is not cached as no work for the selected version, so an idle
owner still catches up a smaller tail without another write. Setting `min_rows`
to 1 preserves immediate eligibility during sustained writes.

Merge admission uses the physical sizes recorded in native index metadata,
including old values outside a segment's current coverage. Four similarly sized
segments may merge within `max_merge_bytes` (64 MiB by default). These are bounds
on each scheduled operation, not a bound on total index size or native allocator
usage. Large shards can retain segments that exceed the merge bound; placement
and shard sizing remain separate work.

Lance filters each merge source by its own effective coverage. Data updates prune
empty siblings but retain a final empty index definition. Native `CreateIndex`
does not retire that definition implicitly. A bounded catch-up build explicitly
removes up to 64 proven-empty status definitions in the same publication, without
reading their obsolete pages. Recovery validates the exact source identities,
empty coverage and fresh build version using the existing format-3 descriptor.
There is no additional history store or retirement protocol.

The native index describes a selected materialized snapshot. Immediate point
reads still consult active/frozen staging and the compatible base. An API for
asynchronous filtering must expose its materialized snapshot and lag; enabling
index maintenance alone does not provide immediate indexed filtering of staging.

Index transactions preserve the annotation cut, request results and body
reference. Recovery decodes the index identity from the retained native
descriptor, so a prepared status operation can resolve even if maintenance is
disabled on the replacement owner. Existing retained-reader and manifest rules
govern later physical index cleanup.
