# Current annotation view over Lance

Refs [#378](https://github.com/lance-format/lance-context/issues/378).
Protocol: [#564](https://github.com/lance-format/lance-context/pull/564).
Field contract: [#567](https://github.com/lance-format/lance-context/pull/567).

`lance_context_core::annotation_staging` implements the local annotation view
and native file preparation component. It has one active map and at most one
frozen map, backed by a fixed Lance snapshot. The
[annotation owner runtime](annotation-runtime.md) supplies durable ordered apply,
request results and publication. HTTP ingresses forward to that owner; see
[Running one annotation shard](annotation-application.md) for role configuration.

## Ordered updates and point reads

The apply caller supplies `(after, through)` from its ordered, deduplicated WAL
consumer. A mismatch with the local applied position is rejected. `through` may
skip control records only after that consumer processes them. This component
cannot verify quorum commit from bare positions; application success is not an
ACK or permission to collect WAL records.

`apply_batch` applies every mutation in one WAL batch before advancing that
batch's position. Repeated keys merge in source order; a failure cannot expose
the earlier part of the batch. This avoids requiring a separate WAL entry for
every turn update. Admission must ensure an atomic batch can fit after draining.

Each patch resolves the complete current narrow annotation from active, then
frozen, then Lance. It preserves fields not mentioned in the patch. A point
update requires an existing turn. The Lance schema must have `id: Utf8`,
`labels: List<Utf8>`, `status: Utf8?` and `score: Float32?`; other columns are
left out of annotation input and point-read projection.

Point reads capture either one staged value or the exact base handle under the
same lock that switches the publication view. They never retain a whole staging
map. For a base-backed read, a compatible published-value cache entry can answer
without storage I/O. A cache miss resolves against the captured base, even if a
new publication finishes before the I/O. An already captured staged value also
survives retirement. There is no WAL-tail scan or body projection in this component.
The service supplies the quorum/application barriers required for cross-ingress
reads. The HTTP owner's recovery cut and authority checks are described in
[annotation HTTP routing](annotation-owner-http.md).

## Registering new turns in the core view

`register_committed` accepts an ordered batch of ID, immutable `rollout_id` and
initial annotation values. It requires a separate five-column annotation table;
it cannot append to a table containing turn bodies. All input is validated before
installation. The first occurrence of an absent ID initializes it. Later valid
registrations of that ID preserve its current annotation and original grouping
key, whether found in this batch, active, frozen, the cache or Lance. Re-imports
of existing IDs create no dirty values. Cold existence lookups project only ID.
Ordinary point patches still reject missing turns.

A newly staged row retains its grouping key and original apply position within
the existing byte/entry limits. Frozen preparation compares that position with
its captured materialized cut: it appends a row only if its first registration
is still unmaterialized. Thus an active update based on a frozen new row becomes
an existing-row update after the frozen batch publishes; it cannot reinsert the
row. This transient metadata can cause backpressure but cannot change a patch's
permanent rejection result when recovery starts from a later published prefix.

Preparation writes existing annotations with `RewriteColumns` and new narrow
rows with the native append writer. A mixed batch combines their file descriptors
in one Lance `Update` transaction; an insert-only batch uses `Append`. Both retain
the same frozen identity and scoped coverage. Lance assigns stable IDs and row
version metadata to the new fragments. Original ID-index coverage is preserved;
new fragments remain uncovered and participate in native filtered reads. A
result-only batch still uses a metadata checkpoint without creating data files.

This is a core component capability, not a mixed-ingestion service endpoint.
The current annotation WAL codec and managed-publication validator remain
update-only. The caller must durably admit the complete turn body, coordinate
annotation visibility, retain request results and recover both parts before
offering a complete ingestion ACK. A bare registration result supplies none of
those guarantees. Tests here cover real Lance preparation/commit/reopen, stable
IDs and index coverage, re-imports during publication, held readers, pressure,
atomic rejection and replay of a suffix against a published prefix. Service
integration and sustained mixed-load qualification remain separate work.

## Frozen publication

Freezing moves the active map into an immutable generation without copying its
values. The caller supplies the publication UUID, allowing reconstruction of a
pending cut with its original identity. Further updates use a new active map.
For existing turns, frozen preparation builds narrow
Arrow batches and explicitly selects Lance 13 `RewriteColumns`, with automatic
conflict retries and cleanup disabled. It rejects a result that does not update
exactly the frozen keys, and records scoped coverage in the native transaction
properties. Repeated preparation calls share the same transaction/file set.

Lance 13's narrow merge also writes the `id` join key, which would invalidate
the ID index on every touched fragment. Preparation retains the original ID
file/column mapping and tombstones the unused ID mapping in the new narrow file.
Only then does it remove ID from the transaction's modified fields. Annotation
indexes still lose coverage where their values changed. The adjustment accepts
only the expected column rewrite with unchanged physical rows and original file
descriptors; it does not rebuild an index or override index metadata. The writer
still emits the unused ID bytes, so this is not a claim of eliminating that I/O.

The caller must durably preserve this identity, source coverage and descriptors,
publish through the fenced managed-manifest path, and reconcile the original
outcome. Only then may it install the selected dataset in this view. Installation
checks the exact native transaction, URI and version, switches the base and
retires only the current frozen generation under one lock. It never removes
active entries by turn ID. Native transaction matching is an additional local
guard, not proof of publication authority or of group-wide WAL coverage.

`install_compacted` switches the base after the caller confirms an exact native
compaction. It requires unchanged scoped annotation coverage and request-result
properties, rejects overlapping frozen publication, and leaves active values
and application progress intact. Captured readers keep their old base registered
for reclamation until released. This view check complements the publisher's
row-preserving rewrite validation and durable publication protocol.

## Binding a separate body snapshot

`prepare_with_body` binds a `BodySnapshot` to the original native preparation.
It records the authorized body URI, native version and transaction identity,
and the logical table incarnation alongside the frozen annotation cut. Capturing
this identity does not scan or copy body rows. Body references and their encoded
metadata consume the existing preparation workspace.

A repeated preparation must use the same reference. A later annotation
publication must retain an explicit body reference with the same destination
and incarnation; it cannot silently drop the reference or change the transaction
identity of an existing body version. Narrow compaction preserves this property
together with the annotation cut and request results. Exact-transaction checks
still apply when installing the selected snapshot, so modifying the reference
after preparation is not permitted.

This is a component for mixed-ingestion integration. The caller must establish
body coverage of the selected source cut, durably retain the original combined
preparation, and protect both datasets' files and reader dependencies before
source retirement or collection. A URI read from metadata is not authorization
to access that destination. The current patch-only managed publisher explicitly
rejects paired body metadata; this API does not enable mixed-ingestion ACKs,
body reads, or a two-dataset collector.

## Bounds

Allocation charges include owned strings and conservative map/set/Arc node
overhead. Both maps, held frozen handles and captured old staged values retain
their byte/entry charges until dropped. Point-read admission separately limits
old base handles. There is no queue of extra frozen maps.
A capacity rejection leaves the value and applied cursor unchanged. The WAL
consumer must stop at that item until publication/read release permits progress.

After confirming a selected publication, installation copies its complete narrow
values into the existing, byte-bounded Lance session metadata cache. Later apply
and point reads check these values after active/frozen and before a cold Lance
lookup. Each staging instance has its own cache namespace, including when
instances share a Session and URI. The copies hold neither staging permits nor base handles;
publication can release dirty capacity and reclaim old files normally.

A cached value records its publication cut. A point read captures the base's
materialized cut together with its base handle, then looks up the cache without
holding the view lock. It accepts only an entry at or before that captured cut;
a newer entry falls back to the original base. Cache replacement precedes each
base handoff, so an older entry that remains present belongs to an unchanged
turn. Each key still holds only its latest cached value. This also keeps an old
reader correct when cache warming is cancelled before the view switch, or when
it resolves after a later publication. Cache hits do not perform a WAL-tail scan
or fetch annotation data files.

Cache warming occurs while the same frozen generation is still installed and
the mutation gate is held. Active/frozen values mask those keys until the atomic
base handoff, so cancellation during warming leaves the previous view valid.
Only this installation writes the cache: delayed reads of old snapshots cannot
overwrite newer values. Row-preserving compaction keeps cached annotations valid.
Eviction, cache clearing, a disabled cache, or a backend without a reported byte
capacity falls back to Lance. This shares the existing metadata budget and can
compete with native metadata; it does not reserve additional cache capacity or
replace reconstruction from the selected Lance state and WAL suffix.

The serialized apply operation may temporarily hold an old and new narrow
value; per-value/input bounds limit that scratch allocation. Read outputs and
Lance's own scan/merge buffers require the service's separate request and engine
workspace admission. These charges are not a claim to bound total process RSS.
Preparation reserves a separate shared merge budget and retains its reservation
with the prepared transaction, so old preparation handles also exert backpressure.

## Service integration and validation

The native tests exercise fields changed during publication, held old reads,
same-value changes with distinct progress, field clearing, retained-generation
and reader pressure, and repeated native publication/reopen. They are component
tests with caller-supplied positions, not RF3 durability or sustained-load proof.

The [ordered applier](annotation-ordered-apply.md),
[result checkpoint](annotation-result-checkpoint.md) and
[runtime](annotation-runtime.md) connect this view to the RF3 WAL and fenced
publication/recovery flow. Native transaction properties alone do not supply
request-result retention, producer fencing or crash recovery. The application
process tests cover owner replacement and physical collection; their scope and
operational restrictions are documented in
[Running one annotation shard](annotation-application.md). Sustained throughput
and drain claims still require workload-specific measurements. Mixed ingestion,
filtering indexes and multiple shards remain later work.
