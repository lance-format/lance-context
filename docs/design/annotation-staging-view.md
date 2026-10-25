# Current annotation view over Lance

Refs [#378](https://github.com/lance-format/lance-context/issues/378).
Protocol: [#564](https://github.com/lance-format/lance-context/pull/564).
Field contract: [#567](https://github.com/lance-format/lance-context/pull/567).

`lance_context_core::annotation_staging` implements the local annotation view
and native file preparation component. It has one active map and at most one
frozen map, backed by a fixed Lance snapshot. It does not yet implement a durable
mutation service. No existing HTTP or immutable ingestion route enables it.

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
map. A captured base miss resolves against that base, even if a new publication
finishes before the I/O. An already captured staged value likewise survives
retirement. There is no WAL-tail scan or body projection in this component.
The service still needs a quorum barrier and application-progress wait before
using it for cross-ingress reads.

## Frozen publication

Freezing moves the active map into an immutable generation without copying its
values. The caller supplies the publication UUID, allowing reconstruction of a
pending cut with its original identity. Further updates use a new active map.
Frozen preparation builds narrow
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

## Bounds

Allocation charges include owned strings and conservative map/set/Arc node
overhead. Both maps, held frozen handles and captured old staged values retain
their byte/entry charges until dropped. Point-read admission separately limits
old base handles. There is no clean-value cache or queue of extra frozen maps.
A capacity rejection leaves the value and applied cursor unchanged. The WAL
consumer must stop at that item until publication/read release permits progress.

The serialized apply operation may temporarily hold an old and new narrow
value; per-value/input bounds limit that scratch allocation. Read outputs and
Lance's own scan/merge buffers require the service's separate request and engine
workspace admission. These charges are not a claim to bound total process RSS.
Preparation reserves a separate shared merge budget and retains its reservation
with the prepared transaction, so old preparation handles also exert backpressure.

## Remaining single-shard service gates

The native tests exercise fields changed during publication, held old reads,
same-value changes with distinct progress, field clearing, retained-generation
and reader pressure, and repeated native publication/reopen. They are component
tests with caller-supplied positions, not RF3 durability or sustained-load proof.

Next connect admission, opaque WAL mutation records, ordered application and
request results to the actual Rust service. Persist frozen coverage using the
existing fenced publication protocol; reconstruct from selected Lance state and
the WAL suffix; test unknown outcomes, process crashes and GC. Finally qualify
sustained update/read/drain behavior before mixed ingestion, filtering indexes
or more shards. In particular, native transaction properties alone do not supply
request-result retention, producer fencing or crash recovery.
