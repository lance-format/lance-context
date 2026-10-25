# Managed annotation preparation

The ingestion crate's `annotation_publication` adapter preserves a frozen
annotation generation's native Lance transaction in WAL prepared metadata. It
uses the existing catalog plan, preparation identity, managed manifest selection,
confirmation and signed source closure; it does not introduce another commit
protocol or a WAL payload format.

The first supported selection contains one shard in one source group. Its
transaction cut uses the table incarnation, source group and shard, and must
match the selected prefix. The existing source pin still has to verify that the
selection includes every relevant source record. Selecting one shard does not
by itself prove coverage of an entire shared group.

Preparation metadata contains the original native protobuf transaction,
including file references, modified fields and matched row offsets. Recovery
uses those descriptors against the original base instead of running the merge
again. The adapter rejects changed identities, versions, cuts, non-column
updates, row membership changes and references that replace body columns.
Replicas keep these descriptors opaque and enforce their size, digest and
annotation policy; application validation does not prove remote file contents.

An unknown commit response preserves recovery evidence. If a scoped quorum
catalog read finds the exact selected manifest, `selected_cut` validates its
native transaction against the recorded preparation. A restarting application
can then build staging from that base and the WAL suffix, including before the
catalog has completed confirmation and closure. Confirmation remains necessary
before source GC. Uploaded files or a blob listing cannot establish selection.

This adapter is not a mutation endpoint or a background applier. ACK timing,
durable request outcomes, cross-ingress point reads and sustained draining under
pressure remain separate single-shard integration gates. Its bounded Rust tests
exercise real Lance files and three disk-backed WAL replicas over TLS; they do
not establish production mutation capacity.

The tests reopen every WAL runtime after durable preparation, after manifest
selection with its reply discarded, and after confirmation before source closure.
They recover the exact descriptors, replay the catalog-discovered suffix, finish
both publications, and reopen again after source GC. Full body bytes, one row per
key, retained staging charges and current file references are checked. Repeated
updates in this fixture leave one body file and one current annotation file per
fragment: native manifest construction removes fully superseded column-file
references. Physical deletion of historical blob objects is a separate retention
operation. Abrupt process-crash and sustained-load qualification remain pending.
