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

## Preparations spanning many fragments

The native transaction grows with the fragments it updates, independently of
the dirty-value budget. A local Lance 13 experiment with 4,096 fragments and one
updated row per fragment produced 1,134,379 bytes of native preparation metadata,
before request-result checkpoints. That exceeds the WAL descriptor limit of
1,048,576 bytes and prevents publication of that frozen generation.

The annotation runtime keeps inline format 4 for transactions within the catalog's
configured descriptor budget (256 KiB by default, at most 1 MiB).
Larger transactions are conditionally created in the same authorized destination
store as their Lance files. Format 5 retains only their table incarnation,
publication identity, byte count and content digest in the quorum catalog. The
object path is derived from that destination and identity; a descriptor cannot
supply an arbitrary URI. This changes preparation metadata, not mutation payloads
or the ACK path. All catalog voters and annotation owners must support format 5
before a publisher emits it; it is not compatible with older voters.

Persistence finishes before Prepare. Unknown upload errors stop the attempt;
only a conditional-create conflict is resolved by reading and checking the exact
object. A durable Prepare always recovers its original bytes and files. Without
one, recovery uses the existing retained-plan protocol. Missing or corrupt
metadata fails recovery without confirming publication or reclaiming its source.
Selected-manifest recovery checks the referenced native transaction against the
manifest, just as it checks an inline transaction.

A 64 MiB limit bounds the raw metadata encoded or fetched by one operation. It
does not bound decoded engine allocations or total process memory, and is
separate from dirty staging admission. Preparations above that limit still fail;
this is not arbitrary-fragment-count qualification. Physical deletion of these
objects, including uploads orphaned before Prepare, remains part of blob
retention work and is not provided by WAL source GC.

Rust regression tests submit real annotation requests to three disk-backed WAL
replicas and update one row per fragment. With 4,096 fragments, they restart after
Prepare, after a lost manifest-selection reply, and after confirmation before GC.
They verify exact original transaction reuse, request-result retries, a concurrent
score-update suffix, untouched rows/body-file mappings and final staging/source
drain. A 1,024-fragment case exercises the default 256 KiB catalog limit below the
1 MiB hard cap. Missing, corrupt and foreign metadata cannot advance source GC.
These are bounded local recovery tests with retained disks, not throughput or
power-loss qualification.

After prepared-state recovery, the 4,096-fragment fixture also submits three more
rounds of updates without waiting for publication between batches. It checks each
ACK's point-read value, bounded staging charges, final values and source drain.
