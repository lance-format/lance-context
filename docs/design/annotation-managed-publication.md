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

The [ordered applier](annotation-ordered-apply.md) and
[publication runtime](annotation-runtime.md) use this adapter to bind successful
application, durable request results and source retirement to the selected cut.
The [owner application](annotation-application.md) supplies cross-ingress routing
and fenced recovery. The adapter's bounded Rust tests exercise real Lance files
and three disk-backed WAL replicas over TLS; they do not establish production
mutation capacity.

The tests reopen every WAL runtime after durable preparation, after manifest
selection with its reply discarded, and after confirmation before source closure.
They recover the exact descriptors, replay the catalog-discovered suffix, finish
both publications, and reopen again after source GC. Full body bytes, one row per
key, retained staging charges and current file references are checked. Repeated
updates in this fixture leave one body file and one current annotation file per
fragment: native manifest construction removes fully superseded column-file
references. Physical deletion of historical blob objects is a separate retention
operation. Separate runtime and application process tests exercise SIGKILL and
replacement with retained storage. Sustained-load qualification requires its own
workload, latency and drain measurements.

## Resolving an uncertain confirmation

A structured unknown Confirm holds the current publishing generation while the
owner resolves that exact publication. It first reads the quorum catalog under
the same authority and table incarnation. An existing matching publication
receipt completes the call. Otherwise, only the unchanged pending plan, durable
descriptors and selected manifest allow resubmitting the identical idempotent
Confirm, with 100 ms between unresolved attempts. It does not upload files again,
create a new generation or resubmit an annotation mutation. The bounded applier
and owned point reads remain live while confirmation waits; source/staging
pressure still prevents acknowledging unapplied writes. Cancellation preserves
the selected files and retained WAL for fenced restart. Explicit rejection,
changed identity and failed quorum observation keep their error behavior.

Native tests cover committed and uncommitted confirmation, fencing during each,
concurrent status/score updates, sustained uncertainty with source backpressure
and owned reads, and cancellation followed by recovery of the original files.
These tests do not establish sustained mixed-load throughput.

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
this is not arbitrary-fragment-count qualification.

For an exclusively owned annotation table, `annotations.exclusive_reclamation`
also collects resolved preparation objects and uploads orphaned before Prepare.
The serial collector first resolves any retained publication plan. It only
classifies metadata paths for the current table incarnation and seals each
bounded deletion page through the existing quorum fence. Selected Lance versions
retain their native transaction files under the existing manifest and reader
protection; they no longer depend on the preparation sidecar. Unresolved DELETE
requests keep their original sealed page and pause publication. WAL source GC
alone does not remove these blob objects; collection remains opt-in under the
exclusive-ownership contract described in `annotation-application.md`.

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
With exclusive reclamation enabled, it also verifies preparation-object removal,
original request results and body bytes after recovery. An insufficient collection
workspace is rejected before deleting anything, preserving the acknowledged
suffix's recovery sources. A publication may finish before a later collection
pass reaches that refusal. Tests require any uncovered suffix to remain in the
quorum source catalog, and reopen to verify original request results and current
values for both unpublished and already published suffixes.
