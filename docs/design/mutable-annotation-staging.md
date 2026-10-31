# Publishing mutable annotations without losing concurrent updates

Status: proposed protocol, not implemented or qualified. This document does not
identify a defect in the existing immutable append path.

Tracking: [#378](https://github.com/lance-format/lance-context/issues/378).

## Scope and current implementation

A turn has one externally visible record and one current annotation. Its body is
immutable; annotations can change. Internal ordering and temporary publication
generations do not create business history or additional visible turns.

Today, [`TableBinding::contract`](../../crates/lance-context-ingest/src/lib.rs)
selects `WritePolicy::ImmutableUniqueIds`, the only policy in
[`IngestContract`](../../crates/lance-context-wal/src/ingest_contract.rs).
The [session reader](../../crates/lance-context-ingest/src/reader.rs) combines WAL
and Lance rows by immutable ID. Neither contract implements mutable annotations.
Repeatedly appending the same turn ID through that API is not this protocol.

The first mutable path should support field assignment and explicit clearing.
An omitted field is unchanged; an explicit null clears a nullable field. Each
update produces the complete current **narrow annotation** in staging. Bodies,
tokens and artifact buffers do not belong in that value. Physical annotation
storage must support replacement by turn key; the existing append transaction
cannot simply append replacement rows and expose both.

## The lost-update race

The body of turn `T` stays unchanged throughout:

| Step | Action | Current staging annotation | Lance annotation |
| --- | --- | --- | --- |
| 1 | Persist and acknowledge status update | reviewed / null | pending / null |
| 2 | Capture a batch for publication | reviewed / null | pending / null |
| 3 | Persist and acknowledge score update | reviewed / 0.9 | pending / null |
| 4 | Publish the captured batch | reviewed / 0.9 | reviewed / null |
| 5 | Incorrectly remove staging by turn ID | absent | reviewed / null |

Step 5 loses online visibility of an acknowledged update. Retained WAL might
permit reconstruction, but does not make the resulting online read correct.
Comparing annotation values is insufficient: changing a value and changing it
back still advances request/application progress.

## One active generation and at most one frozen generation per shard

Here a shard is a stable session bucket scoped to a table incarnation and its
owning replication group. Its position `C` is a position in that group's log,
not a global sequence or an independent counter shared by different groups.
Routing must keep every update of a turn in the same shard. Bucket reassignment
must not proceed without transferring and fencing this state.

1. **Capture.** In the ordered annotation apply path, select a position `C`
   that is durably committed and fully applied. Freeze all dirty annotation
   state through `C`, assign a generation identity, and install an empty active
   generation. Freeze by transferring ownership of the generation; do not copy
   every turn or its body synchronously. No second frozen generation can be
   created while the first is pending.
2. **Continue applying.** Updates after `C` enter active. Resolve the preceding
   full annotation from active, then frozen, then the compatible Lance base.
   Merge only the supplied fields and store the full narrow result. An update
   that must fetch the base cannot race another update of the same turn and
   overwrite it with a result computed from stale state. Asynchronous base I/O
   must not block the shared Raft event loop; serialize or revalidate the
   per-turn apply operation before installing its result.
3. **Prepare and publish.** Materialize the final annotations in frozen in
   bounded batches. Bind the immutable publication to its generation, scoped
   cut, table incarnation, publisher fence and exact Lance transaction. Confirm
   that the selected manifest contains all relevant updates through `C` before
   advancing the materialized boundary. Uploading files is not publication.
4. **Install the new read view.** Open and validate the selected Lance base,
   then atomically replace `(old base, frozen, active)` with `(new base, active)`
   for new reads. Publication on another process alone cannot clear local
   staging. An ingress must install the compatible base/view before dropping
   its overlay, or route the read to an owner that has done so.
5. **Retire that generation.** Release only the identified frozen generation
   after publication and read-view handoff. Active entries with the same turn
   IDs remain untouched. Existing readers may retain the old base and frozen
   until they complete, subject to the memory and retention budgets below.

In the example, step 3 creates `active[T] = reviewed / 0.9` using frozen as the
prior annotation. Publishing `frozen[T] = reviewed / null` cannot remove or
replace that active entry. A later batch publishes the current annotation.

## Read and acknowledgement boundaries

For a point read, resolve `active → frozen → Lance` within one compatible view.
A staging hit is sufficient only because it contains the full narrow annotation.
If staging stores field patches instead, all applicable layers must be folded in
order; returning on the first patch hit is incorrect.

Capturing three unrelated pointers is not a compatible view. For example, a
reader must not miss active, then observe frozen already removed, then use an
old cached Lance handle. Capture the selected annotation or its absence together
with the base snapshot under the same view synchronization. A miss must keep
that base alive across asynchronous I/O. Publication must synchronize with both
read capture and ordered apply. Do not clone an entire active map per point read
or per update just to obtain a snapshot.

Success for a mutation requires durable quorum commit **and** application of its
result to a queryable view. Another ingress must route to the shard owner or
obtain a quorum read barrier and wait for annotation application through that
barrier. Raft log application alone does not prove an independently maintained
annotation view has caught up. A leader that cannot establish the read barrier
must return an error, not silently serve a stale base.

Independent shards can be queried with bounded concurrency. A session query
must reconcile annotations by turn key and emit one current turn; immutable-row
overlap removal alone is insufficient. Annotation predicates must be evaluated
against the reconciled annotation, otherwise a stale base value could exclude a
turn before its staging update is considered. A multi-shard query does not imply
a global snapshot or cross-shard transaction.

## Durable publication, recovery and reclamation

The existing [publication protocol](../../crates/lance-context-wal/src/replication/publication.rs)
provides publisher fencing and durable plan/preparation/confirmation identities.
The [managed manifest handler](wal-managed-manifest-handler.md) resolves exact
selected manifests. The [materializer](../../crates/lance-context-ingest/src/materializer.rs)
separates confirmation from source closure and advancement. Reuse those
boundaries, but extend their validated contracts for mutable annotations before
using their completion as permission to retire annotation state.

A durable publication must bind the selected Lance version to the table
incarnation, frozen generation and scoped coverage positions. Recovery cannot
depend on an in-memory "flush complete" flag. An unresolved frozen boundary
must be recoverable from the original durable plan and retained state/logs;
takeover cannot reinterpret its cut using newly arrived updates.

When a group contains several shards or tables, publishing one shard through
`C` does not retire other shards' records through that group position. Either a
group coverage proof includes every relevant shard of the table, or coverage
remains explicit per shard. Existing `Progress::checkpoint` positions are per
source group: they must not be advanced from a narrower proof. A publication
covering several groups carries a vector of positions, never their maximum.

| Failure boundary | Required recovery |
| --- | --- |
| Before publication | Retain or reconstruct frozen and active from the last confirmed base and committed WAL; continue the original pending plan if one exists. |
| Publication outcome unknown | Query the original transaction's selected result. Preserve frozen and recovery evidence until the result is established; no fresh identity or assumed success. |
| Manifest selected but confirmation lost | Resolve and validate that manifest and its original coverage, finish confirmation, then recover using the established base/cut. |
| Publication confirmed, local view/retirement incomplete | Install the confirmed base with the correct WAL suffix; do not apply already covered mutations again. |
| New writes after the published cut | Replay or retain those writes in active; old-generation retirement cannot cover them. |

Request deduplication/results and annotation application progress must survive
independently of whether a turn still occupies staging. Equal annotation values
are not evidence that requests were processed. Recovery must preserve the result
of an exact retry and must not treat a materialized mutation as a new request.
The mutation API must define bounded result retention and rejection of expired
identities before enabling writes; this is recovery metadata, not permanent
turn history.

Materialized coverage is only one physical WAL collection condition. Reader
pins, request-result retention, recovery checkpoints, index consumers and
replica/snapshot catch-up may retain additional dependencies. The collector
must satisfy every applicable dependency; it cannot truncate an entire shared
group log solely because one annotation generation was published. Old Lance
files likewise remain available to readers holding that version.

## Bounded memory and large batches

Budget active and frozen state, queued/committed-but-unapplied writes, retained
reader views, request results, and publication workspace. Reserve write capacity
before accepting an operation that needs it. A committed operation cannot be
dropped because allocation later fails. Account for actual retained buffers and
map capacity, not only the number of dirty turns.

Once the budget fills while frozen is stalled, backpressure new writes. Reserve
enough capacity for publication, recovery and view retirement to make progress.
Long-lived readers must consume bounded admission and retention; one active plus
one frozen pointer does not bound memory if old reader-held generations grow
without limit. No unbounded queue of extra frozen batches is permitted.

Coalesce repeated changes to a turn within each generation. Prepare large Arrow
batches from the narrow values, keeping immutable bodies out of the mutation
encoding and publication input. Replicas can retain opaque encoded payloads;
the annotation apply path still has to interpret the changed fields. Sharing
Arrow/Bytes ownership removes avoidable copies but is not a claim of zero
encoding or zero physical I/O. Avoid converting payloads to per-row JSON or
decoding the unchanged body to update a score.

Batch size and flush delay must be configurable and measured with concurrent
updates and reads. Compact eligible small annotation files through fenced
publication without advancing mutation coverage. Physical narrow-table versus
column-update layout needs a real Lance experiment before selection: measure
body bytes read/rewritten, current-turn query cost and compaction work. Do not
infer mutable ingestion throughput from the existing append benchmark.

## Implementation sequence and acceptance tests

All items below are pending. Tests must exercise the actual Rust path with real
WAL and Lance storage, using barriers/failpoints for deterministic interleavings.

1. Define the narrow mutation schema and table policy, explicit field clearing,
   scoped request/result identity and ordered application. Keep the immutable
   append policy explicit. Qualify concurrent updates from separate ingress
   processes, not only one local caller.
2. Implement generation capture and compatible point reads. Pause publication,
   apply `status=reviewed`, freeze, apply `score=0.9`, publish/retire the first
   generation, and assert both changes remain. Repeat with two writes to the
   same field and with a value changed back; verify request progress as well as
   the final value. Test a held old-view miss and a new-view read across handoff.
3. Integrate narrow Lance replacement, durable coverage and recovery. Kill and
   restart processes before publication, after an unknown result, and after
   confirmation but before retirement. Verify one visible turn, both field
   updates, exact request results, recovery progress and the post-cut WAL suffix
   without producer replay. Fence the old publisher after takeover and reject
   foreign table incarnations or incomplete shard coverage.
4. Qualify bounded session-query reconciliation, backpressure and compaction.
   Stall materialization while writing until admission closes; assert bounded
   retained memory and resumed progress after release. Hold old readers during
   repeated publications to test retention accounting. Run mixed update/read
   load with large bodies but narrow annotations, recording acknowledged and
   published updates/s, ACK/read p99, backlog, allocation/copy volume, batch/file
   sizes, body I/O and compaction debt. Preserve correctness assertions during
   the performance run.

Each implementation PR should reference #378 and state which of these acceptance
conditions it actually verifies. A design review or an immutable append test
does not qualify this mutable path.
