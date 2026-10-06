# lance-context-ingestion

Experimental streaming ingestion primitives. The implementation currently provides
the pipeline and journal protocol; integration with the existing trace alignment
adapter and a Lance table sink is **not complete**. This crate is not deployed.

```text
replayable source / stable partition-local receipts
    -> bounded concurrent history prefetch
    -> ordered alignment within each virtual partition
    -> bounded WAL queue / byte, count, timer flush
    -> immutable payload + skip links + conditional head publication -> ACK
             |                                  |
             v                                  v
    checkpoint consumer                  table consumer
    coalesced session deltas             independently grouped WAL ranges
             |                                  |
    conditional session states           staged fragments / fenced manifest commit
                                                |
                                         asynchronous indexer
```

## Implemented contracts

- `Partition::start_with_loader` overlaps history loading, alignment and WAL
  publication. Different virtual partitions run independently. Within a partition,
  load completion may reorder, while alignment and durable publication stay ordered.
  Do not change session-to-virtual-partition routing when changing worker count.
- `enqueue` reserves bytes before admission. A reservation remains held through the
  durable ACK, covering input, history, bounded output and serialization. Adapter
  caches, transient adapter allocation, executor and storage-client memory need their
  own budgets. Sources must also bound concurrent requests waiting for admission.
- `Aligner` owns the session cache and speculative state. Prefetched checkpoints may
  be stale: reconcile their revisions with pending deltas. On failure, discard the
  adapter and recover; speculative changes must never become checkpoints directly.
- `Journal` stores immutable segments binding records, state deltas, exact input
  digests, receipt identities, run/schema and predecessor sequence. Conditional head
  updates fence stale writers. Failed or cancelled commits poison a writer.
- ACK follows head publication. An uploaded orphan is not committed. Retry the
  same partition sequence, receipt, session and input bytes after an uncertain result.
  A gap is rejected; older receipts are compared with the committed journal.
- Immutable skip links support bounded chronological recovery pages and historical
  receipt lookup without reading unrelated record payloads. Recovery replays all
  pages after the adapter's checkpoint, not just one next WAL segment.
- Named `Consumer`s have independent durable cursors and coalesce producer segments
  into their own batches. Sink output and input coverage must be committed together;
  cursor writes can fail after output succeeds, so repeated/regrouped input must be
  idempotent. The scheduler owns exclusive consumer assignment and sink-side fencing.
- `SessionCheckpoints` provides an actual object-store checkpoint sink: group by
  session, reduce ordered deltas, then write each session once with a conditional put.
  A partially successful checkpoint batch can leave some session states ahead of the
  global cursor. Restore each state with its own sequence and skip already applied
  deltas when replaying the remaining global prefix.

Use a backend supporting atomic conditional updates, such as a suitably configured
cloud object store. `object_store::local::LocalFileSystem` does not implement the
required update operation and is rejected; there is no unsafe local-lock fallback.
Tests use the real `object_store::memory::InMemory` implementation for CAS behavior,
with fault injection around writes. These tests do not establish cloud durability,
process-crash behavior on a real durable service, or production throughput.

## Remaining integration

1. Adapt the existing revisioned session history/cache and cross-call alignment
   implementation. Preserve its compaction/branch identity rules and source ordering;
   the generic pipeline deliberately does not invent a new turn-ID algorithm.
2. Connect the table consumer to public Lance staging/commit APIs. Persist exact WAL
   input coverage with table publication; staged files alone do not justify a cursor
   advance. Preserve Lance 2.2, Zstd and session ZoneMap configuration in that adapter.
3. Add source fan-out receipts and contiguous source progress. A source call spanning
   partitions is complete only after every required partition ACK.
4. Add worker ownership orchestration, stage timing/queue telemetry, consumer run loops,
   bounded durable backlog and safe WAL reclamation. No WAL files are deleted here.
5. Verify real compacted sessions, process crash/restart with durable storage, Lance
   uncertain commits and source retry integration before a guarded production handoff.
   Existing source-reader local audit history must survive that handoff.

Run focused checks from the workspace root:

```sh
CARGO_TARGET_DIR=/tmp/trace-streaming-target cargo test -p lance-context-ingestion --offline
CARGO_TARGET_DIR=/tmp/trace-streaming-target cargo clippy -p lance-context-ingestion --all-targets --offline -- -D warnings
cargo fmt -p lance-context-ingestion --check
```
