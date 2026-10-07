# lance-context-ingestion

Experimental streaming ingestion primitives. The implementation currently provides
the pipeline, journal protocol and an optional Lance table sink. Application
alignment/source adapters and deployment orchestration remain separate. This crate
does not provide a complete distributed ingestion service.

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

- `Partition::start_with_aligners` can run several session alignment lanes inside
  one stable durable partition, independently of history-loader concurrency and WAL
  batch size. Same-session calls stay ordered; completed lanes rejoin the original
  sequence before publication. Changing lane count on restart reroutes replayed
  session deltas without changing partition identity. An error or cancellation
  stops all speculative lanes. Lane count must fit the queue-entry budget; adapter
  caches are additional to the shared input/output byte budget. This does not
  schedule workers on other machines or remove the WAL ordering barrier.

- Named `Consumer`s have independent durable cursors and coalesce producer segments
  into their own batches. Sink output and input coverage must be committed together;
  cursor writes can fail after output succeeds, so repeated/regrouped input must be
  idempotent. The scheduler owns exclusive consumer assignment and sink-side fencing.

- `ReceiptIndex` resolves exact source receipt/session/input-digest identities after
  an HTTP retry or restart. A dedicated consumer writes immutable receipt mappings;
  `find_many` reads the index concurrently, then reconciles its unindexed WAL suffix
  once for the whole requested batch. It handles index writes ahead of the cursor
  and rejects a receipt reused at another sequence. A miss covers only the returned
  committed position: the admission owner must still check its in-flight map and
  serialize sequence assignment. This index does not schedule source fan-out or
  replace the requirement to ACK every partition before advancing source progress.

- `SourcePartition` adds a serialized admission owner for continuous callers that
  have stable source receipts but no partition sequence numbers. `enqueue_many`
  checks committed and in-flight identities, assigns sequences only to new inputs,
  and returns independent ACK waiters. A repeated in-flight receipt watches the
  original commit without blocking later alignment dispatch. Dropped ACK waiters
  do not cancel work; cancellation during admission requires recovery of the
  unknown admitted prefix. Run its dedicated receipt consumer independently and
  keep HTTP/source queues bounded. This API does not supply HTTP authentication,
  cross-partition fan-out, or a migration from an application's previous WAL format.

- `Writer::with_backlog` optionally limits committed segments outstanding for
  every required consumer. A consumer that has not started is at zero; table and
  checkpoint progress are both required when both are configured. The publisher
  waits before writing another segment, retaining bounded pipeline reservations;
  other partitions and already durable retries remain independent. Consumers must
  continue running while producers drain. Cancellation or ownership transfer fences
  a paused writer. Reapply the same policy on every acquire/restart. The gate checks
  durable cursor metadata and its committed ancestry; it adds storage reads and is
  not itself a throughput optimization. It bounds unconsumed payload bytes by
  `max_segments * max_segment_bytes`, not retained history, orphan uploads or total
  storage. No WAL garbage collection or scheduling is implied.

- `SessionCheckpoints` provides an actual object-store checkpoint sink: group by
  session, reduce ordered deltas, then write each session once with a conditional put.
  A partially successful checkpoint batch can leave some session states ahead of the
  global cursor. Restore each state with its own sequence and skip already applied
  deltas when replaying the remaining global prefix.
  `Reducer::apply_batch` receives only a session's ordered, unapplied deltas and
  lets an adapter decode its state once and encode it once per consumer batch.
  Its default preserves per-delta `apply` behavior and intermediate size checks;
  overrides must preserve those semantics and bound intermediate state themselves.
  The sink also checks final output size before writing. This interface alone does
  not accelerate existing reducers or change the deployed ingestion adapter.

- `SessionCheckpoints::recover` reconstructs one evicted or cold session from the
  checkpoint consumer cursor plus committed WAL suffix, one segment at a time.
  It returns the captured WAL position separately from the session mutation
  sequence and skips partially published checkpoint deltas. This is read-only;
  callers must still reconcile their newer speculative state and bound the cache.
  Use the checkpoint consumer name, never an unrelated table consumer cursor.

- With the `lance` feature, `lance_sink::stage` writes immutable Lance 2.2 files
  using a Zstd-annotated schema. Lance's constant-valued pages use scalar
  encoding before codec selection; even a single large string can take that path. `LanceTableSink::commit_staged` coalesces staged
  partitions and atomically publishes rows plus each partition's covered sequence.
  Fully covered retries are skipped; gaps and partial overlaps are rejected.
  A caller-supplied ownership-guarded commit handler is wrapped by an exact version
  pin to reject implicit rebasing. Uncertain publication poisons the sink; reopen
  and reconcile the table watermarks before retrying. The sink does not acquire
  table ownership or schedule index maintenance.

Use a backend supporting atomic conditional updates, such as a suitably configured
cloud object store. `object_store::local::LocalFileSystem` does not implement the
required update operation and is rejected; there is no unsafe local-lock fallback.
Tests use the real `object_store::memory::InMemory` implementation for CAS behavior,
with fault injection around writes. These tests do not establish cloud durability,
process-crash behavior on a real durable service, or production throughput.

## Bulk sources

For replayable batch sources, use `SourcePartition::start_batched` with a fresh
`BatchFlush` shared by the partition's alignment adapters. Pass the existing source
batch to `enqueue_many`; do not replace its stable record receipts when combining
batches for transport or WAL publication. The admission owner validates the entire
batch, preserves in-flight deduplication, and requests a flush through its final
assigned sequence. An all-duplicate batch does not manufacture a WAL entry.

This mode does not use `BatchPolicy::max_delay`. WAL collection continues until a
source batch boundary, the byte/count limit, output memory headroom, or shutdown.
A later already-admitted batch boundary can coalesce available batches. Limits may
split a large batch into committed prefixes: rows, state deltas and source receipts
remain together in each segment, and callers await **all** returned ACKs before
advancing source progress. Table/checkpoint consumers still group segments
independently. The bulk constructor changes scheduling, not the WAL format.

Same-session alignment sees its speculative state throughout the batch. If an
adapter evicts uncommitted state and needs to recover it, call
`BatchFlush::request_prefix(sequence)` before waiting for that prefix's durability.
This flushes available ordered outputs even when a later batch boundary is pending;
otherwise the blocked alignment could prevent the batch itself from completing.
Do not treat a stale checkpoint as the current batch's state. Cache/input/output
budgets still apply; increasing a batch target does not permit unbounded memory.

## Remaining integration

1. Adapt the existing revisioned session history/cache and cross-call alignment
   implementation. Preserve its compaction/branch identity rules and source ordering;
   the generic pipeline deliberately does not invent a new turn-ID algorithm.
1. Wire table ownership and independently scheduled session ZoneMap maintenance.
   Staging is independent of publication; a distributed worker transport must
   carry validated staged results and retain ownership through manifest commit.
1. Add source fan-out receipts and contiguous source progress. A source call spanning
   partitions is complete only after every required partition ACK.
1. Add worker ownership orchestration, stage timing/queue telemetry, consumer run loops,
   deployment of the backlog policy and safe WAL reclamation. No WAL files are deleted here.
1. Verify real compacted sessions, process crash/restart with durable storage, Lance
   uncertain commits and source retry integration before a guarded production handoff.
   Existing source-reader local audit history must survive that handoff.

Run focused checks from the workspace root:

```sh
CARGO_TARGET_DIR=/tmp/trace-streaming-target cargo test -p lance-context-ingestion --features lance --offline
CARGO_TARGET_DIR=/tmp/trace-streaming-target cargo clippy -p lance-context-ingestion --features lance --all-targets --offline -- -D warnings
cargo fmt -p lance-context-ingestion --check
```
