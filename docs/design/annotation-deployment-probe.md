# Deployed annotation probe

`annotation_probe` exercises existing annotation HTTP roles, persistent replicated
WAL, and Lance. It checks the single-shard update/read/publication loop before
adding mixed ingestion or asynchronous indexes. The executable is a test client;
it does not run the owner or WAL replicas inside its process.

Build with:

```sh
cargo build --release -p lance-context-ingest --features service,http-client \
  --example annotation_probe
```

Pass an annotation application configuration and a workload JSON file:

```json
{
  "turns": 4096,
  "batch_rows": 128,
  "seconds": 120,
  "phase": "steady",
  "slots": [0, 1],
  "ingresses": ["http://ingress-a:7910", "http://ingress-b:7910"],
  "directory": "/test-results/steady"
}
```

The application configuration supplies the authorized table, WAL endpoints,
credentials and storage options. Use a fresh test table and producer slots.
The client needs access to both HTTP endpoints and the WAL/blob configuration.

1. `annotation_probe seed CONFIG WORKLOAD` creates the full rollout schema with
   16 KiB bodies and score `0.5`, using Lance Create mode. Save the emitted source
   version and transaction as `annotations.source` before running the existing
   `annotation-bootstrap` application role. This is an explicit offline handoff.
2. Start one annotation owner and two forwarding ingress processes.
3. `annotation_probe load CONFIG WORKLOAD` runs two writers over disjoint halves
   of the turn IDs. Each batch sets a status identifying phase, producer slot and
   sequence. After every successful update it reads the first and last batch
   rows through the other ingress, checking status and untouched score/labels.
4. `annotation_probe verify CONFIG WORKLOAD` waits for the managed Lance cut to
   cover every journaled ACK and for the last acknowledged source record to be
   reclaimed. It scans all narrow annotation rows, checking values, missing rows
   and duplicates. It reports the selected version and fragment count.
5. `annotation_probe read CONFIG WORKLOAD` checks 32 evenly spaced IDs through
   each ingress, with concurrency two. This samples point reads; the separate
   `verify` operation checks every row in Lance.

The workload requires an even turn count between 256 and 65,536, divisible by
both writers' batch size. Batches contain 1–512 rows and duration is 1–600 seconds.
`max_batches` can cap each writer for an unmaterialized recovery experiment.

## Crash checks and request identity

The driver creates a new result directory and rejects reuse. Before making an
RPC it saves the producer-open operation or exact producer/sequence/Arrow body;
it syncs each returned ACK to a journal. A mutation resubmission is allowed only
for an explicit HTTP 503 with `operation_outcome=not_submitted`, using identical
request identity and bytes. An unknown result stops that writer and preserves
the pending request. The probe does not automatically reconcile unknown outcomes
or open a replacement producer. Do not rerun an ambiguous workload with fresh
identities and count it as the original experiment.

For owner recovery, first finish and verify a steady phase. Start an owner with
a longer publication timer, then run a new workload with different producer
slots, `max_batches: 1`, and a new result directory. Its `prior_journals` list must
include both steady-phase ACK journals. Confirm that the owner has applied the
new ACKs beyond its materialized cut before sending SIGKILL. Start a replacement
against the same durable state after the owner claim expires; run `read` and
`verify` with the recovery workload. Keep the client ACK journals independent of
the owner process and its staging memory.

## Interpreting results

Per-writer output includes accepted batches/rows, Arrow bytes, explicit refusal
count, and millisecond upper bounds for ACK/read p99. ACK latency includes the
client's pending-request and ACK journal syncs. Aggregate update rate includes
producer opening, journal I/O, cross-ingress reads, and completion of final
in-flight requests. This is a closed-loop correctness workload; it does not
establish an offered-rate saturation limit. Report replica, owner, ingress and
writer counts with every result.

Capture the owner's existing once-per-second annotation observations during the
load: applied/materialized positions, active/frozen entries, retained bytes,
source records/bytes, and backpressure. Sampled staging usage is not process RSS.
A final drain alone does not prove sustained drain; check that the materialized
cut advances during the load and that backlog stays bounded. Collect process
resource observations separately, and report the deployment's actual batch and
memory budgets. This probe changes status only; contract, concurrent field merge,
publication uncertainty and fencing are covered by the separate protocol tests.

## Recorded single-shard run

A 120-second run used three WAL processes on separate hosts with persistent
storage, one owner, two ingress processes, and two concurrent writer loops in a
four-thread Tokio test client. The owner and each ingress had a 4-CPU limit.
Lance 13 stored the table in a real blob store. The workload had 4096 turns,
16 KiB bodies, 128-row update batches and no ID index. Staging limits were
16 MiB / 16,384 entries, with 4096 bytes per value and 32 readers; source selection
was bounded to 64 records / 8 MiB. Publication thresholds were 4 MiB or one second.

- 403,072 status updates completed at 3333.35 updates/second, including durable
  client journals and opposite-ingress reads. No explicit admission refusal or
  mutation rejection occurred.
- The larger per-writer ACK/read p99 upper bounds were 87/40 ms. All 6298
  post-ACK reads passed. These are per-writer percentile bounds, not a merged
  latency histogram.
- Owner observations contained 104 distinct materialized positions. Sampled
  retained staging peaked at 2,251,392 bytes and 4224 active-plus-frozen entries;
  source metadata peaked at 32 records. No sampled backpressure was observed.
  Active/frozen entries and source records reached zero after the load.
- The full 4096-row managed-Lance check passed, with one fragment and reclamation
  of the last acknowledged source record. Verification starts after the load
  process is collected, so its elapsed time is not an immediate post-ACK drain
  latency measurement.
- A separate two-batch phase used a 60-second publication timer. The owner was
  killed with 256 acknowledged values still staged, materialized position 3964
  and applied position 3977. After replacement, 64 sampled cross-ingress point
  reads and the complete 4096-row Lance check passed through position 3977;
  source reclamation completed again.

This establishes drain and owner-recovery behavior for this bounded workload.
It does not qualify large-table lookup cost, arbitrary offered load, body-ingest
capacity, other failure combinations, or a 300-billion-turn deployment. The
closed-loop driver serializes durable journal work in its writer futures; use
separate client and service measurements before attributing a capacity ceiling
to the WAL or owner.
