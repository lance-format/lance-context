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
both writers' batch size. Batches contain 1–2048 rows and duration is 1–600 seconds.
`max_batches` can cap each writer for an unmaterialized recovery experiment.

For indexed point lookups, add `"id_index": true` before seeding. Seed builds an
ID B-tree and emits `id_index_uuid`; copy that UUID into `expected_id_index` in
both the steady and recovery workloads. The source version/transaction must come
from this indexed seed. After publication, `verify` additionally requires the
same index UUID, coverage of every current fragment and an annotation point-query
plan using that index. This is separate from a future status/label filter index.

A 2048-row workload also needs a table binding that admits 2048 rows. Set upload
(`ingress_memory_bytes`) and annotation apply (`annotations.apply_memory_bytes`)
budgets independently on the owner and both ingresses. The recorded experiments
use 32 MiB each; the largest decoded input reservation exceeds 16 MiB. Staging
and publication retain their separate bounds.

Choose staging byte/entry and per-value limits before provisioning: they affect
deterministic mutation rejection and are part of the durable schema fingerprint.
Changing them when reopening the same generation is rejected. The publication
aggregation timer and independent upload/apply/publication workspace budgets are
separate configuration; changing those does not change the mutation contract.

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

The writers run as independent Tokio tasks, and both are joined even if one
fails. Per-writer `client_phase_us` splits encoding, pending-journal sync, the
update request, ACK-journal sync and point reads. These are cumulative client
timers, not server CPU profiles, and do not measure internal copies or decode
work. Their sum across concurrent writers is not elapsed wall time.

Capture the owner's existing once-per-second annotation observations during the
load: applied/materialized positions, active/frozen entries, retained bytes,
source records/bytes, and backpressure. Sampled staging usage is not process RSS.
A final drain alone does not prove sustained drain; check that the materialized
cut advances during the load and that backlog stays bounded. Collect process
resource observations separately, and report the deployment's actual batch and
memory budgets. This probe changes status only; contract, concurrent field merge,
publication uncertainty and fencing are covered by the separate protocol tests.

## Single-shard acceptance evidence

Use these checks together; a successful status-only load does not establish all
mutation or failure semantics.

| Gate | Existing evidence |
| --- | --- |
| Field and retry contract | API annotation tests; `annotation_http_two_ingresses_keep_values_and_exact_retries_through_owner_replacement`. CAS remains outside this contract. |
| Bounded staging | Core staging tests for held readers, impossible batches and repeated drain; source-window tests for repeated updates to one turn. |
| Durable ordered updates | `quorum_ack_waits_for_apply_and_another_ingress_reads_without_a_tail_scan`; atomic repeated-key batch and concurrent field-update tests. |
| Immediate narrow point reads | The quorum/HTTP tests plus every successful load batch's opposite-ingress reads; core tests check narrow projection and duplicate IDs. |
| Safe publication | Core old-reader/concurrent-suffix tests and native publication tests; indexed verification additionally checks retained ID-index coverage. |
| Recovery, reclamation and pressure | Unknown-publication/checkpoint tests, `annotation_runtime_recovers_after_sigkill`, HTTP pressure tests and the deployed acknowledged-suffix recovery phase. |

The deployment report must include all six roles/counts relevant to the workload
(shards, replicas, owners, ingresses, writers and turns), configured limits,
refusals and p99 bounds, materialization progress during load, final values and
drain, and the exact failure phase. Keep failed runs with their recovery results;
successful later runs do not convert an earlier failure into a throughput pass.
Mixed ingestion, asynchronous filtering and shard expansion remain later gates.

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
recorded run used the original joined writer futures. The current driver spawns
independent writer tasks, so preserve that distinction in comparisons and use
separate client and service measurements before attributing a capacity ceiling
to the WAL or owner.
