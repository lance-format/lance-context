# Compare one captured historical lookup

`TurnIngestionSpec::lookup_existing_with_parallelism` runs the existing read-only
membership lookup with independently selected task partitions and fragment
readahead. It requires the already opened Dataset's shared native memory pool;
it never substitutes a private pool. The ordinary lookup and ingestion runtime
retain their existing defaults. This method neither opens tables nor creates an
owner, reader service, worker, scheduler or persistent configuration.

Use the existing authorized managed-manifest reader to open the capture's exact
version. Bind the capture's table incarnation, source group/bucket and wire
contract against the existing host binding, and verify its complete ordered
fragment ID list. The captured schema hash is the wire contract, not simply the
stored Arrow schema. Do not silently use the latest version if the selected
version/files were retired. An Arc<Dataset> by itself is not a GC lease.

After those checks, reuse the existing spec, exact unresolved canonical keys,
Dataset and separately admitted key workspace:

```rust,ignore
use lance_context_ingest::turn_ingestion::TurnLookupParallelism;

let baseline = spec.lookup_existing(base.clone(), &keys, &key_workspace)
    .await?.ok_or_else(|| anyhow::anyhow!("key workspace refused"))?;
let expected: Vec<_> = keys.iter().map(|key| baseline.get(key)).collect();
drop(baseline); // Release key workspace before the next lookup.

// Rotate/reverse this order between runs; a warmed cache is not a cold-cache run.
for (task_partitions, fragment_readahead) in
    [(8, 8), (16, 8), (32, 8), (8, 16), (8, 32)]
{
    let result = spec.lookup_existing_with_parallelism(
        base.clone(), &keys, &key_workspace,
        TurnLookupParallelism { task_partitions, fragment_readahead },
    ).await?.ok_or_else(|| anyhow::anyhow!("key workspace refused"))?;
    anyhow::ensure!(result.matches(&base), "snapshot changed");
    for (key, expected) in keys.iter().zip(&expected) {
        anyhow::ensure!(result.get(key) == *expected, "membership changed");
    }
    // Record result.observation() plus the existing host pool's peak/backed/
    // refusal counters in protected diagnostics. Do not print the keys or plan.
    drop(result);
}
```

This reuses production key admission, schema checks, escaped column predicates,
exact composite membership, all-hit early exit and native observation. It changes
only eligible plain unordered FilteredReadExec nodes. Indexed inputs, row-stream
inputs and scans with before/after-filter ranges retain their original options;
inspect actual `scan_partitions` rather than assuming a requested setting took
effect. No Lance OSS change is needed. Zero values or a missing native
pool are rejected. The initial 8/16/32 measurement matrix is not an API ceiling;
actual read/decode allocations still require native pool admission.
Key-workspace refusal remains `None`; native refusal is an
error and never a successful miss. There is no added retry.

The native task partitions consume one shared task stream, so P partitions are
not P copies of the scan. Each consumer awaits its own read/decode task. Increasing
P also increases concurrent decode demand; preserving
`batch_readahead(8)` in the initial Scanner does not cap that demand after the
native threading mode is replaced. F bounds owned fragment opener/task-producer
streams, not all issued I/O and decoder lifetimes. Both modes retain the existing
2048-row, 1 MiB batch and 8 MiB I/O targets, the same native scheduler and the same
shared pool. These targets are not an RSS bound.

First vary P with F=8, then vary F with P=8. Compare exact membership, first-result and
complete lookup latency, scan rows/fragments/ranges/logical bytes, original host
pool demand/refusals and backlog/ACK behavior in any later live experiment.
Metrics snapshot work at lookup completion; early-exit cancellation can leave
already-issued I/O completing afterward. Do not conflate logical Lance reads with
cloud network bytes or cumulative task wait with CPU time. Allow resources to
finish releasing before interpreting pool counters. Record concurrent host
activity: global shared-pool deltas are not exclusive costs of this query.

The native regression covers six settings (including P/F=128/128) over 32 real local Lance fragments,
including a missing key (requiring full coverage), ordinary default equivalence,
invalid options, missing native pool, key-budget refusal, native refusal,
deterministic cancellation during a gated read and an all-hit result before an
irrelevant gated fragment finishes. This validates the comparison entry; it is
not a production throughput experiment. Real captured-key results remain required
before changing owner defaults.

## Runnable fixed-input example

Build the example in a separate experimental environment, outside the live
ingestion build and target directory:

```sh
cargo build -p lance-context-ingest --features service --example body_lookup_probe
```

This command uses the dev profile. Record the actual build profile, CPU
architecture, machine limits and storage/network location alongside results.
A debug build or another machine is not a production-pod performance measurement.

The first command saves the exact authenticated quorum-selected catalog record,
before directory retention can retire the captured version:

```sh
body_lookup_probe record CONFIG.json CAPTURE.json NEW_RECORD.json
```

`CONFIG.json` is a private deployment-authorized JSON object with these fields:

| Field | Existing type / meaning |
| --- | --- |
| `wal` | `application::config::WalConnection`: original layout, replication limits, endpoints, TLS file paths, RPC options |
| `binding` | Original `TableBinding` for this full-body table |
| `shard` | Original `{ "bucket": ... }` |
| `staging` | Original annotation staging `Limits`, needed for the wire contract |
| `spec` | Original serialized `TurnIngestionSpec` |
| `uri` | Authorized storage URI for this binding |
| `storage_options` | Existing storage options (optional, defaults empty) |

Use the actual host values, including staging limits. The full-body wire hash
differs from both the ordinary dataset schema and the mixed body/narrow-table
wire hash. Do not guess it. `CAPTURE.json` is the complete existing serialized
lookup capture, without an extra envelope. Inputs must be regular files, private
on Unix (0600 or stricter). JSON is decoded from a buffered file reader without
a duplicate full-input byte vector or an example-imposed file-size ceiling.
Decoded keys/metadata remain outside native pool accounting. The existing
production key contract still applies. No keys or credential values appear in
stdout/report.

`record` validates the capture against the configured binding, key specification
and shard, then checks the publisher binding through an authenticated quorum read
and reads `Some(capture.base_version)` from the existing manifest catalog. It
never selects latest, acquires an owner, starts an applier or changes retention.
It fails if selection is unavailable, unknown or retired. After successful
selection, a private temporary file is serialized and synced, then atomically
persisted without overwriting an existing destination. Failed selection leaves
no empty destination. Errors emit static stage/category labels, including
manifest-retired/absent/unknown versus RPC failure; raw backend error strings
are never printed. Preserve the authenticated
provenance of this artifact: deserializing a supplied record is not authentication.

The record can be used after catalog-directory retirement while the referenced
manifest/data/index files still exist. `PinnedManifest` is not a file-retention
lease. The default directory retention is 64 versions, not a time TTL; verify the
actual host setting separately. Do not change compaction or retention for this
experiment.

Then run the comparison using the saved record, with explicit experimental
native/key memory limits (MiB), rounds and per-query deadline (seconds):

```sh
body_lookup_probe run CONFIG.json CAPTURE.json RECORD.json NEW_REPORT.jsonl 256 64 3 120
```

These numeric limits are caller inputs, not recommendations for production.
Budgets, rounds and deadline must be nonzero and representable, with checked
MiB-to-byte conversion; the example adds no numeric ceiling. Round order is
generated lazily. Tokio uses its normal available-parallelism default or the
caller's `TOKIO_WORKER_THREADS`; the report records the actual worker count.
The
standalone CLI uses its own bounded pool and one session; it does not share the
live host pool. It is not a production ACK or backlog-drain benchmark. Native
allocation capacity, JSON/plan/cache overhead and RSS are different quantities.
To embed the example, use `probe::open` with the original caller-owned `Storage`
and `probe::compare` with the original key workspace and pool observation closure;
these functions preserve the session, accessors and store wrappers.

The standalone resolver supports the existing Lance storage options/environment;
it does not recreate a host-specific renewable catalog credential accessor. If
the deployed storage uses one, run the embedded functions with that original
`Storage` or supply the equivalent existing authorized provider before opening.
Do not replace renewal with copied short-lived secrets just to run the example.

The probe checks table/incarnation/wire hash/key routing, the exact version and
the complete ordered fragment inventory. It runs ordinary lookup as its result
reference, followed by P/F `(8,8), (16,8), (32,8), (8,16), (8,32)` in rotated and
reversed order over subsequent rounds. Every lookup must return the same complete
membership vector. Results release key admission before the next query. A native
or key-workspace refusal, timeout, read error or mismatch stops the comparison and
is never reported as a successful miss. There is no added lookup retry.

The private JSONL report includes first-result/completion time, actual partition
count, scan rows/fragments/ranges, native logical reads, store read counter deltas
and pool reserved/backed/demand/refusal counters. Native peaks are cumulative over
this standalone session. Cache state persists, so these are not cold-cache runs.
Outstanding I/O may finish after early exit: counters sampled at completion are
not exclusive costs of one query. Range overlap is explicitly `not_instrumented`;
store averages cannot establish duplicate cloud GETs. Physical network bytes and
online ingestion throughput are not measured by this example.

Example tests use a real four-fragment local Lance snapshot: all 16 lookups in
three rounds agree with independently expected two hits and one miss. They also
exercise wrong fragment order, foreign wire/binding/version, incomplete capture,
native refusal, reservation release, private inputs and atomic exclusive outputs.
This verifies the runner, not performance on production data.
