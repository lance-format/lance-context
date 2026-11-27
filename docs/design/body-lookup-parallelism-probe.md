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
effect. No Lance OSS change is needed. Values outside `1..=32` or a missing native
pool are rejected. Key-workspace refusal remains `None`; native refusal is an
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

The native regression covers six settings over 32 real local Lance fragments,
including a missing key (requiring full coverage), ordinary default equivalence,
invalid options, missing native pool, key-budget refusal, native refusal,
deterministic cancellation during a gated read and an all-hit result before an
irrelevant gated fragment finishes. This validates the comparison entry; it is
not a production throughput experiment. Real captured-key results remain required
before changing owner defaults.
