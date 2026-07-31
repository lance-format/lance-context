# Kubernetes deployment

`master.yaml` runs several stateless master replicas that share scheduler state
through a separate etcd cluster.

Before applying it:

1. Replace `DATA_DIR` with the shared object-store prefix used by rollout
   workers.
2. Set object-store credentials and `WORKER_ENDPOINTS` for your environment.
3. Replace the example image name and `UI_DIR` with paths from the published
   master image.
4. Replace `ETCD_ENDPOINTS` and create the referenced auth/TLS Secrets. Remove
   the auth variables or TLS volume/configuration when the application etcd
   cluster does not use them.

## Scheduler state

Scheduler state lives entirely in etcd, so the master is stateless and runs with
multiple replicas. Use a separate application etcd cluster, not the Kubernetes
control-plane etcd. Each etcd member owns its own persistent volume; master pods
have no PVC.

Task enqueue de-duplication, queued-to-running claims, and per-experiment
Compact/IndexId locks use etcd transactions. Claims and locks are attached to
renewed leases. If a master disappears, another replica requeues the task after
the lease expires. Execution is at-least-once, so task implementations must
remain idempotent across crash recovery.

The `_stats.rollout.lance` table remains in `DATA_DIR`. etcd coordinates its
single-writer sections, while readers reload the latest Lance manifest so every
master replica sees current stats.

## Automatic maintenance

Two periodic sweeps run on the master and feed the shared scheduler queue:

- **Compaction** (`COMPACTION_INTERVAL_SECS`, `MIN_FRAGMENTS`) rewrites an
  experiment's base-table fragments locally on the master. Large inline blobs
  make row decoding expensive, so master compaction first attempts Lance
  binary-copy and otherwise uses bounded input batches. Keep
  `COMPACTION_CONCURRENCY=1` and `COMPACTION_THREADS=1` unless the pod memory
  limit is sized for parallel rewrites. `COMPACTION_MAX_SOURCE_FRAGMENTS`
  makes large stores converge incrementally across sweeps, while
  `COMPACTION_MAX_BYTES_PER_FILE` prevents giant output files.
- **WAL merge** (`MERGE_WAL_INTERVAL_SECS`, `MERGE_WAL_MIN_GENERATIONS`) enqueues
  a `MergeWal` task for every experiment whose pending MemWAL generation count
  (from the periodically-scanned stats table) crosses the threshold. The task
  fans out to every `WORKER_ENDPOINTS` worker, each of which folds its own shard.
  The master cannot merge a shard it does not own without fencing the live
  writer, so the merge itself always runs on the owning worker. Set the interval
  to `0` to disable it; the manual "Merge WAL" / "Optimize" UI actions still work.
