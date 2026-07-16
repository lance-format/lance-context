# Kubernetes deployment

Two scheduler persistence modes are supported:

- `master.yaml`: one master with scheduler state in RocksDB on a dedicated
  ReadWriteOnce PVC.
- `master-etcd.yaml`: three stateless master replicas sharing scheduler state
  through a separate etcd cluster.

Before applying it:

1. Replace `DATA_DIR` with the shared object-store prefix used by rollout
   workers.
2. Set object-store credentials and `WORKER_ENDPOINTS` for your environment.
3. Replace the example image name and `UI_DIR` with paths from the published
   master image.
4. For RocksDB, set `storageClassName` on the PVC when the cluster has no
   default class.
5. For etcd, replace `ETCD_ENDPOINTS` and create the referenced auth/TLS
   Secrets. Remove the auth variables or TLS volume/configuration when the
   application etcd cluster does not use them.

Do not apply both manifests at once.

## RocksDB mode

Keep the master at one replica. RocksDB is an embedded single-process database,
and its PVC must not be mounted by multiple active master pods. The `Recreate`
strategy ensures an old pod releases the volume before its replacement starts.

## etcd mode

Use a separate application etcd cluster, not the Kubernetes control-plane
etcd. Each etcd member owns its own persistent volume; master pods have no PVC.

Task enqueue de-duplication, queued-to-running claims, and per-experiment
Compact/IndexId locks use etcd transactions. Claims and locks are attached to
renewed leases. If a master disappears, another replica requeues the task after
the lease expires. Execution is at-least-once, so task implementations must
remain idempotent across crash recovery.

The `_stats.rollout.lance` table remains in `DATA_DIR`. etcd coordinates its
single-writer sections, while readers reload the latest Lance manifest so every
master replica sees current stats.
