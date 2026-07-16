# Kubernetes deployment

`master.yaml` deploys one `lance-context-master` process with a dedicated
ReadWriteOnce PVC for its RocksDB scheduler state.

Before applying it:

1. Replace `DATA_DIR` with the shared object-store prefix used by rollout
   workers.
2. Set object-store credentials and `WORKER_ENDPOINTS` for your environment.
3. Replace the example image name and `UI_DIR` with paths from the published
   master image.
4. Set `storageClassName` on the PVC when the cluster has no default class.

Keep the master at one replica. RocksDB is an embedded single-process database;
the PVC must not be mounted by multiple active master pods. The `Recreate`
strategy ensures an old pod releases the volume before its replacement starts.
