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

As a safety net, each master writes a heartbeat lease
(`_master.singleton.json`) into `DATA_DIR` at startup and refreshes it every few
seconds. A second master pointed at the same `DATA_DIR` refuses to start while a
live lease exists, and logs which host holds it. A crashed master's lease goes
stale after ~30s so a replacement can take over automatically; a graceful
shutdown releases the lease immediately. This guards against the common mistake
of leaving an old master running while starting a new one — even when the two
use separate PVCs but share one `DATA_DIR`.
