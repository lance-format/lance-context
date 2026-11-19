# Publishing a registered table before recovery

`POST /api/v1/stores/{kind}/{name}/flush` asks any master replica to publish
worker-received data for one registered store. It returns HTTP 204 only after
every configured worker acknowledges publication. This is a synchronous barrier,
independent of the maintenance task queue, WAL merging, and compaction.

The initial interface supports the master's configured `MANAGED_STORE_KIND`.
The store must exist in its registry. Unsupported kinds and invalid names return
400; unknown stores return 404. Built-in rollout and generic stores are not
enabled by this interface until their workers implement the local barrier contract.

Application workers implement:

```text
POST /api/v1/{kind}/{name}/flush-local
```

The local endpoint waits for admitted appends and publishes that table's resident
writer. It returns 204 when the data is readable, or immediately if the worker has
no resident writer for the table. It must not forward the request to the master.
The caller must stop the previous run's writes before starting the barrier; this
does not drain client queues or recover dead/retired workers.

Set `WORKER_ENDPOINTS` on every master to the same complete list of direct worker
URLs. A load-balanced worker Service URL is insufficient: each worker may hold
unpublished data. Empty configuration fails with 503. Each request contacts all
workers, with up to eight requests in flight, and does not report partial success.
Redirects, HTTP statuses other than 204, and transport failures fail publication.
Other workers are still awaited after an individual failure, up to the overall
deadline.

`TABLE_FLUSH_TIMEOUT_SECS` defaults to 20 seconds and bounds registry lookup and
the entire fan-out. Configure the caller's timeout above it. A timeout returns
503; some workers may already have published. Publication must be safe to repeat,
so the caller can retry through another master replica. No leader election or
shared publication task is required.

Clients can continue using their existing application Service: its flush handler
forwards to the master without holding a local store lock, and the master calls
back through `flush-local`. Normal application reads and writes stay on workers.
The application must not substitute a local flush when the configured master is
unavailable. Update the worker list during controlled scaling; automatic worker
registration and discovery are outside this interface.

`master_table_flush_total{result="ok"|"failed"}` and
`master_table_flush_duration_seconds` measure master publication outcomes and
latency. Worker metrics remain necessary for diagnosing storage failures.
