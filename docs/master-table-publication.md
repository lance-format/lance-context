# Publishing a registered table before recovery

`POST /api/v1/stores/{kind}/{name}/flush` asks any master replica to publish
worker-received data for one registered store. It returns HTTP 204 only after
every selected worker acknowledges publication. This is a synchronous barrier,
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

For a fixed deployment, set `WORKER_ENDPOINTS` on every master to the same complete
list of direct worker URLs. On Kubernetes, use Service discovery as described below
so scaling does not require editing that list. A load-balanced worker Service URL
alone is insufficient: each worker may hold unpublished data. Empty membership
fails with 503. Each request contacts all
workers, with up to eight requests in flight, and does not report partial success.
Each worker must return HTTP 204. Connection/request transport errors and HTTP
408, 429, 500, 502, 503 and 504 receive up to three additional attempts for that
worker. Successful workers are not called again. Retries use exponential backoff
with jitter: 250–500 ms, 500–1000 ms, then 1–2 seconds. A worker's `Retry-After`
header (seconds or HTTP date) sets a minimum wait. Redirects, other HTTP statuses,
and exhausted retries fail publication.
Other workers are still awaited after an individual failure, up to the overall
deadline.

`TABLE_FLUSH_TIMEOUT_SECS` defaults to 20 seconds and bounds registry lookup and
discovery and the entire fan-out. Configure the caller's timeout above it. A timeout returns
503; some workers may already have published. Publication must be safe to repeat,
so the caller can retry through another master replica. No leader election or
shared publication task is required.

All attempts, retry waits and the final membership check share this deadline;
retries do not reset it. A slow attempt can consume the remaining time without
leaving room for another attempt. Changed membership still fails the whole barrier
so the caller can start again with a fresh worker list.

Clients can continue using their existing application Service: its flush handler
forwards to the master without holding a local store lock, and the master calls
back through `flush-local`. Normal application reads and writes stay on workers.
The application must not substitute a local flush when the configured master is
unavailable.

`master_table_flush_total{result="ok"|"failed"}` and
`master_table_flush_duration_seconds` measure master publication outcomes and
latency. Worker metrics remain necessary for diagnosing storage failures.
`master_table_flush_worker_retries_total` counts additional worker publication
attempts actually started; waiting for a retry that is cancelled by the deadline
does not increment it.

## Kubernetes Service discovery

For application workers behind a Kubernetes Service, configure the master with:

```text
MANAGED_STORE_KIND=events
MANAGED_WORKER_SERVICE=event-workers
MANAGED_WORKER_NAMESPACE=events
MANAGED_WORKER_PORT_NAME=http
```

The namespace defaults to the master's mounted service-account namespace; the
port name defaults to `http`. Discovery overrides `WORKER_ENDPOINTS` only for the
managed store kind. It also supplies that kind's background merge fan-out and
merge admission. Other store kinds retain their existing routing.

Each publication request lists the Service's EndpointSlices through the Kubernetes
API, using Most Recent read semantics and following every page. It contacts the
individual IP addresses and then lists membership again. Added, removed or
replaced Pod identities make the barrier return 503; retry starts a new complete
barrier. Changes to unrelated Kubernetes objects do not fail the request.
There is no periodically refreshed local membership cache and no direct access
to Kubernetes' control-plane etcd.

The master includes NotReady and terminating endpoints. Those workers may still
hold acknowledged writes; excluding them could falsely report publication success.
Each endpoint must include a non-empty `targetRef.uid` identifying its Pod.
Missing, null, empty or whitespace-only UIDs fail discovery with 503; address-only
endpoints cannot distinguish a replacement Pod at the same address. Use Pod-backed
Services whose EndpointSlices provide these identities. The requirement applies
to both membership snapshots and the managed kind's background merge discovery.
An unavailable endpoint, API error, missing port or empty list also fails the request.
This favors recovery correctness over availability during worker replacement.

The master needs its projected service-account token, cluster CA and access to
the Kubernetes API. The following example grants only namespace-scoped LIST of
EndpointSlices. Kubernetes RBAC cannot limit LIST to a Service label, so this
role can list all EndpointSlices in the namespace; the request filters by the
configured Service label.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: worker-discovery
  namespace: events
rules:
  - apiGroups: [discovery.k8s.io]
    resources: [endpointslices]
    verbs: [list]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: worker-discovery
  namespace: events
subjects:
  - kind: ServiceAccount
    name: event-master
    namespace: events
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: worker-discovery
```

Mount the standard service-account credentials on the master (for example with
`automountServiceAccountToken: true`). The master rereads the token for each API
request so projected-token rotation does not require restarting it.

### Recovery and scaling boundaries

Stop the previous run's writes before starting recovery. Writers must reach
workers through this Service: a Pod admitted to its routing is represented by an
EndpointSlice. A fresh API snapshot therefore avoids the independent, stale worker
list that could previously omit a newly scaled writer.

Discovery is not a historical writer registry. A Pod removed before the first
snapshot is no longer discoverable. Graceful scale-down must stop new appends,
drain and publish the retiring worker before its endpoint disappears. Abrupt
worker loss, direct-to-Pod writes outside this Service, and concurrent old-run
writes require additional recovery or fencing; this barrier does not provide it.
