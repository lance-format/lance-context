# Running one annotation shard

The ingestion binary provides `annotation-owner CONFIG` and
`annotation-ingress CONFIG`. They operate on an already provisioned managed
annotation generation whose narrow columns and mutation contract match the
configuration. Existing unbound Lance tables require an explicit handoff before
these roles can use them; normal startup does not adopt a table or overwrite its
bootstrap identity.

Use the existing TLS/WAL endpoints, authorized table binding, HTTP workload token,
receipt keys, and storage configuration. Add an `annotations` block:

```json
{
  "annotations": {
    "owner_url": "http://annotation-owner:8080",
    "apply_memory_bytes": 134217728,
    "staging": {
      "retained_bytes": 67108864,
      "retained_entries": 100000,
      "value_bytes": 65536,
      "readers": 128
    }
  }
}
```

This block is an addition to the existing process configuration, not a complete
standalone config. Exactly one table and one virtual shard are supported.
`ingress_memory_bytes`, `apply_memory_bytes`, and `materializer_memory_bytes` are
independent budgets. Publication needs capacity for the full bounded frozen
generation and its checkpoint metadata. `worker.publication` controls maximum
source records/bytes, minimum batch bytes, maximum batch wait, and retained
manifests. Ingestion compaction, session indexing and remote preparation options
are rejected for these annotation roles.

The owner requires the same authorized etcd prefix on every candidate. It claims
the table with the existing durable scope/owner/request tuple. Acquisition reads
the WAL epoch before validating the etcd lease, preventing a paused candidate
from using a successor's epoch. A busy candidate exits without acquiring an
epoch. Lease loss or a runtime/listener error ends the role; a process supervisor
may replace it. This does not automatically retry uncertain user mutations.

The owner address may name a stable Kubernetes Service. Every ingress forwards
to that address and opens neither Lance nor a local applier. Both roles install
the existing workload-token authentication middleware. The configured workload
token authorizes the table; an edge proxy remains responsible for TLS and any
narrower end-user permissions. `GET /api/v1/health` uses the same authentication.
Owner health checks the running runtime and catalog epoch; each point read still
waits for its apply barrier. Frontend health checks local draining state.

Graceful shutdown closes admission, lets admitted requests finish while the
applier runs, then joins the listener and runtime before releasing the claim.
A crash leaves the claim to expire. A replacement reconciles publication and
replays the required suffix through the existing runtime. Owner observations
report applied/materialized positions, retained staging bytes, active/frozen
entries, source bytes/records, and application pressure once per second.

The application process test uses real RF3 RPCs, etcd, Lance storage, two ingress
executables and an owner executable. It checks cross-ingress reads, publication
and source GC, exact retries, competing owner exclusion, SIGKILL plus natural
lease expiry, replacement behind the same address, field clearing, and lease-loss
shutdown. It also gives ingress processes an unusable Lance URI to verify that
forwarding does not open storage. It is a correctness test with a small fixture;
it does not qualify sustained throughput or large-table cold lookup performance.

Refs #378.
