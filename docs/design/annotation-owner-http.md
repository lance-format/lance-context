# Single-shard annotation HTTP routing

Multiple HTTP ingresses forward annotation operations to one fenced runtime.
The runtime owns the bounded staging view, committed WAL applier, and continuous
Lance publisher. Forwarding ingresses do not open Lance or keep their own staging
views. Deployment configuration supplies the owner address, which may be a
stable Kubernetes Service address.

All routes are POST requests beneath `/api/v1/annotations/{table}`:

| Route | Request | Successful response |
| --- | --- | --- |
| `producer` | Existing `ProducerOpen` JSON | Signed producer credential |
| `update` | Arrow IPC file with existing producer/sequence headers | Signed receipt and persisted mutation result |
| `read` | `{"turn":"T"}` | Current narrow annotation or JSON null |
| `close` | Existing producer and optional last receipt JSON | HTTP 204 |

The owner checks the catalog authority and table incarnation before processing
and before returning success. An update returns HTTP 200 only after quorum
persistence and application to staging. A persisted mutation rejection returns
HTTP 422 with its receipt and outcome. Producer sequencing failures return 409.
After an ambiguous 503 or transport failure, the caller retains the original
credential, sequence, and bytes. The forwarding layer sends each operation once;
it does not invent a new mutation or automatically retry an uncertain outcome.

Reads wait for the owner's committed apply barrier and return annotation columns
only. They do not scan WAL payloads or read turn bodies. A successor recovers the
published checkpoint and the required WAL suffix before satisfying the read
barrier. An old owner is fenced even if its HTTP listener remains reachable.
Replacing the runtime behind the same owner address requires no frontend route
change.

Forwarders stream the incoming `Bytes` frames into the owner request. They do not
decode Arrow, re-encode it, or coalesce the complete upload. The owner still
coalesces and validates its upload, and the committed background applier decodes
the WAL record. This is not an end-to-end zero-copy claim. Data and control lanes
are bounded separately, request/reply memory is reserved, and response bytes hold
admission until released by HTTP. Both routers require the application's existing
authentication boundary; credentials alone do not authorize a table.

The RF3/Lance integration test uses two independent forwarding listeners and one
owner listener. It checks cross-ingress visibility, original Arrow payload
identity, control traffic during a partial upload, definite overload rejection,
original outcomes after publication and source GC, epoch fencing, replacement at
the same address, field clearing, and producer slot reuse. This is a correctness
test, not a throughput measurement. Application configuration, authenticated role
composition, and etcd scheduling integration are subsequent work.

Refs #378.
