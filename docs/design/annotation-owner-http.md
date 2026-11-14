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

Within that owner call, an uncertain WAL append is resolved by a fresh quorum
observation of the exact producer identity. If it is not yet accepted, the owner
may resubmit the original prepared append once, within the caller's original
deadline. The producer credential, sequence, metadata and Arrow bytes stay
unchanged; the payload is not prepared or encoded again. Absence is not proof of
failure: WAL deduplication returns the original position if the first proposal
commits late. Conflicting or unavailable observations do not permit resubmission.
If the second attempt fails, its outcome remains unknown unless a final quorum
observation confirms exact acceptance, even if that attempt was explicitly
refused. An accepted request still waits for its applied mutation result before
success. Apply timeouts do not resubmit the mutation, and successful appends gain
no additional RPC.

At acquisition, the runtime captures a quorum position after fencing its
predecessor. Owned reads wait for application through that recovery position,
then capture the current narrow staging/base view. This prevents a replacement
from returning an old Lance value before replaying a predecessor's successful
updates. All successful annotation writes in the epoch must await this runtime's
applier, through its HTTP routes or shared service handle. A raw WAL acceptance
is not an annotation write ACK.

Once recovery is applied, a later unacknowledged proposal does not hold up reads
of already applied values. For example, if a later write is waiting behind
staging pressure, the earlier successful annotation remains readable. Reads
project annotation columns only and keep the authority/incarnation checks before
processing and before returning success. An old owner is fenced even if its HTTP
listener remains reachable. Independently connected `Service` readers retain
their per-request quorum/apply barrier. Replacing the runtime behind the same
owner address requires no frontend route change.

Forwarders stream the incoming `Bytes` frames into the owner request. They do not
decode Arrow, re-encode it, or coalesce the complete upload. The owner still
coalesces and validates its upload, and the committed background applier decodes
the WAL record. This is not an end-to-end zero-copy claim. Data and control lanes
are bounded separately, request/reply memory is reserved, and response bytes hold
admission until released by HTTP. Both routers require the application's existing
authentication boundary; credentials alone do not authorize a table.

On the initial append attempt, a quorum-confirmed WAL capacity refusal returns
HTTP 503 with `error=wal_capacity`
and `operation_outcome=rejected`. The update was not accepted and its producer
sequence was not consumed. After capacity is reclaimed, the caller may submit
the same identity and bytes. Exact retained requests still return their original
result at capacity because WAL deduplication precedes the capacity decision.
An unresolved append or expired apply deadline remains an unknown outcome.
A refusal after an earlier uncertain attempt does not establish that the earlier
request was rejected. These distinctions do not eliminate waits for updates
already accepted while native publication is stalled.

Local WAL disk admission is a separate boundary. A proven refusal before Raft
proposal returns HTTP 503 with `error=wal_disk_capacity` and
`operation_outcome=not_submitted`. This describes this attempt only, including
when it is an exact retry of a previously committed request. Keep the same
identity and bytes after capacity becomes available; never replace an unresolved
request with a new sequence. The transport does not automatically retry it, and
the disk/control headroom limits remain unchanged.

The RF3/Lance integration test uses two independent forwarding listeners and one
owner listener. It checks cross-ingress visibility, original Arrow payload
identity, control traffic during a partial upload, definite overload rejection,
original outcomes after publication and source GC, epoch fencing, replacement at
the same address, field clearing, and producer slot reuse. This is a correctness
test, not a throughput measurement. Application configuration, authenticated role
composition, and etcd scheduling integration are described in
[Running one annotation shard](annotation-application.md).

A separate RF3 test retains an acknowledged predecessor update only in WAL,
starts the replacement listener before its applier, and verifies that reads wait
for recovery. It then pauses that runtime, commits a later proposal without an
annotation ACK, and checks the earlier value through the other ingress. Resuming
application returns the original proposal position and exposes its new value.

Refs #378.
