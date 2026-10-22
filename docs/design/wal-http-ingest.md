# HTTP rollout ingestion and recoverable Rust producers

The optional `lance-context-ingest/http` feature supplies an Axum gateway over
the public TLS WAL client. `http-client` adds a Rust producer with a persistent,
shared group-commit journal. Both retain the same rollout schema, session bucket,
producer generation and exact-retry contract as `Ingress`. No Lance OSS changes
are required.

The application resolves authorized, immutable table bindings before constructing
`Gateway`, and installs its router behind the application's access-control layer.
The configured map is a fixed set of qualified table names, not a namespace
registry or a mechanism for migrating/rebinding live legacy tables. Multiple
gateways can serve the same binding. They share persistent receipt/producer
signing keys, and each process shares physical WAL client pools and a memory
budget across its ingresses.

## Requests

All request bodies require a single valid Content-Length and Content-Type.
Control bodies are JSON bounded to 32 KiB. Arrow append bodies use
`application/vnd.apache.arrow.file` and are bounded by the registered contract.
Chunked uploads, conflicting headers, trailers and mismatched lengths fail
before an append is submitted. Proxy request/header/connection limits remain an
application deployment responsibility.

- `POST /api/v1/ingest/{table}/producer` accepts the persisted `ProducerOpen`
  tuple. It returns a generation, scope fingerprint and authenticated producer
  credential after durable allocation. Retry an uncertain open with the same
  slot, expected generation, operation UUID and bucket.
- `POST /api/v1/ingest/{table}/append` carries the original Arrow IPC body, an
  `x-lance-producer` credential and an `x-lance-sequence` header. A 202 response
  carries a durable receipt, original WAL position, sequence, row count and body
  digest. It does not wait for Lance publication.
- `POST /api/v1/ingest/{table}/publication` accepts a receipt and returns an
  optional minimum visible Lance version from a compact quorum query.
- `POST /api/v1/ingest/{table}/close` accepts a producer credential and its last
  receipt, if any. The caller retains the original close state on uncertainty.

Producer and receipt credentials have distinct authenticated domains and include
the complete binding fingerprint. They cannot be interchanged. A producer
credential is portable across gateways; append does not require a gateway-local
producer cache or another allocation round trip. The WAL still fences stale
producer generations. These credentials do not authorize end-user access.

## Resource and cancellation behavior

Data and control handlers have independent bounded admission. A saturated data
lane cannot consume the control slots needed for producer/visibility operations.
Append reserves shared workspace before reading its body. Reads have idle and
total deadlines, and memory admission has a bounded wait. A single HTTP frame
retains its Bytes allocation; fragmented bodies are coalesced once. Validation,
digest computation and WAL transport then share the same immutable body.

Admitted handlers run independently of the HTTP socket and retain their guards
through completion. `Gateway::shutdown` closes admission and joins those
handlers, including disconnected callers. It does not abort a submitted write.
The application also stops/drains its HTTP listener. The gateway reports local
in-flight counts, which are not a quorum or publication-health assertion.

## Durable client recovery

`http::journal::DurableClient` opens one bounded local journal for an ordered set
of producer lanes. Those lanes share the journal's group commit, HTTP pool and
memory reservation. Each lane is serialized; different lanes can make progress
concurrently. The parent journal directory must exist on persistent storage.
The client does not make this journal replicated or reassign ownership of its
volume to another process.

Persist producer-open tuples in application configuration. Reopening requires
the same layout, complete table binding, ordered producer-open tuples and journal
settings. Changing the gateway endpoint is allowed. The supplied HTTP client
must target trusted gateways with redirects disabled.

`stage` journals the exact body and sequence before any append. `append` combines
staging and transmission. `resume` resends an existing staged/recovered intent;
it never allocates a new sequence after an uncertain response. The client
validates the returned scope/generation and ACK sequence/group/digest before
journaling the ACK, releasing the pending body, or advancing its sequence.
`last_accepted` exposes the durable latest ACK after a restart; calling `append`
again after a known ACK is a new application batch, not a request-ID retry.

A network failure leaves the original intent available. A failed or cancelled
local journal operation fences that lane until the journal is reopened, because
the disk update may have completed without an observed reply. A different body
cannot replace an unresolved intent. Explicit shutdown joins the disk worker off
the async executor. Recovery parsing and payload verification also run off the
async executor. This is bounded logical accounting, not zero-copy disk encoding
or an allocator/cgroup hard quota.

## Qualification and remaining integration

The new qualification connects real HTTP listeners to real TLS replicas and a
real rollout Lance dataset. It injects an intermediary error after a durable
append, restarts the client journal, gateway and original replica directories,
checks the exact original ACK, restarts again with prepared Lance files, then
publishes and verifies complete rows and subsequent journaled sequences. A
second test saturates the data lane with an incomplete body, exercises control
requests, and checks that draining joins the body reader without consuming a
producer sequence. Both integration checks passed with the ingress/materializer
tests, HTTP feature Clippy checks and client-journal recovery regressions. The
application restart checks use joined shutdown/reopen; they are not process
SIGKILL or sustained throughput measurements.

This supplies application library interfaces, not a complete deployable gateway
or materializer worker. Bootstrap/namespace integration, distributed worker
scheduling and discovery, application readiness and shutdown orchestration, and
sustained Kubernetes/blob performance qualification remain integration work.

Tracking: #378.
