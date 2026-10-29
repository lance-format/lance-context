# WAL client transport

`replication::runtime::client` connects trusted ingress and materializer services
to the public replica runtime. It uses mutual TLS, explicitly pinned service
certificates, a versioned ALPN and cluster/layout/policy/node/request correlation.
A trusted client identity permits access to the node's hosted groups. The HTTP
layer still must resolve and authorize each table; this transport does not accept
end-user credentials or infer permission from a table UUID.

Each physical node has separate data and control listeners and connection pools.
Every hosted group shares those pools. Appends and source reads use the data lane;
producer allocation, publication, quorum queries and leader hints use control.
Both lanes have bounded connections, logical byte credits and whole-exchange
deadlines. Metadata lengths are checked before variable-sized allocations.
The server connection allowance is independent of each client's pool size, so
several ingress processes can share the listeners without exhausting their
connection capacity merely by retaining healthy idle connections.
Fetched responses retain their byte credit until the response owner is dropped.
These are logical ownership bounds, not a hard allocator/RSS quota; callers must
budget retained clones and input allocations.

An append sends a bounded metadata frame followed by its original `Bytes` body.
There is no separate upload-grant round trip and no whole-body concatenation or
Arrow decoding in this layer. TLS encryption still copies encrypted buffers.
The server waits for the runtime's durable completion before reporting a commit;
queueing a command or sending bytes cannot produce an accepted ACK. A failed or
cancelled exchange discards its connection. Keep the original producer sequence
and payload for explicit reconciliation; errors do not prove non-commit.

A disk-budget refusal before `Raft::step` returns
`AppendNotSubmittedDiskBudget`. Only this attempt is known not to have been
submitted; a prior attempt with the same identity may already have committed.
Retain the original identity and bytes. The client returns the refusal without
automatically replaying it. Other execution errors, cancellation and lost replies
retain the unknown outcome.

This reply is appended to the existing RPC enum; existing messages and persisted
WAL records are unchanged. Upgrade WAL clients before replicas. An older client
cannot decode the new reply and conservatively sees a transport error/unknown
outcome; it must not treat that error as a successful append or definite rejection.

Routing follows only configured-voter hints returned before command submission.
A cached hint is never treated as an ACK. Unknown outcomes are returned to the
caller instead of automatically advancing/reissuing logical operations. The
source-fetch API returns local committed data; a quorum-confirmed source pin and
matching metadata are required before materialization. Missing local data does
not prove that an uncertain append was rejected.

With `lance-integration`, `Client::manifest_catalog()` implements the actual
`ManagedManifest` catalog interface over these RPCs. This supplies networked
quorum reads and durable manifest reservations. The complete HTTP/core ingestion
path, standalone node configuration and automatic maintenance remain follow-up
work; this module alone is not a production ingestion deployment.

Tests use four independent client pools, real TLS and the durable runtime. A
three-process test writes both groups, kills their leader, reconciles the exact
request, restarts all original directories and checks every payload on every
replica. Separate tests abandon a committed reply, exhaust data connections,
reject foreign identities and oversized frames, and verify no-quorum behavior.
A real Lance dataset is opened through the networked catalog before and after
all replicas restart. These checks establish correctness, not throughput.

Tracking: #378.
