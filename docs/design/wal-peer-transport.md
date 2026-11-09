# Authenticated WAL peer transport

The optional `peer-transport` feature connects the durable replica workers over
TCP with mutual TLS. `replication::runtime::peer::Transport` owns one outgoing
connection per physical peer, shared by all hosted Raft groups. The embedding
service supplies a bound listener, ordinary verifying rustls configurations,
and a peer map covering exactly the other voters of its hosted groups.

Each peer has an endpoint, TLS server name and SHA-256 fingerprint of its DER
leaf certificate. TLS validates the certificate chain and server name; the
fingerprint binds that identity to a configured node ID. The receiver rejects
messages claiming a different sender, group, cluster, layout or policy before
passing them to Raft. Both directions require ALPN `lance-context-wal-peer/1`.
Certificate provisioning, rotation and endpoint discovery belong to the caller.

## Framing and ownership

Frames separate protobuf metadata from entry bodies. Outgoing `Bytes` bodies
share their original allocation and pass through the vectored frame writer
without concatenating the payload. The receiver validates bounded metadata,
reserves body credits before allocation or reading, fills uninitialized buffer
capacity, then splits shared body slices into the reconstructed entries.
TLS encryption/decryption and kernel I/O can still copy bytes; this is not an
end-to-end zero-copy claim.

Connection counts, receive bytes and each destination's queued message count and
bytes are bounded. Outgoing runtime credits remain owned until the transmission
completes or fails. Per-destination credits cover both queued and actively sent
messages and must leave runtime byte capacity for another voter. Disconnected
destinations discard queued traffic; Raft retransmits it. A slow follower cannot
retain the entire physical runtime outbox's byte allowance.

Within that same destination budget, queues with at least two slots reserve
one quarter of their slots (between one and four) for small Raft progress
messages. Only entry-free, snapshot-free heartbeats, append responses and
vote/pre-vote requests or responses with a charge of at most 1 KiB qualify.
All Appends, including empty Appends, retain their order in the data queue.
The sender serves at most four control messages before a ready data message.
This lets a small ACK enter a destination whose data queue is full without
increasing the total queue capacity or opening another connection.

Each reserved slot also reserves up to 1 KiB from the existing destination
byte allowance, capped so the configured maximum frame still fits. With no
spare bytes, both lanes share the existing cap without a byte reserve; a
one-slot queue keeps its original behavior. Runtime and destination credits
remain held through the actual send. This reservation cannot preempt a frame
already being transmitted or bypass upstream runtime admission, and does not
guarantee delivery when the control lane or shared byte allowance is full.

Handshake and complete-frame deadlines close uncertain connections; partially
written frames are never reused. Reconnects have a bounded delay. Receive-frame
credits remain held until runtime admission has acquired its own credits. These
are logical-byte bounds, not total process-memory measurements: decoded objects,
TLS buffers and allocator overhead also consume memory.

## Service lifecycle

Open the durable `Node`, spawn `runtime::Runtime`, and pass its handle and outbox
to `peer::Transport::spawn`. Submit work through the runtime handle. Transport
counters report network events only: sending a frame does not acknowledge a
durable write. Only the runtime's durable quorum completion resolves a commit.

Await transport shutdown before runtime shutdown, then await runtime shutdown
before reopening its log directories. Snapshot packets remain rejected. Durable
snapshot transfer, disk admission, automatic maintenance and public client/server
ingestion integration remain separate work; this module alone is not a complete
ingestion service.

## Validation

CI runs the feature's Clippy checks, library tests and existing replication
process suite. New tests cover shared outgoing payload allocations, rejection
before reading a body when credits are unavailable, certificate/node identity
mismatches, two groups across three TLS endpoints and durable reopening.

A slow-follower test stops consuming authenticated application traffic and
requires the other two replicas to continue committing while retaining runtime
headroom. A separate test starts three actual Rust child processes: the parent
sends test control requests but never relays consensus packets. It kills the
leader, continues on the surviving quorum, kills all nodes, reopens the original
logs, verifies exact retry indices and reads every payload from all three logs.
These tests establish the exercised correctness cases, not network throughput
or production readiness.

Queue tests fill the data lane before admitting a Raft ACK, keep control
traffic arriving while checking Append order and progress, and exhaust byte
credits while retaining an in-flight packet. They also check payload sharing,
closed-queue credit release, oversized control contexts and the single-slot
and maximum-frame cases. End-to-end throughput, tail latency, refusal rate
and materialization backlog still require a matched workload comparison.

Tracking: #378.
