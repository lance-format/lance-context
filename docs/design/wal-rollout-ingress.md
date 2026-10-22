# Durable rollout ingress

The independent `lance-context-ingest` workspace crate binds core's fixed rollout
schema to the WAL client. An application resolves and authorizes a `TableBinding`,
then supplies a shared memory budget and a configured mutual-TLS client. Connect
checks the entire durable table contract and manifest incarnation through a
quorum read. This is an adapter for newly bound immutable-ID rollout tables;
legacy upserts, existing table migration, and table delete/recreate are separate
protocols.

This integration crate is not published while the independent WAL remains
unpublished. The existing core crate keeps its current dependency and registry
release boundary. WAL replicas do not depend on core, Arrow decoding or Lance.

`rollout_id` is the session routing key. An atomic request must belong to one
virtual bucket, including when different buckets share a physical replica
group. The adapter rejects mixed buckets instead of splitting a previously
atomic request. Worker count does not affect placement. Row IDs must be globally
unique and immutable by the caller's contract; validation rejects duplicates
within one atomic request but does not implement a global uniqueness index.

## Write and retry contract

1. Persist the entire `ProducerOpen` tuple before opening a bounded producer
   slot. Reconcile the same tuple after an uncertain response. Do not allocate a
   different identity merely because an RPC times out.
2. Reserve ingress workspace before buffering the body. Prepare uncompressed
   Arrow IPC against the actual rollout schema, validating arrays, required
   fields, row limits, row IDs and session routing. Decoding and hashing run on
   the blocking executor. Consecutive rows from one session reuse its routing
   check. The original reference-counted body is retained without re-encoding.
3. Persist the producer, sequence and exact body in the caller's durable retry
   journal before append. Only a quorum-committed `Accepted` reply creates an
   `AcceptedAppend`. An unknown result retains the same sequence and bytes;
   advancing the sequence or changing the body is not reconciliation.
4. Persist the authenticated receipt before discarding the client retry body.
   A publication timeout does not cancel an ACK. `publication_status` uses a
   compact quorum view and the receipt's source-group checkpoint. A published
   response supplies the minimum Lance version needed by a reader.
5. Close with the confirmed final receipt. The WAL preserves exact retry of the
   last accepted tuple while fencing new writes. Slot reuse additionally needs
   durable source retirement; closing alone does not recycle it.

The preparation reservation follows the body through retries and cancellation.
The configured maximum request must fit the supplied shared budget. This is
logical admission, not an allocator or cgroup hard limit. Application clones of
exposed `Bytes` require their own lifetime accounting. TLS encryption and Arrow
alignment repair may copy; this is not an end-to-end zero-copy claim.

## Restartable receipts

`receipt::ReceiptKeys` issues authenticated, bounded tokens with no payload or
storage URI. Scope hashing includes the cluster, layout, table, destination,
incarnation and ingestion contract. Authentication precedes claim parsing, and
restoration checks all membership fields against the authorized binding. These
are trusted-service membership proofs, not user authentication or authorization.

Keys must come from persistent trusted configuration. During rotation, install
read keys everywhere before switching issuance and retain previous keys for as
long as old receipts must remain verifiable. Receipts have no automatic expiry.
Losing a key or deliberately revoking it never authorizes deleting acknowledged
data. A token cannot replace the durable producer/body journal when the ACK was
never received.

## Validation and integration boundary

Integration tests use three actual TLS-connected replicas, persistent original node
directories, and a real Lance genesis manifest. They exercise durable append,
exact retry, whole-cluster restart, full payload reads on every replica, receipt
restoration/rotation, mismatched bindings, bounded admission and producer close.
Preparation coverage checks same-group/different-bucket rejection, duplicate IDs
across IPC batches, row limits and original body allocation identity.

This crate does not yet wire the server HTTP route, a standalone node command,
automatic maintenance, or the full materialization/publication loop. The tests
must not be described as a sustained ingestion benchmark or as evidence that
newly appended rows have already become visible in Lance.
