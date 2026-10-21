# Image and maintenance transitions in the replica runtime

The `runtime::Handle` now exposes the existing durable node's image lifecycle.
`prepare_checkpoint` captures an applied cut; run that task on an admitted
blocking worker, then pass its result to `install_checkpoint`. Newer log entries
remain after installation. Image building does not hold a Raft Ready or prevent
the owner from accepting ordinary append and peer work.

For an authenticated incoming snapshot, reserve transfer and disk capacity before
calling `receive_snapshot`. Feed sequential file chunks and finish validation on
the blocking worker, then call `step_snapshot` with the completed image. The
owner revalidates scope and hands it to Raft, which may ignore an obsolete cut.
The runtime releases the result only after a successful durable poll. An error
or panic leaves an uncertain outcome; a completed download or queued install is
not a durable acknowledgment.

Each outgoing snapshot captures its corresponding `Arc<Image>` immediately
after poll. The `Outbound` owner protects that image from collection through the
transfer, even when a newer checkpoint replaces the installed image. Consume it
with `Outbound::finish_snapshot` after the remote durable installation response.
Dropping an unfinished owner reports failure automatically, including rejection
by a full queue, cancellation and network disconnection. Successful network
transmission alone does not prove persistence.

Transfer feedback has a separate bounded lane and retains its original outgoing
credit until the worker polls. It cannot be denied by ordinary control traffic.
Feedback from an earlier transfer attempt, term or snapshot index cannot change
a newer pending transfer. Failure of this reserved lane fences the runtime.

`prepare_image_collection` returns the existing resumable, bounded directory scan.
One physical-node image permit excludes concurrent builders, receivers and
collectors across all sibling groups. `prepare_log_maintenance` returns a handle
for the shared physical engine: schedule one pass per node, not one per group.
Its native rewrite can delay persistence and needs reserved rewrite space.

Control credits cover queued transitions and unconsumed replies. The returned
image tasks additionally own the physical-node image permit and captured state,
whose size is governed by the replicated node policy, not the control-lane byte
allowance. After consuming a future, callers own its returned handles and must
budget, serialize and join the blocking work before shutdown/reopen. Dropping an
install future cannot revoke an already queued operation. Captured tasks and
images keep the native engine lock alive after runtime shutdown.

This is the runtime integration boundary. It does not schedule automatic
maintenance or impose a disk quota. The optional `peer::snapshot::Transport`
provides separate authenticated image streams and requires configured shared
node disk admission; attach it before enabling compaction. See
`wal-snapshot-transport.md` and `wal-node-disk-admission.md`.

Tests use real files and owner threads with a test-only image relay. They cover
two compacted groups catching up a lagging follower, every original engine
reopening with full payload and exact retry indices, sibling image exclusion,
cancelled installation, collection retaining a captured old image, physical
maintenance and native-lock ownership after shutdown. A partial transfer cannot
advance applied state; injected installation sync failure stops sibling workers
and releases neither a successful completion nor a peer response. Cancellation
with all sender control credits held still retries and catches up the follower,
and every original engine reopens with the complete payload. Existing TLS
process tests also run. Separate snapshot tests use actual TLS and files;
a separate three-process qualification kills a receiver mid-file and verifies
rejoin, leader loss and complete engine restart with exact payloads and retries.

Tracking: #378.
