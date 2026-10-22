# Overlap replication messages and local persistence

The WAL runtime normally delivers peer packets after the entire synchronous
Raft poll finishes. An append therefore waits for the leader's local storage
before a follower can begin storing that append.

The experimental node setting `runtime.pipeline_replication: true` dispatches
ordinary, non-snapshot `Ready.messages` through the existing bounded outbound
queue before the local persistence call. Packets move their shared payload
buffers into that queue. Byte and request credits remain owned through transport;
an exhausted queue drops a packet for Raft to retransmit and never blocks the
group owner.

The default is false. This is a candidate for controlled comparison, not an
established throughput improvement. It does not make filesystem I/O asynchronous
or let the owner process timer ticks while a filesystem operation blocks.

`Ready.persisted_messages` still wait for durable HardState, entries and snapshot
installation. All current sync barriers, commit/application ordering, client
completion and quorum-read boundaries remain. Snapshots use the existing
captured-image ownership and reserved completion lane after the poll. The
synchronous public `Node::poll` retains its existing behavior.

With the option enabled, safe replication packets can already be in flight when
the leader's local persistence fails. The node still fences its shared engine
and completes outstanding callers with an uncertain outcome, never a successful
ACK from that failed poll. Another replica can retain the original proposal;
recovery must reconcile the same producer identity instead of assuming rejection.

Qualification must demonstrate a follower receiving an append while the leader's
real sync barrier is blocked, with no client ACK before release; injected leader
and follower sync failures; bounded outgoing credits; snapshot catch-up; and
complete payload/exact-retry recovery after reopening the original directories.
Independent-process recovery and a same-binary performance comparison remain
necessary before recommending this setting.
