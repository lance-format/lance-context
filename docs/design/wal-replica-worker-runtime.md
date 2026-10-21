# Dedicated replica workers

The optional `runtime` feature exposes asynchronous callers over the durable
`replication::Node`. Each Raft group has one dedicated owner thread. Groups on
the same physical node retain the existing shared log engine, synchronization
barrier and storage failure fence. Payload validation and blocking persistence
run on these threads, outside the caller's async executor.

`Handle::commit` accepts typed data and publication commands. Admission returns
a `Pending` future; it does not acknowledge durability. Only durable Node
completions resolve that future, including explicit unknown outcomes after
leadership loss. `Handle::query` resolves through the existing safe read-index
path. `campaign` only schedules an election and is not a leadership receipt.
`observation` returns a cached local view, whose timestamp is not a quorum lease.

## Ownership and progress

Data, control and incoming peer lanes have separate bounded queues. Request and
logical byte credits are shared across every group on the physical node. Query
admission includes a conservative allowance for its bounded response. Credits
cover queued work, in-flight work and replies waiting for their caller. Dropping
an unresolved future leaves ownership with its worker until completion or stop;
it cannot cancel an accepted Raft operation. After a future is consumed, its
result belongs to the caller, which must bound retained results itself.

Append scheduling retains the original shared `Bytes` without copying its body
or allocating another box for the batch. Larger control metadata is boxed to
keep queue elements small. Each iteration drains a bounded number of commands
from each lane before polling persistence. An outgoing packet owns its peer
credits through transmission. A full outbox drops packets for Raft retransmission
instead of blocking other group workers. Transport must retain the packet owner
until transmission completes, including cancelled sends.

An internal poll error or panic stops the shared runtime. Outstanding futures
resolve as uncertain errors, not successful acknowledgments. Callers must await
`Runtime::shutdown` before reopening the original log directories; dropping or
timing out shutdown cannot cancel filesystem operations already in progress.

## Integration boundary

This module does not open a network listener. The optional
[`peer-transport` feature](wal-peer-transport.md) provides authenticated TCP/TLS
connections for these workers. The embedding service still needs
authenticated routing, bounded framing before allocation, deadlines and retry
reconciliation, durable snapshot transfer, physical disk admission and automatic
maintenance. Snapshot packets are rejected until that transfer path is connected.
Logical frame budgets are not a measurement of allocator use: transport must
also account for oversized backing allocations retained by small byte slices,
decode buffers, application state and results retained after consuming futures.

## Validation

Runtime tests use real engine files and dedicated threads with an in-process
packet relay. They cover two groups on three replicas, exact retry indices and
full payload reads after reopening all engines, isolated leaders, cancellation
and shared credits, completed but unconsumed replies, independent control
admission, a saturated peer outbox, and a real file-sync error fencing sibling
workers. Existing replication process tests continue separately in the same CI
step. These checks do not establish network throughput, production readiness or
multi-process qualification of this new worker layer.

Tracking: #378.
