# Bound work before durable Raft progress

The runtime polls Raft after draining a batch of control, peer and data work.
Without a time bound, a proposal can wait behind subsequent expensive work
before its replication messages are emitted, or before its completion is
delivered. A command-count bound alone does not distinguish small metadata
messages from large payload validation and local reads.

The worker first drains a bounded prefix of control requests so queued reads
can share one fresh quorum barrier before expensive data work. The prefix ends
at `batch_requests`, an empty control queue, or `batch_time` (1 ms by default),
whichever comes first. The worker then visits peer and data work at least once,
even if the control prefix consumed its budget. This preserves write and replica
progress under continuous control traffic. Further rounds visit snapshot results,
peers and data until the same time or per-lane count bound is reached; they do not
drain additional control requests before this poll.

Service JSON represents `batch_time` as milliseconds, from 1 to 1000. The budget
is checked after each control operation and between peer/data rounds: it cannot
interrupt one blocking operation, bound disk latency, or guarantee a request
deadline. There is no new timer task, sleep or storage format. Queries admitted
after a barrier was sent still require their own fresh quorum confirmation.

Polling retains all existing persistence, quorum, apply and failure barriers.
Credits remain owned through the poll, and client completions follow the
successful poll exactly as before. More frequent polling can increase native
write/sync or transport activity; a mixed-load comparison must assess throughput,
refusals, tail latency and sustained materialization before recommending it.

The regression uses actual RF3 workers and engines. A controlled slow round
queues an append ahead of the next blocked control request. Replica traffic
must be emitted before that next request is released, while the append stays
unacknowledged until its actual quorum/apply completion. The test reopens every
original engine and checks full payload bytes. The old count-only scheduler
fails the intended progress assertion after cleaning up all original workers.

A second regression queues 16 reads ahead of controlled slow data work. It
delivers only the first quorum heartbeat response from an actual follower and
checks how many reads complete. The interleaved scheduler completes one; the
bounded control prefix completes all 16. A later query remains unacknowledged
without another quorum. All original queries and a new append finish, and all
three original engines reopen with their full payloads before comparing the
result. This is a scheduling experiment, not evidence of network or end-to-end
Lance throughput improvement.
