# Replica selection for pinned source reads

Materialization previously tried the first configured voter for every payload.
Multiple preparers can therefore concentrate reads on one replica per source
group. The WAL serves those reads on its Raft owner thread, alongside replication
and control work. That is a potential contention point, not proof of saturation.

`worker.publication.source_read_policy` accepts:

- `first_voter` (default): preserve the existing configured voter order.
- `spread`: hash the immutable group/index pair to choose the first voter, then
  try each remaining configured voter once if the record is unavailable.

The small hash uses only two integers, without touching the payload. Hashing the
pair avoids the concentration that raw index modulo replica count can produce
when data and control entries interleave at a regular stride. This preference
is not persistent routing state; the hash mapping may change across releases.

The option applies to buffered and streamed preparation, including remote
preparers. Configure it on the processes doing the source reads. It changes no
plan, source pin, manifest authority, wire format, fetch concurrency or workspace
reservation. Responses must still match the exact pinned metadata and pass
payload, schema and row validation. A metadata mismatch remains an error rather
than being silently hidden by another replica.

The rollout recovery tests exercise both preparation modes with this policy:
reopen WAL directories after preparation, reuse durable descriptors, fence the
previous owner, resume after manifest publication, then verify complete columns,
exact retries and retirement across subsequent rounds. These tests establish
correctness under those scenarios, not a throughput improvement.

An additional network failure test leaves only one replica's payload endpoint
reachable at a time, exercising each of the three replicas as the sole data
source. The other payload endpoints accept and close connections, while real
quorum/control/peer endpoints remain available. Every case publishes and reads
back all columns, and the test checks that failed data routes were exercised.

Keep the default until a controlled comparison uses the same binary, workload,
replicas, storage and preparer count, varying only this option. Measure actual
Lance publication throughput, ACK/publication latency, backlog, retries and node
resources, and require complete payload readback after restart. Single-node
reader work, payload decoding and Lance encoding remain costs of either policy.
