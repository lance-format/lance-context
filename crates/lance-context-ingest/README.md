# Lance Context ingestion

The ingestion application provides `annotation-bootstrap CONFIG`,
`annotation-owner CONFIG` and `annotation-ingress CONFIG` for managed annotation
and mixed body/annotation bindings. Start with the
[annotation deployment guide](../../docs/design/annotation-application.md) for
binding, bootstrap, ownership, routing and recovery requirements.

Configuration belongs to the running process. Owners consume their own
`worker.publication` policy, including optional `turn_id_index` maintenance;
ingresses forward requests and do not run that maintenance. HTTP admission and
memory budgets are also local to each process. See
[role configuration](../../docs/design/annotation-application.md#configure-each-running-role)
before copying a bootstrap configuration or comparing worker counts.

The owner's periodic annotation observation reports `backpressure_reason` as
`apply_memory`, `source_window`, `staging`, or `null`. It identifies the last
admission gate that prevented ordered apply; progress clears it. The existing
`backpressured` flag is true whenever a reason is present. These are sampled
states, not refusal counts or blocked durations. In particular, staging pressure
can include values held by readers or an old frozen generation after publication.

HTTP/2 annotation uploads that cannot immediately reserve memory receive
`memory_capacity` with `operation_outcome: not_submitted`. Waiting before reading
one stream's body can block other admitted uploads on the same connection.
The refusal occurs before decoding or appending to the WAL.

The owner's `append_rpc` observation separates WAL append byte admission,
connection acquisition (including pool waits and TLS), sending, and waiting for
the response. WAL `/status` reports `append_execution`, from complete body receipt
through worker execution and quorum completion; it excludes transport admission
and reply delivery. Execution includes validation and scheduling, not just disk I/O.
Each phase reports started/finished intervals, cumulative microseconds and the
largest finished interval. Errors and cancellations finish intervals too; use
the existing WAL outcome counters to distinguish outcomes. Observations are
independent samples, phases can overlap across requests, and neither means nor
maxima are p99 latency. An append redirected between nodes has multiple exchanges.

`publisher_binding_rpc` reports the same four phases for the compact
`PublisherBinding` ownership query. It includes ownership checks from writes,
reads and maintenance, not only point reads. Admission is counted once per
exchange; connection, send and response also include attempts made by the
existing read-only reconnect path. Other query kinds have separate behavior and
are excluded. Response time includes network, server scheduling and quorum work;
it is not a measurement of Raft execution alone.
