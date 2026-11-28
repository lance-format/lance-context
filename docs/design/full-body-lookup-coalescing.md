# Bounded full-body lookup coalescing

`Runtime::with_body_lookup_batch_wait(wait)` optionally gives an underfilled
historical lookup window time to collect newly committed records. It defaults to
zero, applies only to full-body ingestion, and accepts at most one second. A
producer lane still waits for its applied outcome before its next request;
coalescing does not change request identity, acknowledgement or recovery.

This targets a timing gap: after an applied prefix releases producer lanes,
client acknowledgement persistence and the next quorum append may complete after
the driver's first refill. The ordinary foreground path and promotion of a
partially prefetched window both retain a fixed deadline. Cancelling and resuming
a step does not renew it. No apply/publication gate is held during the wait.

Only a short window whose validated source response reports the local applied
end waits. Recovery cuts, lagging cursors, filtered metadata before the local end,
request/byte bounds and memory refusal do not wait. Refills retain the existing
8-record, 32-MiB limits and all admission charges. The initial preparation retains
its existing I/O timeouts; additional refills stop at the coalescing deadline.
Low-volume input therefore progresses without waiting indefinitely for a full
batch. A full window incurs no intentional delay.

The extra wait is a latency tradeoff, not a throughput guarantee. Independent
producer lanes bound total outstanding records across foreground and ahead
windows; this option does not create additional producer concurrency. It also
does not reduce the cost of scanning history for one particular key set. Enable
only with a measured comparison of completed keys/scan, acknowledgement latency,
refusals and sustained backlog growth or drainage.

Existing prefetch observations count preparations, including repeated refills
and retained input. Use completed `lookup_requests / completed_lookups` for input
records per lookup and `(requested_keys - staged_keys) / scans_started` for
historical keys per scan, within the same owner/runtime interval. Preparation
counts are neither unique input counts nor a completed-query histogram.

Local RF3 tests cover a late committed tail from eight independent producers,
foreground cancellation and a fixed deadline under low traffic, and arrivals
after promotion's first read. Existing tests retain publication, restart,
SIGKILL recovery, source paging, pressure and first-create checks. These are
correctness tests with controlled scheduling, not a production benchmark.
