# Distinguish wrapper lock waits from native WAL I/O

Each group's status now includes `durable_poll.storage_wait`: acquisition counts,
cumulative microseconds and lifetime maximum microseconds for the ordinary write
path's maintenance read gate, append read gate, sync mutex and exclusive append
gate. These locks sit outside raft-engine, so native write/sync timings omit
their waits. Without separate counters, a slow durable poll cannot distinguish
that waiting from native I/O.

Counters stay on the calling thread, use fixed storage, and are sampled into the
cached status only after successful polls. A failed poll retains the last good
snapshot. The non-runtime storage build does not collect timing. Lock order,
shared sync barriers, failure fencing and durable acknowledgment rules stay the
same; acquiring a lock is never a durable receipt.

Use before/after differences of counts and cumulative time on the same worker.
Maxima are lifetime maxima, not interval maxima or p99. These waits include local
checkpoint writes executed on that worker, but exclude group-state locks, native
I/O, other threads and maintenance's own exclusive gate acquisitions. They
overlap poll elapsed time and cannot be added to it or interpreted as per-request
latency. The counters themselves do not establish a performance improvement.

Tests persist and reopen a real record, check per-thread isolation and no-op
writes, and hold the real leader sync gate while a follower durably appends. The
client remains unacknowledged until the gate opens; the successful worker reports
the held interval, and sync failure retains the previous observation and still
releases no successful acknowledgment.

Refs #378.
