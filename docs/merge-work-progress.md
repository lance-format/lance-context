# Actual merge progress

`GET /api/v1/merge-progress?target=<table>` reads the exact canonical task,
execution and work report from etcd. It performs no task inventory or table
payload scan. A live legacy task without execution telemetry returns
`progress_unknown`; a lease heartbeat is not classified as merge progress.
Older owned executors without detailed reports return
`progress_details_unavailable` alongside their existing progress sequence.

Resident maintenance reports executor process identity, canonical task ID,
execution ID, observed time, last progress time, monotonic idle duration and
configured idle allowance. Append merge work reports:

* WAL batches/rows read and decoded Arrow bytes (not network byte throughput).
* Encoded bytes, rows and files (not yet committed generations).
* Successful base commits and generations reclaimed after manifest drain.
* Active phase counts and the last completed stage. Several shard preparation
  futures may be reading, waiting for memory or encoding at once.

Phase transitions and reporting heartbeats do not advance the existing progress
sequence. A continued WAL read, encoding callback or acknowledged commit does.
The existing no-progress watchdog remains authoritative: continued work outlives
the idle allowance, while unchanged completed work triggers cancellation of
preparation followed by commit drain/storage fencing before replacement. Retry
backoff applies after a failure; it is not evidence of activity or a substitute
for progress detection.

Detailed reports and the established sequence are published in one
execution-qualified transaction. The sequence keeps its original wire format,
so mixed-version no-progress compare-and-swap remains valid. The API rejects a
sample if the execution changed or independently read sequence/report versions
disagree. Consumers must compare samples of the same execution and check the
report timestamp; `reported` is not a health verdict. Zero append counters on a
different maintenance kind are not evidence that its work is stalled.

The report is live execution state and is removed on normal release. A final
structured log records completed work when the work future ends, before commit
drain; that log alone does not certify safe writer termination. The existing
task completion record remains the terminal result.

This does not make legacy unfenced handlers cancellable, replay raw WAL or
change ownership admission. Such handlers require a separately proved stop or
storage fencing barrier before another publisher may take over.
