# Failed memtable flushes in Datadog

Pending WAL generations measure published generations waiting for merge. They
exclude failed memtable flushes. A low merge backlog does not establish writer
health or restoration of historical failures.

The following metrics come from the background generation flusher, including
idle retries, rather than successful/no-op sweeper calls. They have no table or
shard labels; use Kubernetes Pod tags for attribution and logs for the table.
These metrics require upgraded writers. Missing series on an old writer mean
unknown, never healthy.

| Chart | Exported metric | Aggregation |
| --- | --- | --- |
| Retained flush retry generations | `rollout_memtable_flush_retry_generations` | Sum across worker Pods; optional lines by Pod. Nonzero means a failed predecessor or generations queued behind it. |
| Resident bytes in flush retry queues | `rollout_memtable_flush_retry_bytes` | Sum across worker Pods, bytes. Keep failures counted while their retry is in flight. |
| Background flush failures | `rollout_memtable_flush_attempts_total{result="error"}` | Counter rate, summed across worker Pods. |
| Background flush completions | `rollout_memtable_flush_attempts_total{result="ok"}` | Counter rate, summed across worker Pods. Counts actual generation completions, not no-op sweeps. |
| Flush retry attempts | `rollout_memtable_flush_retries_total` | Counter rate, summed across worker Pods. Retries alone are not progress; compare completions and retry queue depth. |
| Writes rejected by memtable backpressure | `rollout_memtable_flush_backpressure_rejections_total` | Counter rate, summed across worker Pods. Includes memtable backpressure errors, not every HTTP 500/503. |

Use the metric names as mapped by the installed OpenMetrics integration (its
namespace and counter suffix may differ from the exporter names above). Apply
`falcon-phx-ca` / `rollout-db` filters matching the existing dashboard. Do not
use a fleet `max` for additive counts or bytes. Do not fill missing samples with
zero. Counters need rate/count semantics, not raw cumulative values.

Put retry generations, retry bytes, and write rejections beside merge backlog.
Flag any retry queue or backpressure rejection for investigation; persistent
queue depth with retries and no completions identifies failed recovery.

The gauges describe live retained retry queues. They are not a persistent
historical recovery ledger, and dropping a writer/process is not evidence that
its data was recovered. Historical failures require separate batch-level
commit and readback verification. Successful post-flush grace buffers are not
included. Queue bytes can include indexes/shared Arrow allocations and are not
an exact process RSS measurement.
