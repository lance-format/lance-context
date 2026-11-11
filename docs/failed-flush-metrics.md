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

## Persistent failures, progress and alert routing

New gauges are evaluated on each `/metrics` scrape from resident writer state;
no storage or etcd requests are added. They have no table/shard/generation labels.

| Metric | Meaning / Datadog aggregation |
| --- | --- |
| `rollout_memtable_flush_health_complete` | 1 if all resident writer observations were readable. Group by Pod; missing or 0 is unknown. |
| `rollout_memtable_flush_health_sample_timestamp_seconds` | Scrape-time snapshot timestamp; use alongside collector health and expected Pod coverage. |
| `rollout_memtable_flush_active_generations` | Unfinished head flushes, including failures; sum across Pods. Not all queued generations. |
| `rollout_memtable_flush_terminal_writers` | Writers retaining fenced failures; sum across Pods. These cannot recover by retrying the same writer. |
| `rollout_memtable_flush_oldest_failure_age_seconds` | Max age since the first still-unresolved failure. Use max across Pods, with Pod breakdown. |
| `rollout_memtable_flush_checkpoint_idle_max_seconds` | Max time without advancing to a new durable checkpoint. Retries do not reset it. |
| `rollout_memtable_flush_unresolved_writer_drops_total` | Counter: a writer with failed retained batches was dropped. Investigate WAL replay; a vanished queue is not proof of recovery. |
| `rollout_memtable_flush_rows_total` | Counter of rows in successfully published generation flushes, including successful retry. |
| `rollout_wal_persisted_bytes_total` | Counter of bytes in successful WAL appends; not incoming request bytes or a count of published generations. |
| `rollout_memtable_flush_duration_seconds` | Successful attempt duration, excluding prior failed attempts/backoff. |

`GET /api/v1/internal/flush-health` returns live active/failed generations with
writer instance ID, table URI, shard, generation, checkpoint, age, first-failure
age, attempt count, terminal-fence status, and retained queue generations/bytes.
It uses the same internal-route access path as existing executor diagnostics.
No payloads or credentials are returned. Writer IDs are process-local: compare
only within the same Pod UID/container lifetime. `complete=false` means the
snapshot is partial; aggregate health gauges are omitted on that scrape.

Checkpoints are `started`, `wal_durable`, and `prepared` (immutable files ready
for manifest publication). Completion retires the active observation. Only a
**new** durable checkpoint advances progress; a retry repeating `started` or a
previous checkpoint does not. This is coarse telemetry: encoding/upload may
still progress within one stage. A large idle value triggers investigation,
**never automatic cancellation/restart**. Merge work already has finer byte,
row and commit progress via `/api/v1/merge-progress`; reuse that endpoint for
merge diagnosis rather than adding polling of etcd per table.

### Configure these monitors in the existing Datadog integration

Use the deployed OpenMetrics name mapping and production cluster/namespace
filters. Evaluate by Pod identity, not host. Keep gauges `.fill(null)`, turn on
no-data notification and verify expected Pod coverage. Do not mask one missing
Pod by summing the remaining Pods. Use rates/increases for counters, not their
lifetime absolute value. No extra scrape job is required.

1. **Retained flushes:** warn whenever retry generations >0; escalate if they
   persist for five minutes. This condition remains active even when the error
   counter stops growing or unrelated tables keep flushing.
2. **Terminal writer:** any terminal writer >0 triggers immediate investigation;
   don't wait for retries that deliberately cannot run after a fence.
3. **No new checkpoint:** active generations >0 and checkpoint idle >300s for
   two consecutive scrapes. Inspect the same writer/generation and actual I/O;
   do not automatically kill it based on the timer.
4. **Impact:** any increase in backpressure rejections or public read/write
   HTTP 503; monitor existing HTTP duration histograms for p95/p99 latency.
   Maintenance-route failures are a separate alert. Retained bytes measure
   failures, not all unflushed memory; zero rejections does not rule out writers
   waiting on backpressure or requests still in flight.
5. **Unverified recovery:** increase in unresolved writer drops. Zero retry
   gauges after a process replacement cannot clear this alert without replay
   or original-row evidence.
6. **Coverage:** missing/zero flush health completeness, stale collection, or
   expected workers absent; independently alert on unavailable/stale master
   backlog snapshots using existing `master_wal_backlog_snapshot_*` metrics.
7. **Backlog risk:** use existing largest-store pending, growth and actual
   reclaimed-generation rates. A coarse rolling scan is not an instantaneous
   arrival-rate counter; do not subtract snapshots as exact throughput.

Route state changes to the existing incident/agent trigger endpoint, deduplicate
by alert/Pod and incident, and send recovery after verified fresh recovery
observations. Normal agent review can run every 15 minutes using these existing
observations. Collectors and alerts continue at their normal cadence. Test one
synthetic alert through the real notification destination and verify agent
acknowledgement before calling alert-to-agent wakeup deployed. These instructions
do not themselves configure a Datadog monitor or notification destination.
