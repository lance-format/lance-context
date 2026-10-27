# Etcd operation accounting

The master and worker expose two monotonic counters for instrumented etcd client calls:

- `etcd_operation_attempts_total{operation}` increments when the client future is first polled.
- `etcd_operation_completions_total{operation,outcome}` records `ok`, `error`, or `abandoned` exactly once for each polled call that returns or is dropped.

`ok` includes a successfully received Txn response whose comparisons rejected the mutation. `abandoned` means the local future was dropped, including an enclosing timeout or cancellation; the remote outcome remains unknown. Neither outcome changes ownership or grants permission to repeat a mutation. Calls still in flight contribute to attempts but not completions. A future dropped before its first poll contributes to neither counter. Each explicit retry is a separate attempt.

Labels come from the finite `lance_context_metrics::etcd::Operation` enum. No key, table, endpoint, execution ID, error string, or other unbounded value is exported. Metric keys and labels are static. The wrapper does not box or spawn futures; recorder registration can still allocate counter handles. There are no per-operation histograms. The current 31 operations allow at most 124 counter series per process before infrastructure tags; only observed series are registered.

## Covered paths

| Operation labels | Client calls |
| --- | --- |
| `merge_reserve`, `merge_start`, `merge_terminal`, `merge_release` | Execution lifecycle CAS, including terminal cancellation acknowledgements |
| `merge_progress_initial`, `merge_progress_changed` | Initial sequence-zero and subsequent simple progress publications, including rejected publications |
| `merge_work_progress` | Detailed work-report publication transactions |
| `merge_commit_read`, `merge_commit_write` | Commit permit observations and authorization transactions |
| `merge_read`, `merge_scan`, `merge_failure_read` | Execution/progress snapshots, execution inventory, failure-ledger reads |
| `merge_owner_watch`, `merge_progress_watch` | Opening ownership/progress watch streams |
| `merge_other` | Other coordinator transactions: recovery, legacy retirement, background admission, mode and failure bookkeeping, request acknowledgement |
| `demand_read`, `demand_publish`, `demand_scan` | Demand-event joins, durable merge requests and request inventory |
| `task_read`, `task_read_txn`, `task_scan` | TaskStore point reads, batched admission/dependency snapshots and paginated inventories |
| `task_enqueue`, `task_claim`, `task_finish`, `task_other` | TaskStore admission, claim/promotion, completion and remaining bookkeeping calls |
| `dispatch_read`, `dispatch_write`, `dispatch_watch` | Informer snapshots and leader observations, dispatch offers/tickets, and opening dispatch/maintenance/preparation watch streams |
| `lease_grant`, `lease_stream_open`, `lease_revoke` | TaskStore lease management, including reconnect attempts |

These are operation groups, not gRPC method labels. For example, `merge_read` includes a conditional read-only Txn as well as Range calls, and `task_other` includes several bookkeeping methods. Nested Txn operations are not counted as separate calls.

A watch-opening call is counted once. Adding subscriptions on that existing stream, receiving watch events, or sending lease keepalive messages is not another watch-opening RPC. Connection establishment, transport-internal retries and uninstrumented callers are outside these counters. Registry access in `lance-context-core`, direct client calls in other controller/administrative modules, and other etcd clients remain outside the coverage above. Do not equate the sum with complete fleet/server RPC volume or infer zero traffic from an absent series.

Existing claim, informer-list and worker progress counters remain available and overlap this accounting. Do not add them to the new attempt counter. Worker `rollout_wal_cleanup_total{result="merged"|"noop"}` and `rollout_wal_generations_reclaimed_total` report work outcomes, not etcd calls. Missing-store early returns and generic stores are outside those rollout counters; background and legacy calls can share them.

## Dashboard queries

PromQL examples:

```promql
sum by (operation) (rate(etcd_operation_attempts_total[5m]))
sum by (operation, outcome) (rate(etcd_operation_completions_total[5m]))
```

In Datadog, use the corresponding imported monotonic-count metrics, `sum` across instances, `.as_rate()`, and group by `operation` (also `outcome` for completions). Use the actual OpenMetrics namespace/name mapping in that installation. Filter to the intended service/environment. Do not use the maximum pod counter or interpolate absent series as successful calls.

Compare the same workload/time window with server Range/Txn/Watch-start rates, successful merge generations, pending WAL and read503/latency. Handle process restarts before summing rates. Mixed-version rollout windows have partial coverage. Counts identify caller traffic; they do not prove that removing those calls is safe or that their traffic caused etcd latency.

## Validation

Deterministic recorder tests cover unpolled and dropped futures, underlying future destruction, error identity, compare-false success values, retries and interleaved operations. The existing private-etcd CI harness also runs `private_permit_operation_metrics_match_real_rpc_and_rejected_cas`: its instrumented lifecycle matches exactly 9 server Txn and 2 Range calls, including a rejected second reservation and a rejected progress publication after ownership release. Run it with:

```sh
python3 scripts/ci/test_commit_permits.py --etcd /path/to/etcd
```

That harness starts its own loopback etcd and joins its children; never supply a production endpoint to tests.
