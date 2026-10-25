# Task admission telemetry

`master_task_admissions_total{kind,origin,outcome}` counts acknowledged admission
decisions in the task store. It does not count scheduler calls as newly created
tasks. The change adds no etcd requests or persisted fields and does not change
task selection, retries, fences, frontier consumption, or recovery behavior.

| Outcome | Evidence |
| --- | --- |
| `created` | The task creation transaction returned success. |
| `deduplicated` | Admission returned an existing queued/running task, or the cached queued-task acknowledgement CAS succeeded. |
| `superseded` | Admission returned a definitive changed observation / rejected frontier or request. This can include lost publication ownership; it does not mean the table has no work. |

Intermediate CAS conflicts that retry inside one call do not count as separate
outcomes. A failed cached acknowledgement falls back to ordinary admission and
does not emit its own outcome. Transport errors, cancelled futures, and retry
exhaustion emit no definitive outcome. A transaction can commit while its reply
is lost, so this counter is **not an exactly-once durable task audit**. Ambiguous
aggregate resolution does not count again as a fresh admission.

Origins are a fixed set:

| Origin | Admission path |
| --- | --- |
| `explicit` | Default direct enqueue, including explicit dependency chains. |
| `periodic_stats` | Periodic stats sweeps, including explicit sweep invocation. |
| `scan` | Early merge admission from a fresh scanner observation. |
| `aggregate_frontier` | Leader admission of an aggregate seal frontier. |
| `writer_request` | Durable merge-request admission and queued-task acknowledgement. Requests can come from writers **or continuation/recovery producers**; this is not a claim about the original producer. |
| `execution_recovery` | Admission conditional on an observed execution revision. |
| `failure_recovery` | Admission conditional on an observed failure revision. |
| `wal_tail` | Low-count WAL tail service. |

Each recorded outcome emits a structured `task admission outcome` log with
`kind`, `origin`, `outcome`, `target`, and the returned/created `task_id` when
available. Stats paths also include `observed_base_version`, `scanned_at_ms`, and
`pending_wal_generations`; conditional paths include `observation_revision`.
These identifiers and observations are deliberately absent from metric labels.
There are at most 4 kinds × 8 origins × 3 outcomes per process.

## Dashboard and diagnosis

- Plot the summed **rate** of `master_task_admissions_total`, grouped by `origin`
  and `outcome`, filtered to the production service/cluster. Sum rates across
  masters: executors and publishers can legitimately record different decisions.
  Do not take the maximum of cumulative pod counters or filter everything to one
  leader.
- Plot `outcome:created` separately from `outcome:deduplicated`. Neither is merge
  throughput: compare against committed/reclaimed generation counters and actual
  pending observations over the same window.
- Join creation logs to completion logs by task ID. Repeated new task IDs with
  `origin:periodic_stats` and the same observation evidence identify a reused
  stats row. A zero-result merge alone is not proof of wasted work: scoped passes,
  new writer seals, partial passes, and recovery still need examination.
- A rise in `superseded` is not an error count. Use existing request/queue error
  and etcd latency metrics for errors and ambiguous calls.

The legacy `master_task_enqueued_total{kind}` now counts only acknowledged
`created` outcomes from these admission paths. Older versions counted some
dedupe returns and omitted direct leader paths. Do not compare that legacy
counter across a mixed-version rollout as if its meaning were unchanged. Use the
new metric with deployment coverage for attribution.

This telemetry does not suppress stale-stat admissions. Any suppression change
must preserve new seals, remaining work after bounded passes, and execution or
failure recovery, with storage evidence rather than a zero-result or age test.
