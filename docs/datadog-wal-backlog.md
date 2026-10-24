# WAL backlog in Datadog

Use the `master_wal_backlog_*` metrics for backlog dashboards. They are explicit Prometheus **gauges**, without a counter-style `_total` suffix. This namespace is new so legacy masters cannot contribute their old, indefinitely retained scan values. The old `master_wal_pending_generations_*` and `master_stores_pending_*` gauges are no longer published by updated masters.

## What is measured

The existing registry scan computes one compact summary of the stats snapshot it successfully commits. A conditional write to one key in the service's own etcd verifies the original stats-writer token. An expired scanner cannot overwrite the new writer's summary. Failed/ambiguous stats commits never publish a new summary. This key is outside scheduler notification ranges and does not wake the task pools.

Masters read only this small key at startup and at most once per minute thereafter, with a five-second deadline. `/metrics` reads an in-memory cache only. There is no new table scan, per-table polling, ConfigMap, or request-path etcd call. At 90 replicas the steady-state read budget is at most 1.5 small-key reads/second, plus one summary write per completed scan. Replicas may differ for one refresh interval after a scan: this is a bounded propagation delay, not independent fleet totals.

The scope is the stores represented by the registry/stats scan, and **flushed** WAL generations; unflushed writer memory and stores outside the registry are not counted. It is a rolling scan, not a transactionally simultaneous observation of every table.

| Metric suffix after `master_wal_backlog_` | Meaning |
| --- | --- |
| `generations` | Sum of pending generations across the represented stores. |
| `largest_store_generations` | Largest store's sum across all its shards. Do not draw a read-cap line here. |
| `stores_over_1000` | Stores with >1,000 pending generations, including those above 4,096. |
| `largest_shard_generations` | Largest single-shard pending count observed. |
| `stores_with_shard_over_4096` | Stores with any shard >4,096. Reference-threshold risk, not observed HTTP 503s. |
| `observed_stores` | Stores represented by a complete, usable snapshot. |
| `snapshot_age_seconds` | Age from the older of scan start and oldest retained row observation. Scraping/re-reading cannot reset it. |
| `snapshot_complete` | 1 only if no registry entry failed observation, including failures without an older row. |
| `snapshot_fresh` | Observations are within `MASTER_BACKLOG_METRICS_MAX_AGE_SECS` (default 900 seconds). |
| `snapshot_readable` | This replica successfully read the shared key within its refresh deadline. |
| `snapshot_available` | Readable, fresh and complete. 0 means **unknown**, not an empty backlog. |
| `snapshot_failed_stores` | Failed observations in the shared snapshot. |
| `snapshot_revision` | Shared summary etcd revision, as a gauge value, never a high-cardinality label. |

On startup, read failure, malformed data, incomplete observation or expiration, backlog **values are omitted**, not set to zero. A successfully committed complete empty scan does publish zero. Scan-publication errors increment `master_backlog_snapshot_publication_errors_total`; the previous summary remains observable with its original age until expiration. Newer summaries refresh all replicas, even those that never win the stats-writer lock.

## Dashboard queries

The Mango master OpenMetrics configuration currently uses namespace `rocketkeep_master`; therefore the example queries use `rocketkeep_master.master_wal_backlog_generations`. If a deployment uses a different OpenMetrics namespace, change that prefix. Confirm metric type is gauge in Metrics Summary. Existing `metrics: [".*"]` collection already includes these names; do not add another scrape job.

Use the chart-building instructions below with the Datadog agent after deployment verification. Set the cluster filter to the cluster's actual tag and namespace to `rollout-db`. These instructions do not modify the live dashboard.

- Fleet values: `max:rocketkeep_master.master_wal_backlog_generations{$cluster,$namespace}.fill(null)`. `max` deduplicates replicas of the shared snapshot; **never sum replicas**. After publication it may show the prior larger snapshot for up to one refresh interval. The revision min/max diagnostic shows propagation disagreement.
- Keep `.fill(null)` so Datadog's gauge interpolation cannot extend an expired value. Do not apply `default_zero()` to backlog, age or coverage. Large graph rollups can still combine points within a time bucket; use a suitable resolution when inspecting short changes.
- Coverage: `max:rocketkeep_master.master_wal_backlog_snapshot_available{$cluster,$namespace}.fill(null)`. At least one current complete replica is usable. Alert separately on sustained zero **and no data**. A missing gauge never proves health.
- For publisher diagnosis, group by `pod_name` (or the actual Pod-identity tag emitted in your environment). Deployment/host can conflate multiple Pods. Do not hardcode a host or assume the scanner has a stable leader role.
- Coverage-age/failed-store charts are grouped by Pod, so a stale reader is visible. `snapshot_revision` is exposed numerically; do not turn each revision into a tag.

4,096 is a **per-shard reference threshold**. Two shards of 2,500 make a store total of 5,000 without either hitting the default read cap. Workers may override or disable `ROLLOUT_WAL_PENDING_MAX_GENERATIONS`; confirm their deployed setting before interpreting the reference-threshold chart as cap risk. The >1,000-store count is not a measured latency threshold, and the old arbitrary count-of-20 line is not an established SLO.

## Refusals, slow reads and no data

Backlog is a risk signal. Use worker HTTP response counters/application signals and latency histograms for actual responses; neither a low backlog nor a missing refusal series proves zero 503s/slow reads. In particular `rollout_reads_refused_pending_total` is recorded at a shard-cap rejection and can be incremented more than once for one request reading multiple shards. Do not label it exact refused HTTP requests/second. Its per-store series is created on an event: no series may mean no event since process start, missing collection, a wrong query or a missing process. On-demand-index counters have the same no-event versus missing-telemetry ambiguity.

Check `up`/collector health and the relevant endpoint/process coverage alongside event counters. Do not fix a missing event counter by filling an unverified scrape gap with zero. These hints exclude ingress-only failures and errors while streaming response bodies.

## Instructions for the Datadog agent

Update the existing WAL backlog dashboard after confirming the new metrics are being collected. Preserve its other panels. Use the actual production cluster tag and `kube_namespace:rollout-db`; do not mix clusters or staging. In the queries below, `$cluster` and `$namespace` are dashboard template variables with prefixes `kube_cluster_name` and `kube_namespace` respectively. Confirm those tags exist before building.

Build time-series charts with solid, thick lines and a default one-hour window. Use gauge values, not rates or counts. Every query below must end in `.fill(null)`; do not sum across Pods or use `default_zero()`. The common query is `max:rocketkeep_master.master_wal_backlog_<suffix>{$cluster,$namespace}.fill(null)`.

| Chart title | Metric suffix | Threshold / interpretation |
| --- | --- | --- |
| Pending WAL generations — fleet | `generations` | No fixed threshold. |
| Largest single-store backlog | `largest_store_generations` | No read-cap threshold; sums shards. |
| Largest single-shard backlog | `largest_shard_generations` | Dashed reference line at 4096; configured caps can differ. |
| Stores with a shard above 4096 | `stores_with_shard_over_4096` | Dashed line at 0; above zero indicates risk, not measured 503s. |
| Stores above 1000 pending | `stores_over_1000` | Includes stores above 4096. Do not label 1k–4k or imply measured slow reads. |
| Usable backlog snapshot | `snapshot_available` | 1 = usable, 0 = unknown. Alert on sustained 0 and on no data. |
| Stores represented | `observed_stores` | Coverage of the committed scan, not an independent registry count. |

Add diagnostics grouped by `pod_name` (verify the actual Pod tag): `snapshot_age_seconds`, `snapshot_failed_stores`, and `snapshot_readable`. For example, `max:rocketkeep_master.master_wal_backlog_snapshot_age_seconds{$cluster,$namespace} by {pod_name}.fill(null)`. Draw the configured observation-age limit on the age chart (default 900 seconds).

Add one propagation chart with two ungrouped queries, `min:rocketkeep_master.master_wal_backlog_snapshot_revision{$cluster,$namespace}.fill(null)` and the corresponding `max:`. These should converge after the 60-second refresh interval plus the read deadline. Do not interpret a revision difference as generations.

Replace the four old backlog queries; remove a fifth duplicate backlog panel only if it contains no additional information. Keep actual HTTP error/latency and index-event panels separate. Do not rename pending-risk series as observed 503s or fill missing event telemetry with zero.

Validate over at least two completed production scan publications: totals should follow the new snapshots, revisions should converge, and the age curve should reset only on a new scan. A gap or availability 0 is a telemetry problem to investigate, not proof of an empty backlog. Never pin a query to today's scanner Pod or host.
