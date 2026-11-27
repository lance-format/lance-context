# Applied timing observations

`Service::phase_observation()` exposes bounded, process-local cumulative
observations. The following phases distinguish work inside an Applied wait:

| Phase | Interval |
| --- | --- |
| `annotation_ordered_wait` | Acquiring the shared gate in apply, freeze or installation. |
| `annotation_freeze_hold` | Capturing a frozen checkpoint while holding that gate. |
| `annotation_install_hold` | Installing a confirmed publication, including transaction validation, warming, base switch and frame/result retirement, while holding the gate. |
| `annotation_full_body_apply` | Full-body registration, including core locks, validation, admission and membership warming. Counts attempts, including stale lookups and backpressure; completion is not an Applied row count. |
| `annotation_apply_handoff` | After resolving an Applied/Rejected outcome: lookup accounting, frame retention, source/result accounting and delivery of local progress. Contains no await. |
| `annotation_apply_memory_pressure` | An episode reporting apply-memory pressure. |
| `annotation_source_window_pressure` | An episode reporting source-window pressure. |
| `annotation_staging_pressure` | An episode reporting staging pressure. |

The three pressure episodes end on a different reported reason or successful
progress delivery. Repeated reports of the same reason keep one interval.
They include retries and intervening work, not just time a resource is
unavailable. Dropping the driver cancels an outstanding episode. There is at
most one episode per driver; no request identities are retained for timing.

`Observation::immutable_membership` also exposes `apply_warming` and
`publication_warming`. These measure the existing optional membership loops,
once per batch, including resident probes and skipped/empty warming. They do
not time each key. Apply warming includes the staging checks around those
probes; it excludes the subsequent state-write-lock reacquisition. Publication
warming excludes transaction validation, the core mutation-lock acquisition
and the final base switch. The counters are children of full-body apply and
publication installation respectively. Existing `warms` and `refusals` remain
aggregate cache-operation counts, not counts attributed to either loop.

For all these observations, elapsed/max time includes finished cancelled work,
but excludes work still running. `active` is an approximate count derived from
independently sampled started/completed/cancelled counters. Phase `completed`
includes its failed subset, whereas the infallible optional warming loops have
only completed/cancelled outcomes. A cancelled interval does not establish that
its side effects were undone. Reconstruction resets all counters.

Parent phases, different producer waits, lookup windows and pressure episodes
can overlap. Do not sum them into wall/CPU time or subtract cumulative history
time from cumulative Applied waits to invent a per-request remainder. The
existing `commit` phase ends before `annotation_install_hold`. These timings
neither change durability/read visibility nor prove a production bottleneck.

`Service::body_lookup_observation()` additionally reports full-body batches
whose actual atomic registration inserted zero first-create rows:
`all_duplicate_batches`, `all_duplicate_rows`,
`all_duplicate_known_arrow_batches`, `all_duplicate_arrow_backing_bytes` and
`all_duplicate_reservation_bytes`. Rejected/stale/backpressured attempts do not
increment them. The original registration API remains available; its counted
variant returns the same outcome and this call's insertion count without a
second observation or an await between applying and handing off progress.

The existing frame-retention walk deduplicates backing allocations within one
frame; aliases are not summed as separate buffers. Unknown allocation capacity
keeps its existing conservative reservation and does not enter the known Arrow
byte total. The reservation total includes bookkeeping and may overlap backing
bytes: do not add them. These are cumulative bytes admitted for these batches,
not current retained memory, RSS, bytes released or proven savings. Frames,
publication cuts and request results keep their existing lifetimes.
