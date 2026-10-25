# Keep owner fencing queries small

Annotation HTTP requests validate the owner's catalog authority before work and
before returning success. Fetching the complete publisher table for that check
also reserves space for publication plans, prepared file descriptors, and retained
manifests. Under the default 4 MiB runtime control budget, only two of those
responses fit concurrently. A read, an update, and the background publisher can
therefore exhaust that budget even with tiny annotation payloads.

The catalog now has a `PublisherBinding` quorum query containing only authority,
destination, mutation contract, and table incarnation. Owner checks use this
projection and the existing 16 KiB control reply allowance. Publication still
reads its full state when needed. The same epoch, scope, and incarnation checks
remain in force; the change does not add automatic mutation retries.

The new query/value variants are appended to the RPC enums, preserving the
existing variant encodings. Upgrade WAL servers before sending this query from
new annotation owners. This query is not a persisted WAL record or checkpoint
format change.

The opt-in pressure test runs two writers through separate forwarding HTTP
listeners against one RF3/Lance owner. Each writer updates a different turn and
reads through the other ingress after every acknowledgment. One data request may
enter the owner at a time; staging retains at most three entries and the active
source window at most two records. The test counts definite pre-submit refusals,
checks every acknowledged value, samples bounded staging during repeated
publication, and requires complete drain at the end. Unknown mutation outcomes
fail the test. Only explicit `not_submitted` refusals resubmit the original request.

```bash
ANNOTATION_PRESSURE_SECONDS=60 cargo test -p lance-context-ingest \
  --features service,http-client --lib annotation_http_sustained_pressure \
  -- --ignored --nocapture
```

The output includes per-writer ACK/read latency upper bounds in 1 ms buckets,
refusal counts, staging/source samples, and final drain time. This two-turn test
proves behavior under repeated churn; it does not qualify cold lookup, large-table
index cost, deployment throughput, or a production latency target.

Refs #378.
