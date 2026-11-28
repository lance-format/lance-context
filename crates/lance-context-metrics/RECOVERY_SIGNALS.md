# Bounded application request hints

Set `RECOVERY_SIGNALS_ENABLED=1` (or `true`) at startup to enable
`GET /api/v1/internal/recovery-signals?nonce=<alphanumeric-or-hyphen>`.
Disabled instances return 503. This endpoint is intended for the internal
recovery observer, subject to the same network isolation as other internal
metrics routes.

The existing HTTP middleware records attempted data access before awaiting the
handler, and records table-specific read 5xx responses when headers return.
Thus a still-running request can suggest a table to inspect. Maintenance routes
(merge, compact, flush, checkout) do not generate their own demand. GET/HEAD are
reads; POST search/retrieve/get-rows are also reads. Responses do not include
bodies, query strings, record IDs or user IDs. Canonical names use the recovery
convention: rollout `name`, or `generic:name`, `context:name`, `datagen:name`.

This is a lossy, five-minute observation window, not an operation progress or
ownership protocol. It cannot authorize cancellation, replacement or writes.
It does not see failures before application middleware, failures while sending
a streaming body, or requests on unmatched routes. Consumers must check
Pod/container incarnation and readiness around reads, nonce, sample freshness,
coverage start time and `complete`; a missing/disabled endpoint is unknown,
never zero errors. A new process has no coverage before its stated start time.

The cache has 32 shards, at most 128 entries per shard, bounded name lengths,
and a maximum 1 MiB response. Requests use `try_lock` and never wait for the
observer. Contention/cardinality loss increments `dropped_total` and makes
`complete=false` for the following five minutes; coverage recovers after that
loss ages out. Busy shards or reply truncation also mark a snapshot incomplete.
Concurrent scrapes return 503 rather than building several snapshots at once.
Recent read errors have priority if long table names fill the response budget.
Successful access never erases an unexpired read error.

Generic single-row GET failures can additionally include `read_error_diagnostic`
on the same table entry. Its closed `stage` values distinguish store open,
initial lookup, refresh after a miss, and retry lookup. `kind` identifies the
typed error family (`internal`, `overloaded`, `lance_io`, `lance_timeout`,
`lance_arrow`, or `lance_other`), not a provider-specific root cause. No raw
error text is retained. The process-local `sequence` identifies that observation
together with the verified worker incarnation and `coverage_started_at_unix_millis`;
it is not a client request ID or a storage request ID.

This optional diagnostic comes from that exact response's in-process extension;
it does not alter HTTP headers, bodies, status, retries, or store locking. It
expires with the corresponding read error and becomes null if a newer error has
no diagnostic. Consumers must accept absent/null fields from older instances or
other routes and must not carry a previous diagnostic forward onto a new error.

Validation: metrics crate unit and real Axum middleware tests, Clippy with
warnings denied, and opt-in/disabled tests in isolated processes. The manual
release-mode microbenchmark performs one million parse+cache-update operations
in about 410 ms on the development pod (~0.41 microseconds each). This measures
an uncontended in-process path, not end-to-end production latency or concurrent
scrape overhead; production rollout still requires observing loss and memory.
