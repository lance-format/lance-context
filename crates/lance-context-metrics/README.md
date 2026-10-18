# Runtime resource observations

The shared router exposes `GET /api/v1/internal/runtime-resources?nonce=TOKEN`
on both master and worker HTTP ports. It reads the current container's cgroup-v2
`memory.current`, `memory.max`, and `memory.events` directly, without opening
stores, acquiring merge locks, running commands or contacting other services.
Each kernel file is limited to 8 KiB. Responses use `Cache-Control: no-store` and
echo an optional alphanumeric/hyphen nonce (up to 64 bytes).

A successful response contains schema version 1, sample time, memory usage/limit,
and the raw `oom` and `oom_kill` counters. An unlimited limit is JSON `null`, not
an invented memory budget. Missing, unsupported, oversized or malformed cgroup
observations return HTTP 503; no previous successful observation is cached.
Memory pressure or prior OOM is reported as data, not hidden behind a generic
HTTP success/health flag. This endpoint does not restart writers or infer stalled
merge progress.

Recovery callers must bind the request to the actual Pod IP/container identity,
check that identity before and after reading, verify their fresh nonce and reject
stale, failed, missing or unlimited-budget observations. Existing readiness,
expected-worker configuration and OOM/memory admission rules still apply. The
endpoint supplies evidence; it does not authorize takeover or replace native
ownership/fencing.
