# Request hints for recovery

With `RECOVERY_SIGNALS_ENABLED=1`, a worker exposes recent per-table request
hints at `GET /api/v1/internal/recovery-signals?nonce=<caller-nonce>`.
The existing schema 1 response now also includes slow-read observations:

| Field | Meaning |
| --- | --- |
| `slow_read_header_threshold_millis` | Envelope threshold: 5,000 ms. |
| `read_slow_at_unix_millis` | Completion time of the latest qualifying read. |
| `read_slow_status` | That read's HTTP status, including 2xx or 4xx. |
| `read_slow_header_millis` | Time from middleware entry until response headers. |

These fields are additive. Consumers of older workers must treat absent fields
as unavailable observations, not proof of fast reads. A null per-table field
means there is no retained slow-read observation in the five-minute window.
Successful activity does not clear unexpired errors or slow-read observations.

This catches cases such as a `/records` request that spends four minutes scanning
before returning 404: it was absent from the 5xx hints and could not be assigned
to a table from route-level latency histograms. Neither request parameters nor
record IDs or payloads enter this cache. Maintenance routes remain excluded.

Request recording uses a nonblocking lock and at most 4,096 entries. Snapshots
remain capped at 1 MiB; recent 5xx hints are preferred, then slow reads, then
ordinary activity. Check `complete`, `dropped_total`, coverage start, and nonce
before interpreting an empty snapshot. Loss is explicitly reported.

The threshold only selects observations. It does not cancel requests, retry
writes, declare a merge stalled, or initiate a merge. Consumers should correlate
the table with fresh WAL metadata and actual execution progress before choosing
a recovery action. A slow read can have causes other than WAL backlog.

These observations exclude response-body streaming and requests that have not
returned headers (including disconnected requests). Use client or ingress
measurements for end-to-end latency. An empty window with no reads is not proof
that the read path is healthy.
