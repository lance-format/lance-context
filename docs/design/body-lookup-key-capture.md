# One-shot historical lookup input capture

`Runtime::body_lookup_capture()` provides a local diagnostic handle for a
full-body runtime. It uses that runtime's actual publisher acquisition identity.
It adds no endpoint, index, scanner, background task or automatic logging.

Retain the handle before moving the runtime into the existing owner loop. At a
chosen diagnostic boundary, arm it once with a separate bounded memory budget:

```rust,ignore
let capture = runtime.body_lookup_capture()?;
// Start the existing runtime as usual and retain `capture` with its diagnostics.
let receipt = capture.arm(MergeMemoryBudget::new(8 << 20))?;
let keys = tokio::time::timeout(Duration::from_secs(30), receipt).await???;
// On the caller's diagnostic task, write to its existing protected output.
serde_json::to_writer(output, &keys)?;
drop(keys); // Releases admitted key/metadata buffers.
```

Only one request may be armed per runtime. Dropping its receiver cancels it;
an explicit later arm can replace the cancelled request. An all-staged window
does not consume it. A quiet or stopped runtime may produce no capture, so the
caller owns the timeout. No capture is armed by default.

The capture point follows the atomic base/staging view selection and precedes
native historical lookup. It includes all distinct unresolved canonical
composite keys, preserving their JSON string-array representation. Decode each
string as a JSON array to get original UTF-8 components in `primary_key_columns`
order. It contains no body arrays, table URI, storage options or credentials.
Keys are still workload identifiers: route the artifact through the deployment's
existing protected diagnostics, never general metrics or public logs.

Each receipt binds one capture ID, publisher owner/epoch/grant, table incarnation,
schema fingerprint, source group/bucket, captured Lance version, that manifest's
fragment IDs, and input first/last WAL record, request count, wire bytes and
speculative flag. The WAL interval may contain gaps or unrelated records. The
fragment IDs describe the whole captured base, not actual scan/pruning results.

This is an attempted query's input, not a successful lookup, applied cut or source
ACK. A later scan can fail, be cancelled or prove stale. Capture does not retain
a Dataset or GC lease and does not prove the owner remains current. A subsequent
read-only projection must open this exact version and record any unavailable-file
error; silently using a later version invalidates the comparison.

Admission precedes copying. The complete buffer size must fit the supplied budget;
insufficient capacity returns a capture error without truncating keys or changing
ingestion. The reservation follows the returned receipt until drop, including
while waiting in the one-shot channel. Caller serialization, allocator overhead
and process RSS are not covered by this key-buffer budget. Keep it separate from
apply/publication budgets so diagnostics do not consume their capacity.

RF3 tests exercise staged/frozen exclusion, publication into a new base, exact
owner/version/fragment bindings, cross-page eight-request intervals, cancellation,
single pending request, memory refusal without ingestion failure and release.

For an embedding host with shared memory accounting, use
`handle.arm_with_pool(diagnostic_budget, host_pool)` where `host_pool` implements
`lance_context_wal::replication::runtime::client::payload_memory::Pool`.
The existing nonblocking pool admits the estimated copied key/metadata buffers
before allocation. Its reservation is owned by the capture and survives copying,
sending, caller timeout and failed send, then drops after the captured data.
A pool refusal fails only the diagnostic; ingestion continues. The separate
`MergeMemoryBudget` remains a per-capture ceiling. A guard owned only by the caller
or a check that this budget is currently zero cannot cover a callback that took
an armed request before the receiver was cancelled. Serialization and output
buffers still require the host's own accounting.
