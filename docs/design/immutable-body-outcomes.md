# Distinguish lookup work from applied rows

For a single-body owner, `Service::body_lookup_observation()` records completed
lookup windows. `requested_keys = staged_keys + history_hits + history_misses`
until counter saturation. Each window counts distinct requested composite keys.
`history_hits` counts exact matches in the captured Lance base, excluding staged
hits and unrelated cross-product candidates returned by the scan. A miss means
the successful scan exhausted its results without finding that key. Failed,
cancelled or unadmitted lookups produce no hits or misses. A repeated lookup of
the same keys counts again; these totals measure work, not unique ingestion.

`Service::observation().await.0.immutable_apply` records successful ordered
registrations under the same staging lock as the applied position. `rows` counts
all input rows, including repeated keys within a batch. `inserted_rows` counts
new staged bodies; `duplicate_rows` counts rows skipped because their key was
already found by the captured lookup, active/frozen staging, or an earlier row
in the batch. Their sum equals `rows` until saturation. `batches` counts these
successful registrations. Stale lookups, backpressure and rejected batches add
nothing, and request retries resolved before registration add nothing.

Compare deltas within one owner lifetime. Both sets of counters reset when the
owner/staging is reconstructed, and real applications of a recovered WAL suffix
count again. They are not durable deduplication state, publication ACKs or global
unique-row counts. A historical miss can still become a duplicate at ordered
apply, and a completed lookup may be invalidated before application. Do not
subtract the two sets of counters to infer lost rows.

Zero new rows with advancing publication can be explained by growing applied
duplicate counts: request results and WAL progress still require publication
even when no body file is added. Neither a version increment nor a completed
lookup alone establishes that explanation. Keep table, shard, owner and
observation-window identity when comparing these counters with installed rows.
