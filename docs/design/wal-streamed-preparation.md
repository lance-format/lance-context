# Streamed preparation within a publication

Refs #378.

The application currently fetches and decodes every selected source before it
starts any Lance file writer. An optional `worker.publication.pipeline_preparation`
setting removes that barrier while preserving the original buffered path as the
default and as a comparison control. This is local preparation concurrency; it
does not distribute one table across worker Pods.

Both paths use the same immutable, pinned publication plan, the same round-robin
assignment of whole source records to writers, and the same ordered descriptor
aggregation. The streaming path moves Arrow batch handles into Lance 13's public
streaming insertion API. It adds no row conversion or payload reserialization.
Arrow alignment repair and Lance encoding can still copy buffers; this is not an
end-to-end zero-copy claim. The streaming insertion API also retains its own
encoding and write buffering, so removing the application barrier does not prove
that network or object writes overlap on every workload.

All writers share one fetch/decode semaphore, sized by `fetch_parallelism`.
Blocking decode owns its permit and source reservation even if a stream is
cancelled. Each writer buffers at most `fetch_parallelism` decoded source results;
the aggregate source reservation still covers the entire admitted plan. It is not
a hard bound on Lance encoder workspace or process RSS.

File writers run within the existing detached admitted job. They all finish
before their results are aggregated, including when another source stream fails.
Only a complete successful descriptor set can enter durable Prepare. An error can
leave immutable orphan files, but cannot publish partial rows or retire sources.
Cancellation of an observer cannot release the reservation while its admitted
writers or blocking decoders still run.

The `blob_write` phase now encloses source waiting for streamed writers. Its time
overlaps fetch/decode counters and must not be added to them or interpreted as
isolated object-store latency.

Qualification covers the existing real WAL/Lance preparation-restart,
publication-restart, exact retry, retirement and full-column readback scenario in
both modes. A second test injects a source error while another actual Lance writer
is blocked, cancels its observer, and checks that the reservation stays held until
that writer exits and no new manifest is published. Results and an actual-process
application test are required before deployment; no throughput improvement is
claimed from source changes or local correctness tests.
