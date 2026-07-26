# Rollout artifact blobs: memory behavior and streaming roadmap

Status: describes the memory characteristics of the rollout `binary_payload`
("blob") path after the download-streaming + budget work, and proposes the next
(larger) step toward true streaming storage. The first three items below are
**implemented**; the fourth (§4) is a **proposal**, not yet built.

## Background: where blob bytes live in memory

`binary_payload` is stored as a plain **inline `LargeBinary`** column, not a
lance blob-v2 (`lance-encoding:blob`) offloaded column. This is deliberate:
rollout reads go through the MemWAL LSM scanner, which has no blob-materialization
step, so a blob-v2 column reads back as `None` there. Inline storage is currently
the only encoding that round-trips (see the module docs in
`crates/lance-context-core/src/rollout_store.rs`).

The consequence is that **every blob is fully materialized in RAM** at each hop:

- **Upload**: the whole request body is buffered — JSON base64 (~+33% expansion)
  via `axum::body::to_bytes`, or each multipart part via `field.bytes()` — then
  appended into a `LargeBinaryBuilder` whose `finish()` copies it into a
  contiguous Arrow buffer.
- **Download**: `get_blob` locates the row, then `take_rows` materializes the
  `binary_payload` Arrow buffer and `.to_vec()` copies it into an owned `Vec<u8>`
  (≈2× the blob size at that instant).

With a 1 GiB per-request ceiling (`MAX_ROLLOUT_UPLOAD_BYTES`) and no concurrency
cap, N concurrent large requests needed ≈`2 × size × N` bytes and could OOM the
worker.

## 1. Streaming downloads (implemented)

Both blob-serving handlers — worker `fetch_rollout_blob` and master
`download_experiment_blob` — now send the payload as a **chunked** body
(`blob_stream_body`, 256 KiB frames) instead of one `Body::from(Vec<u8>)` frame.
Frames are refcounted `Bytes` slices of the one backing allocation (no per-frame
copy). This removes the extra full-blob copy the HTTP send path would otherwise
hold and lets a slow client apply backpressure at frame granularity rather than
after the entire payload is queued. A `Content-Length` header is still set so the
response is not chunked-transfer on the wire when the length is known.

> Note: this does **not** remove the single in-RAM `Vec<u8>` produced by
> `get_blob` — the bytes are still fully read from storage first. True
> read-from-storage-in-frames requires §4.

## 2. Single-scan record + blob for downloads (implemented)

`download_experiment_blob` previously did `get_by_id` (a full-row point scan) and
then `get_blob` (a second point scan over the same shard) — two scans to serve
one download. `RolloutStore::get_record_with_blob` folds these into one
base-first scan that returns `(record, payload)` together, halving the scan work
on the master download path. It keeps the base-table-first fast path and the
NotFound-tolerant WAL fallback semantics.

## 3. In-flight blob-byte budget (implemented)

A process-wide admission budget (`BlobBudget` in the worker's `AppState`, sized by
`ROLLOUT_MAX_INFLIGHT_BLOB_BYTES`, `0` = disabled) bounds the total blob payload
held in memory across concurrent uploads and downloads:

- **Uploads** reserve their declared `Content-Length` before the body is
  buffered; the reservation is held for the whole handler.
- **Downloads** reserve the payload size once known and hold the reservation
  inside the streamed body, so a slow client keeps the bytes accounted for until
  the last frame flushes.
- When the budget cannot admit a request it is rejected with **`503` `OVERLOADED`**
  (a `rollout_blob_budget_rejections_total` metric is incremented) instead of
  proceeding to allocate. Backpressure moves to the edge, not the allocator.

The budget bounds *concurrency*, not maximum blob size: a lone request larger
than the whole budget is admitted when the instance is otherwise idle, so a
single big blob never permanently 503s against its own limit.

Operational guidance: size `ROLLOUT_MAX_INFLIGHT_BLOB_BYTES` to a fraction of the
pod's memory limit that leaves headroom for the ≈2× transient copy per in-flight
download plus base overhead — e.g. for a 4 GiB pod, a budget of ~1–1.5 GiB.

## 4. True streaming storage (proposal — NOT implemented)

Items 1–3 bound and smooth memory but every blob is still fully resident once per
request. Eliminating that requires storing/reading blobs in a form that supports
**ranged, chunked I/O**, i.e. lance blob-v2 offloaded columns + `BlobFile` range
reads.

### The blocker

The rollout read path is the MemWAL LSM scanner, which unions the base table with
flushed WAL generations and has **no blob-materialization step** — a
`lance-encoding:blob` column reads back as `None` through it. So we cannot simply
flip `binary_payload` to blob-v2 without the scanner learning to resolve blob
descriptors.

### Proposed approach (two sub-steps, each shippable)

1. **Reader-side: `BlobFile` range read for the base table.**
   Keep `binary_payload` inline in the WAL (small, short-lived generations) but,
   for the *base table*, store artifact bytes as a blob-v2 column. Change
   `get_blob`/`get_record_with_blob` so that on a base-table hit it opens the row's
   blob descriptor and returns a `BlobFile`/reader that streams ranges from object
   storage, wiring that reader into `blob_stream_body` (frames pulled from storage
   on demand rather than from an in-RAM `Vec`). WAL-fallback rows stay inline (they
   are the un-merged tail and are folded into the base on merge). This gives
   streaming reads for the common already-merged case — the 99% path — without
   touching the LSM union.

2. **Writer-side: streamed ingest into the blob column.**
   Replace the "buffer whole body → `LargeBinaryBuilder`" ingest with a streamed
   writer that appends blob bytes to the blob-v2 column in frames as the request
   body arrives (multipart parts stream naturally; JSON base64 would need a
   streaming base64 decoder or be deprecated in favor of multipart for large
   blobs). Upload peak memory drops from O(blob) to O(frame).

### Effort / risk

- Sub-step 1 is a core (`rollout_store.rs`) change plus a lance API dependency on
  `BlobFile` range reads; medium effort, low blast radius (reads only, base-only).
- Sub-step 2 touches the ingest handler and the write path; higher effort and
  needs care around the atomicity guarantees of a rollout append (currently one
  `RolloutStore::add`).
- Merge/compaction must learn to carry blob-v2 columns from WAL-inline to
  base-offloaded; verify the LSM merge path preserves the descriptors.

### Recommendation

Ship §1–3 now (this PR). Schedule sub-step 1 (base-table `BlobFile` streaming
reads) next as its own PR — it delivers the bulk of the read-side memory win —
and treat sub-step 2 (streamed ingest) as a follow-up once the read path proves
out.
