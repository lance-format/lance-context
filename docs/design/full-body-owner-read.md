# Read an immutable body before materialization

The full-body owner and forwarding routers expose
`POST /api/v1/annotations/{table}/read-body`. Install the routers behind the
application's existing authentication boundary, as with annotation RPCs.
Trusted deployment configuration supplies owner addresses.

The JSON request is `{"turn":"<business key>"}`. Construct that key with the
public `TurnIngestionSpec::business_key` used by ingestion. A
composite key includes every primary-key field; a physical Lance row ID is
not a business key. Forwarders route using the spec's session field and the
persisted shard layout.

For a spec whose primary key is `(session_id, id)`, call
`spec.business_key(&[session_id, id], limits.value_bytes)`. The public decoder
`http::annotations::full_body::decode(bytes, &spec, &key)` verifies the exact
original schema, one-batch/one-row shape, valid non-null required fields and the
complete requested key. Bound the HTTP response and admit input/decode workspace
before receiving it; retain that admission while using the decoded batch.
Uncompressed binary values can share input buffers, while alignment repair may
allocate. Decode only a successful Arrow response; 204 denotes absence.

A successful response is an Arrow IPC file containing exactly one row with
the original user schema, with `Content-Type: application/vnd.apache.arrow.file`
and an explicit `Content-Length`. A confirmed missing key returns 204 with no
body. Invalid keys return 400; unavailable capacity, failed reads and fenced
owners return errors, never a false 204. This route accepts full-body tables.
The existing narrow annotation route continues to serve annotations separately.

The owner waits through its acquisition recovery cut, captures a compatible
active/frozen/base view, and resolves the body without scanning the WAL tail.
Its post-resolution quorum fence rejects a takeover during storage I/O.
Acknowledged writes in this epoch have already been applied to that shared
view. A newer committed but unapplied write does not extend the read's wait.
An owner whose applier has stopped fails closed.

Read/encoding workspace and output are admitted together, without waiting for
additional credit while retaining a read. Both owners and forwarders refuse
when their configured byte budget cannot admit a response. These conservative
charges derive from the table's maximum record size, so a valid ingestion
budget may need additional capacity for body reads. Admission charges are not
a bound on total process RSS.

The owner serializes the selected Arrow row once into a size-limited IPC buffer.
The forwarding ingress streams those bytes without Arrow decoding, re-encoding
or coalescing the entire response. Retained response bytes keep their read view
and reservations until consumption or cancellation releases them. This is not
an end-to-end zero-copy claim.

This endpoint provides a current point read for a known business key. A full
session query also needs discovery and merging of unmaterialized keys; adding
this point route does not make an existing snapshot-only session reader fresh.
Owner discovery, authentication and application reader integration remain the
embedding application's responsibility.

Focused regression command:

```sh
cargo test -p lance-context-ingest --features service,http-client --lib full_body
```
