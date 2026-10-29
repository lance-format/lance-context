# Running one annotation shard

The ingestion binary provides `annotation-owner CONFIG` and
`annotation-ingress CONFIG`. They operate on an already provisioned managed
annotation generation whose narrow columns and mutation contract match the
configuration. Existing unbound Lance tables require an explicit handoff before
these roles can use them; normal startup does not adopt a table or overwrite its
bootstrap identity.

For an existing rollout table, run `annotation-bootstrap CONFIG` once before
starting the owner. Stop and drain the old writer first, then persist the exact
Lance version and native transaction UUID in `annotations.source`, for example:

```json
"source": {"version": 42, "transaction": "00000000-0000-0000-0000-000000000042"}
```

The UUID is read from that version's native transaction, not generated for this
field. Keep the existing table binding, incarnation, URI, complete `bootstrap`
identity, and source snapshot together for recovery. The command adds missing
nullable `labels`, `status`, and `score` columns using a metadata-only native
transaction and registers the resulting version as the managed genesis. Existing
values and fragment/file references are preserved. A null physical label list
means an empty logical label set. The source needs Lance v2 data files and unique,
non-null UTF8 turn IDs. Existing annotation column types must match the contract;
`score` and `status` must be nullable so the contract can clear them.

This is an offline handoff of a quiesced, otherwise unmanaged table. A conflicting
WAL policy is rejected before writing Lance. The command neither drains another
writer's WAL nor migrates a live immutable-ingestion binding. Following handoff,
route annotation reads/writes through these roles: the managed manifest catalog
selects subsequent publications, so a plain canonical Lance open is not the
annotation read path. Keep old writers stopped. Combining new ingestion with
point updates remains separate work.

### Separate narrow annotation table

For a fragmented body table, the same `annotation-bootstrap` command can create
a separate narrow destination before handing it to the owner. Configure
`tables[0].uri` as a fresh annotation URI and replace `annotations.source` with:

```json
"projection": {
  "uri": "s3://example/body-table.lance",
  "storage_options": {},
  "snapshot": {
    "version": 42,
    "transaction": "00000000-0000-0000-0000-000000000042"
  }
}
```

The source must be a quiesced, unmanaged rollout table with unique, non-null
UTF8 `id` values and a non-null UTF8 `rollout_id` grouping key. Stop and drain its
old writer before starting one bootstrap process. Persist the exact source native transaction UUID, both
URIs, the target table/incarnation and complete bootstrap identity before the
first attempt. Source credentials remain in configuration and are excluded from
the persisted provenance. Source and destination may have different storage
options; use a separate, exclusively owned destination root.

Projection reads only `id`, `rollout_id`, `labels`, `status` and `score`, preserving
existing annotations and filling missing annotation columns with null. It leaves
every source file and its manifest unchanged. The immutable `rollout_id` is the
real grouping key; it also leaves an unchanged column for Lance 13's narrow
`RewriteColumns` operation. The target receives stable row IDs and an ID B-tree
before the existing fenced handoff selects version 3.

Rows flow directly from the native projected scanner into the native writer;
there is no whole-table batch collection or body copying. Scanner I/O, output
batches and writer buffering use slices of `materializer_memory_bytes` (at least
1 MiB). Up to four source fragments overlap their I/O through the shared scan
buffer; oversized output batches fail instead of accumulating. Existing decode
settings are retained. For plain Lance 13 v2 scans, `batch_readahead(1)` alone
does not enforce single-batch decoding: the native path uses the compute-thread
and execution-partition limits. File size uses
the workspace as a soft target. Native decode, manifest metadata, index-building
memory and process RSS are not hard-bounded by these buffer settings. In
particular, this is not a large-scale index-build or recovery-cost qualification.

On an interrupted or unknown result, rerun the original configuration. A
completed projection or index is checked and reused; a completed handoff resolves
through the current managed catalog without reimporting original annotations.
Changed source snapshots, identities, foreign destinations and unexpected native
versions are rejected. Uncommitted files from an interrupted native write may
remain; the command does not erase the destination to retry. Once the normal
exclusive collector is enabled, it uses the usual protection and retirement
protocol. No source body files are eligible for the annotation collector.

Annotation reads must use the owner/ingress path. Existing body-table reads still
contain the original annotation columns and do not automatically join current
annotations. The source body table and its snapshot remain under their existing
reader/retention contract; this command neither maintains nor collects them.
Unified body-plus-annotation reads, online migration, new-body ingestion and
re-import synchronization require subsequent integration. Do not resume the old
annotation writer or route current annotation reads to the source after handoff.

On an uncertain result, repeat the same command/configuration. It checks the
original source transaction, persisted handoff marker, exact output transaction,
and managed bootstrap identity. An interrupted schema commit can be registered
without writing another schema version. A completed handoff, even after later
annotation publications, returns without taking another ownership epoch. A
changed snapshot/identity is rejected. No source WAL is reclaimed by this command.

Use the existing TLS/WAL endpoints, authorized table binding, HTTP workload token,
receipt keys, and storage configuration. Add an `annotations` block:

```json
{
  "annotations": {
    "owner_url": "http://annotation-owner:8080",
    "apply_memory_bytes": 134217728,
    "staging": {
      "retained_bytes": 67108864,
      "retained_entries": 100000,
      "value_bytes": 65536,
      "readers": 128
    }
  }
}
```

This block is an addition to the existing process configuration, not a complete
standalone config. Exactly one table and one virtual shard are supported.
`ingress_memory_bytes`, `apply_memory_bytes`, and `materializer_memory_bytes` are
independent budgets. Publication needs capacity for the full bounded frozen
generation and its checkpoint metadata. `worker.publication` controls maximum
source records/bytes, minimum batch bytes, maximum batch wait, and retained
manifests. Optional `worker.publication.compaction` uses the existing bounded
native small-file rewrite, starting after the first annotation checkpoint. It
preserves the cut and request results. After an annotation publication, it allows
one compaction and starts further rewrites only within the smaller of
`max_batch_wait` and one second while source updates are waiting. An already
retained plan must finish even if its native rewrite exceeds that window; the
window limits starting more work, not the duration of a rewrite. While idle,
compaction can continue draining small files. Its configured workspace must fit
the publication budget. `TryBinaryCopy` may
fall back to re-encoding bodies for multi-file column layouts; this is not a
zero-copy guarantee. Session indexing and remote preparation remain unsupported.

Compaction recovery tests retain or SIGKILL the three local WAL replicas and
annotation applier after the maintenance plan, prepared files, an unknown
manifest-selection reply, and confirmation before retirement. Reopening checks
the original successful and rejected request results, a concurrent score update,
body bytes, ID-index coverage, later publications and file collection. These
small indexed fixtures exercise recovery with retained disks; they do not qualify
large fragmented tables or sustained production throughput. Native annotation
preparations within the catalog's descriptor budget stay inline; larger ones use
a scoped immutable reference to destination-store metadata, capped at 64 MiB of
raw encoded bytes. That cap does not bound decoded engine memory. All voters and
owners must support the reference descriptor before it is emitted; see
`annotation-managed-publication.md` for recovery and collection requirements.

The owner requires the same authorized etcd prefix on every candidate. It claims
the table with the existing durable scope/owner/request tuple. Acquisition reads
the WAL epoch before validating the etcd lease, preventing a paused candidate
from using a successor's epoch. A busy candidate exits without acquiring an
epoch. Lease loss or a runtime/listener error ends the role; a process supervisor
may replace it. This does not automatically retry uncertain user mutations.

The owner address may name a stable Kubernetes Service. Every ingress forwards
to that address and opens neither Lance nor a local applier. Both roles install
the existing workload-token authentication middleware. The configured workload
token authorizes the table; an edge proxy remains responsible for TLS and any
narrower end-user permissions. `GET /api/v1/health` uses the same authentication.
Owner health checks the running runtime and catalog epoch; each point read still
waits for its apply barrier. Frontend health checks local draining state.

Annotation forwarding uses HTTP/2 to the owner, including cleartext HTTP/2 for
an `http://` owner URL. An early admission refusal can cancel one upload stream
without closing a connection while the ingress is still writing its body.
Owners and forwarding ingress listeners continue accepting HTTP/1 clients.
Upgrade owners before forwarding ingresses; any proxy at `owner_url` must support
HTTP/2. There is no automatic HTTP/1 fallback or mutation replay after a transport
failure. Such failures still report an unknown outcome and retain the original
request identity for explicit resolution. Forwarding shares Arrow body frames
without decoding or coalescing the complete upload.

Graceful shutdown closes admission, lets admitted requests finish while the
applier runs, then joins the listener and runtime before releasing the claim.
A crash leaves the claim to expire. A replacement reconciles publication and
replays the required suffix through the existing runtime. Owner observations
report applied/materialized positions, retained staging bytes, active/frozen
entries, source bytes/records, and application pressure once per second.

Each observation includes `observed_unix_ms` and the existing cumulative phase
counters under `phases`. `collect` measures the collector step, `compact` the
maintenance check/rewrite, `prepare` native preparation, `blob_write` descriptor
persistence or a compaction task's file writes, and `commit` managed publication
resolution. The shared catalog also reports `state`, `prepare`, and `retire`
work. This distinguishes a full source window waiting on maintenance from one
waiting on native preparation or publication without logging individual turns.

Collection also reports three nested phases. `collect_list` measures each
tag/branch probe and the bounded data-listing slice, including continuation waits;
its counts are not cloud request counts. `collect_protect` measures retained
manifest reads and removal of live-reader/snapshot references from the candidate
page, including their bounded workspace admission. `collect_delete` measures a
sealed page's concurrent DELETE calls and result
handling, including retries of unresolved paths. Its completion does not mean all
paths were deleted: `reclamation.delete_failures` and `pending_objects` retain
that distinction. These phases leave catalog/history fencing and page-workspace
admission in the parent `collect` phase. They identify which part of collection
delays the serial publisher without adding per-object labels or changing its
ordering.

Counters report starts, completions, failures, cancellations, total elapsed
microseconds and the longest finished interval. Time still in flight is not
included in elapsed/max counters. A completed interval only means that measured
method returned; it is not proof of durable publication. Collection and compaction
checks may do no work, so their counts are not physical delete/rewrite counts.
Parent phases include child phases and observations are sampled independently;
do not sum them into wall/CPU time or treat them as one atomic view. These fixed
counters reset with the owner and are diagnostic only. They do not change
admission limits, publication ordering or failure handling.

The `annotation_io_owner` example also installs a stderr tracing subscriber.
It defaults to warning-level logs and accepts `RUST_LOG`. To observe the object
store SDK's existing retry reasons, attempt counts and backoff delays, run it
with `RUST_LOG=warn,object_store::client::retry=info`. This changes diagnostic
output only; the SDK retry policy and application failure handling stay the same.
These events are not a per-request trace: concurrent operations can interleave.
Transport errors can include object URLs, so sanitize logs before sharing them.

## Physical file collection

For a table served exclusively by its annotation owner, set
`annotations.exclusive_reclamation` to enable the existing bounded collector:

```json
"exclusive_reclamation": {
  "max_objects": 512,
  "max_bytes": 2097152,
  "delete_parallelism": 16,
  "interval": {"secs": 1, "nanos": 0}
}
```

The option is disabled when omitted. The owner protects captured annotation
readers, retained managed manifests and unresolved publications before sealing
each deletion page through the fenced catalog. It collects superseded data,
manifests and transactions between publications. Forwarding ingresses may use the
same configuration; only the owner runs collection.

All reads and writes must use the managed annotation roles. Raw Dataset readers,
other writers, tags, branches and reusing the destination URI for another table
are outside this exclusive-ownership contract. A raw reader's retained version
is not registered in the owner's reader protection.

After managed history has actually retired older versions, the bootstrap anchor
is retained as an adoption receipt: its manifest and native transaction remain,
but it no longer pins initial data that compaction has superseded. Current and
retained snapshots, plus captured annotation readers, still protect every data
file they reference. Before that retirement, bootstrap data remains protected
by the configured history window. Keeping the bootstrap receipt does not provide
permanent time travel to its original rows under exclusive reclamation.

`max_objects` and `max_bytes` bound each candidate page; `delete_parallelism`
bounds concurrent deletes. The interval starts after a collection pass finishes.
Page workspace must fit `materializer_memory_bytes`; invalid limits are rejected
before ownership acquisition. Owner observations include process-local
`reclamation` counters for passes, examined objects and completed deletes. These
counters are not a current object-store inventory.

The application process test uses real RF3 RPCs, etcd, Lance storage, two ingress
executables and an owner executable. It starts by handing off a full rollout
table with existing scores and body columns. It checks cross-ingress reads, publication
and source GC, exact retries, competing owner exclusion, SIGKILL plus natural
lease expiry, replacement behind the same address, field clearing, and lease-loss
shutdown. It also gives ingress processes an unusable Lance URI to verify that
forwarding does not open storage. It is a correctness test with a small fixture;
it does not qualify sustained throughput or large-table cold lookup performance.
With collection enabled, it also checks invalid configuration before acquisition,
physical file draining across owner replacement, exact retries after collection,
and retained body contents after further updates through both ingresses.

Refs #378.
