# WAL component experiments

Feature-gated Rust measurements for choosing a WAL payload path and durable
local log engine. This crate does not change production ingestion or Lance OSS.

Implementation and validation are tracked in
[#378](https://github.com/lance-format/lance-context/issues/378). Related PRs should
include `Refs #378`; individual steps should leave the tracking issue open.

The optional `client-journal` feature stores recoverable client state in one
local engine and shares durable barriers across writers. It returns latest
values for a fixed set of keys, with immutable scope/settings binding, bounded
admission, reclamation and a shared failure fence. Callers still own producer
ordering, exact retry identity and recovery validation; a local completion does
not establish a remote quorum ACK. See [the API and recovery contract](../../docs/design/client-journal.md).

```bash
cargo test -p lance-context-wal --features client-journal --lib client_journal
```

The independent `contract` feature defines ingestion routing keys, the schema
fingerprint, immutable-ID policy and record byte/row limits. Validation checks
the representation after decoding; callers must durably bind and compare the
complete contract for each table generation. The ingress still validates actual
Arrow input, and the materializer must support the registered maximum record.
This component does not enforce persistence, authentication or ID uniqueness.
Schema hashing accepts already canonical bytes; it does not normalize JSON or
parse Arrow. The module requires neither Arrow nor a replication/log engine.

```bash
cargo test -p lance-context-wal --features contract --lib
```

The optional `arrow-contract` feature adds
`ingest_contract::arrow_schema_fingerprint(&Schema)` for ingress and materializers.
It recursively sorts JSON object keys, including nested metadata, and preserves
field order, types, nullability and serialized dictionary attributes. Calculate
it when binding a table generation, then retain the fingerprint on replicas.
Only Arrow schema metadata is required; this feature does not decode IPC or load
a replication engine. Arrow serialization changes require compatibility review
before upgrading writers of existing contracts.

```bash
cargo test -p lance-context-wal --features arrow-contract --lib
```

The optional `record` feature provides opaque `Batch`/`BatchMeta` envelopes with
shared `Bytes` payload ownership. `BatchMeta::check_body_length` validates frame
lengths and a caller-supplied data/control limit before payload allocation;
`Batch::validate` also checks the complete SHA-256 digest. Construction and cloning
retain the input allocation. Digest computation scans the payload; this is not
end-to-end zero-copy. Replication can use this feature without Arrow or a log engine.

Routing, producer sequencing/fencing, control commands and durable commit still
require separate validation. The record domain tags reserve the single-group,
multi-group and source-control formats used by the ingestion experiments. This
PR adds the envelope, not their state machines. Serialize metadata separately
from bulk bytes in network paths; `Debug` excludes the body.

```bash
cargo test -p lance-context-wal --features record --lib
```

The optional `ipc` feature exposes `ipc::decode(Bytes, expected_schema)` for
materializers. It decodes multiple uncompressed record batches and dictionaries
while sharing binary payload ownership with the input. Numeric alignment repair,
schema parsing and Arrow metadata still allocate. The codec validates footer and
block/array-buffer ranges, native endianness, schema equality, row/count limits
and unsupported expansion before constructing arrays. Compressed batches and
dictionary deltas are rejected.
Callers still reserve input and decoding workspace within their own memory budget.
Replication does not need to enable this module or decode Arrow. The `lab` codec
measurement now uses the same validated path, including its validation overhead.
Arrow dependency panics from malformed metadata become errors in panic-unwind
builds; panic-abort builds cannot recover from dependency panics.

```bash
cargo test -p lance-context-wal --features ipc --lib
```

The independent `placement` feature provides `placement::Layout`, session bucket
hashing and scoped `Receipt` types. A validated layout assigns each stable virtual
bucket to one RF3 group, with explicit voters and StatefulSet DNS addresses. Use a
canonical immutable tenant/table identity in the routing key, not a mutable display
name. Worker count does not participate in routing. Persist the layout and its
fingerprint; changing it requires a migration protocol, which is not implemented
here. A receipt identifies a position and is not authentication or proof of commit.

```bash
cargo test -p lance-context-wal --features placement --lib
```

Compatibility tests preserve a synthetic layout fingerprint and a fixed
session-hash vector. Other tests reject ambiguous ownership
and ensure routes survive JSON serialization and restart.

The independent `admission` feature provides one proposal budget shared by all
groups on a node. Data proposals leave 25% of the configured budget (up to
256 MiB) for control traffic. Reservations release on drop; encoded engine-write
counters account for peer, control and rewrite traffic between directory scans.
Refresh measures file lengths and allocated blocks, rather than estimating
segment sizes. Successful admission does not walk the directory.

`Budget::limit_data` can lower the shared producer ceiling to leave more space
for checkpoint copies and log rewrites. It never raises the total budget or
cancels outstanding reservations. Maintenance callers use the total-budget
reservation path; using producer admission for checkpoints can prevent recovery
of space precisely when producers are throttled. The default 25% control window
alone does not guarantee enough temporary space for full-copy maintenance.

This is proposal admission, not filesystem space reservation or a limit on
already accepted replication traffic. Callers must supply a monotonic write
counter covering every group and protect file lifecycle changes with the shared
lock. Snapshot collection and replica-pressure coordination are still required
for sustained operation. No production writer is connected to this module yet.

```bash
cargo test -p lance-context-wal --features admission --lib
```

```bash
cargo build --release -p lance-context-wal --features lab --bin context-wal-lab
target/release/context-wal-lab codec --payload-bytes 262144 --rows 8 --iterations 30
target/release/context-wal-lab disk --directory /tmp/wal-disk-new \
  --payload-bytes 262144 --batch 4 --writers 4 --groups-per-writer 4 --rounds 64
```

`codec` measures encoding, Arrow FileReader, shared-buffer FileDecoder, an
unaligned shared buffer, and decoding followed by per-row payload copies. All
paths use the same IPC bytes; data generation and correctness checks are outside
timing. JSON reports time, allocation count, allocated bytes and logical IPC
MiB/s. Shared-buffer decoding may inspect only metadata and retain payload
buffers: logical IPC MiB/s is not memory-copy bandwidth or a full payload scan.
Unaligned numeric buffers may need repair copies. The allocator counts allocation
requests, including reallocations, rather than peak or retained memory.

`disk` rotates each writer over separate Raft groups sharing one real raft-engine.
Every nonempty write uses `sync=true`; JSON includes batch latency, payload
throughput, engine bytes, reopen time and separate verification time. It reopens
the engine and compares every persisted payload. Inputs are deterministic bytes,
prepared before timing and reused across rounds. Compression is disabled unless
`--compression-threshold` is supplied. The directory must be new and remains for
inspection. This measures local durable writes, not replicated ACKs, crashes,
power loss or Blob/Lance publication.

Disk footprint is observed after the writers finish and the engine closes,
outside write and reopen timing. `disk_apparent_bytes` sums regular file lengths;
`disk_allocated_bytes` sums their allocated blocks on Unix and is null elsewhere.
`disk_bytes` conservatively sums the larger apparent/allocated size of each file
(apparent size on other platforms). These are directory footprint observations,
not physical bytes written or write amplification. The pinned engine's segment
estimate is reported separately as `engine_estimated_disk_bytes`: large records
can overshoot the target segment size and make that estimate too small.

The pinned raft-engine 0.4.2 has a separate `sync-probe` command to inspect empty
sync versus a nonempty synced barrier:

```bash
strace -f -e trace=fsync,fdatasync target/release/context-wal-lab \
  sync-probe --directory /tmp/wal-sync-new
```

Use the printed markers to associate syscalls with each phase. Do not infer a
durability boundary from an empty batch without checking the engine's behavior.
For comparisons, preserve the source commit, binary hash, release/debug mode,
command, filesystem/storage placement and JSON. CI runs tiny correctness smoke
cases; their numbers are not performance evidence.

## Durable Raft log storage

The replication layer also provides [durable publisher bindings and epochs](../../docs/design/wal-publisher-authority.md)
through its catalog group's normal consensus path. This is an incremental
publication prerequisite, not a production Lance writer or a manifest lease.

[Quorum application queries](../../docs/design/wal-quorum-queries.md) capture bounded
catalog, producer-tail and accepted-page views after a safe read-index barrier.
They share pending capacity with proposals and do not fetch bulk payloads.

[Producer slots](../../docs/design/wal-producer-slots.md) provide opt-in bounded
writer state on fresh groups. Durable generations fence old requests after a
closed lane's published source has been retired and its slot reused.

[Source closure proofs](../../docs/design/wal-source-closure-proofs.md) attest that
an exact catalog-authorized closure committed at its source. Optional per-group
keys keep catalog private keys off source-only replicas; catalog checkpoint
advancement and physical reclamation remain separate steps.

[Immutable publication plans](../../docs/design/wal-publication-plans.md) retain
the exact source vector and Lance target through takeover, snapshots and restart.
They share bounded metadata between views. [Source pins](../../docs/design/wal-source-pins.md)
verify signed catalog quorum observations against complete source prefixes and
survive snapshots and restart. Outcome resolution and retirement remain separate
integration steps; neither a plan nor a pin permits an external Lance commit.

The optional `raft-log` feature exposes `raft_log::PhysicalLog` and its cloneable
`GroupLog` handles. One raft-engine directory hosts all groups assigned to a node
by the existing immutable `placement::Layout`. Opening binds the full layout,
node identity and each group's three voters before returning a handle. A changed
identity/layout, a foreign nonempty engine or inconsistent durable metadata is
rejected. The directory format is separate from earlier experiment binaries.

`GroupLog` implements raft-rs `Storage`. `persist(entries, hard_state)` puts the
entries, term ranges and optional term/vote/commit state in one engine batch,
then waits for a physical sync barrier. Different groups can append concurrently
and share a later barrier. The barrier is a nonempty engine write: the pinned
engine's empty `sync()` call does not sync. Syncing is isolated from concurrent
native engine writes because that engine can panic on sync failure. A write
error or an unwinding write panic invalidates every handle on the physical node;
stop its driver and reopen all handles to resolve the uncertain durable state.
Validation errors before engine submission leave the log usable.

With `runtime`, each group's cached `durable_poll.storage_write` reports completed
append batches, encoded append bytes, explicit sync-marker barriers and marker
bytes on its worker. Compare interval deltas summed across group workers to
observe batch size and shared-barrier amortization. Counters exclude work on
other threads, including background maintenance, and retain the last successful
poll's sample on failure. They are not client QPS or filesystem sync-call counts;
native file rotation can perform additional syncs.

Uncommitted suffixes may be replaced. Committed entries, votes in the same term,
and monotonic term/commit state are protected. Term lookup uses small cached
ranges rather than reading and decoding entry bodies. Range reads request at
most 256 engine entry descriptors at a time instead of allocating descriptors
for an entire history before applying the byte limit. Returned entries obey the
Raft byte limit, including its required first-entry exception. At a chunk boundary,
one extra entry may be decoded before determining that it does not fit; callers
still need an explicit maximum record size and memory admission. This storage
layer does not decode Arrow, and it does not provide end-to-end zero-copy I/O.

A successful local `persist` is not a quorum ACK. The subsequent Ready driver
must order durable writes, outgoing messages, commits and state-machine apply
correctly. This increment supports static membership and retains the complete
log; snapshots, compaction, replacement, transport, producer deduplication and
Lance publication are separate integration steps tracked by
[#378](https://github.com/lance-format/lance-context/issues/378). It does not switch
production ingestion. Storage assumes exclusive ownership of the directory and
that successful filesystem syncs survive the supported failures.

```bash
cargo test -p lance-context-wal --features raft-log --lib --test raft_log --test raft_log_cache_identity
cargo clippy -p lance-context-wal --features raft-log --all-targets -- -D warnings
```

Tests use actual raft-engine files for shared-group recovery, conflict replacement,
vote/commit protection and byte-limited reads. Independent-log cases verify cache
isolation for simultaneous engines, dropped handles and a recreated directory;
they require the vendored engine correction described in
`vendor/raft-engine/LANCE_CONTEXT_PATCH.md`. Linux subprocess cases exercise
SIGKILL after a synced prefix (including segment rotation) and a real write
failure imposed only on the child process. Unit tests use a test-only filesystem
wrapper around real files to count sync calls and inject a blocked failing sync
while another group writes, both at the explicit barrier and during rotation.
That wrapper is absent from ordinary builds. A RawNode test drives persistence,
elections, minority isolation and restart with deterministic in-process message
delivery. These tests establish component contracts; they do not establish
power-loss recovery or distributed quorum durability.
