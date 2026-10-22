# Local entry-cache isolation correction

This is the published `raft-engine` 0.4.2 source, whose upstream revision is
`084c5ca0e4f8291ab67afbfe8b9df02747838c86`. Its Apache-2.0 license, original
manifest, source, tests and notices are retained. The registry's `.cargo-ok`
installation marker is omitted. The generated Cargo manifest is unchanged.
The registry archive checksum is
`1213c3a24e3fee8afcc74b2be08c4081adde96f092e0fc1c607abb3e16ae722e`.

The entry-cache correction changes `src/engine.rs` and `src/purge.rs`. The accompanying
`read-cache-identity.patch` records their complete delta against the published
source. This correction belongs to the ingestion work tracked in
[lance-context #378](https://github.com/lance-format/lance-context/issues/378).

The upstream thread-local entry cache identifies a block by queue, file sequence,
offset and length. Two independent engines can have identical block handles but
different bytes. The cached bytes can then escape through scalar or batch reads;
physical rewrite also uses this cache and can persist the wrong payload with a
valid checksum. Dropping the first engine does not invalidate the thread cache.

Each engine's purge manager now owns a process-local, monotonically allocated
cache identity. Normal reads and rewrite reads include that identity in the
cache lookup. The identity cannot wrap and is not a pointer or pathname, so
closing an engine and reusing its allocation or directory cannot alias its cache.
Allocation happens once per engine lifetime; cache hits add an integer comparison,
without a new allocation or atomic operation. Disk and wire formats are unchanged.
No throughput improvement is claimed by this correctness change.

The context WAL crate uses a direct path dependency, and is currently unpublished
(`publish = false`). A workspace-root patch alone would not carry this correction
to downstream users of a Git dependency. Publishing a crate that depends on this
local source will require a separately resolved packaging plan; it must not
silently fall back to the uncorrected registry version.

Run the actual-file regressions with:

```sh
cargo test --locked -p lance-context-wal --features lab --test engine_cache
```

They check scalar reads, batch reads, a closed/replaced engine, and physical
rewrite followed by fresh-thread readback after reopening. The standalone
qualification of these same four cases returned wrong payloads with the original
dependency and passed with this source correction. Application and full workspace
checks remain separate requirements; this is not a production WAL enablement.
# Explicit directory-lock release

Engine shutdown now explicitly unlocks its directory before closing the owner's
lock descriptor. On Unix, a duplicated descriptor (including one inherited by a
concurrent child before exec) shares the same open file description and can keep
`flock` held after the engine has closed. The guard records its acquiring process
ID so a forked child's destructor cannot unlock the parent's live engine. The
guard is still released only after the engine's pipe owners finish dropping.

`directory-lock-release.patch` records this additional vendor change. A Linux
client-journal regression duplicates the actual engine lock descriptor, verifies
exclusion while the owner is live, closes the journal, and reopens it while the
duplicate remains open. Closing that old duplicate must not unlock the new owner.
The regression fails before the change and passes after it without retry/sleep.

```sh
cargo test -p lance-context-wal --features client-journal --lib
```

# Reuse uncompressed read buffers

The scalar, batched, and rewrite entry readers now move their freshly read block
into the decoder. An uncompressed block retains that allocation after the same
complete block checksum succeeds; only the trailing checksum is truncated. The
LZ4 path retains its existing decompression and checksum checks. Cache identity,
entry parsing, file formats, persistence, and reclamation rules are unchanged.
This removes one block-sized allocation and copy on an uncompressed cache miss;
it does not remove Protobuf field copies or make the ingestion path zero-copy.

`owned-block-read.patch` records this additional vendor delta. The actual-file
`engine_cache` suite also exercises both compression modes after reopen, all
entry fields, corruption on a cache miss, and successful reads after restoring
the original byte. Separately recorded Rust diagnostics measure the real log
read path and allocator-requested bytes; they are not end-to-end throughput or
RSS benchmarks.
