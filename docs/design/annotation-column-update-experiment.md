# Native annotation column update experiment

Refs [#378](https://github.com/lance-format/lance-context/issues/378).
Related staging protocol: [#564](https://github.com/lance-format/lance-context/pull/564).

Updating a small annotation beside a large immutable body must not implicitly
rewrite that body. Lance 13 exposes `MergeInsertWriteMode::RewriteColumns`; this
experiment compares it with `Auto` and `RewriteRows` using the same partial
`id/status/score` source and explicit uncommitted preparation followed by commit.
It uses the existing Lance API without modifying Lance OSS.

Run each case into a fresh local directory:

```bash
cargo build --release --locked -p lance-context-core \
  --features lance-io-metrics --example annotation_column_update
target/release/examples/annotation_column_update \
  --output /tmp/annotations-dense --rows 64 --body-bytes 1048576 --rounds 4 --stride 1
target/release/examples/annotation_column_update \
  --output /tmp/annotations-sparse --rows 64 --body-bytes 1048576 --rounds 4 --stride 32
```

The synthetic fixture has 64 deterministic nonconstant 1 MiB bodies. Four
rounds assign and clear scores, including returning to the same value. Every
round verifies all body bytes, one current row per key, unchanged visibility
before commit, old captured snapshots after commit, and current values after
reopen. The column-rewrite arm additionally requires identical original body
file paths. Verification I/O is outside prepare/commit measurements.

Observed ranges per round in these two local runs:

| Updated rows | Mode | New data bytes | Prepare requested read bytes | New body files |
| --- | --- | ---: | ---: | ---: |
| 64 | Auto / RewriteRows | 67,113,598–67,113,604 | 67,116,812–67,129,352 | 1 |
| 64 | RewriteColumns | 3,052–3,076 | 3,052–16,840 | 0 |
| 2 | Auto / RewriteRows | 2,098,231–2,098,237 | 67,116,616–67,129,352 | 1 |
| 2 | RewriteColumns | 1,950–2,110 | 4,124–17,110 | 0 |

These results support explicitly choosing column rewriting for narrow updates:
the default strategy rewrites bodies even with a partial source. Requested
storage bytes include metadata and may benefit from warm caches; they are not
physical disk/network traffic. The sequential local runs do not establish
distributed throughput, p99, WAL durability, staging recovery or zero-copy.

The fixture deliberately uses 16 rows per original file to expose file behavior.
Column rewriting writes four narrow files per dense round and two per sparse
round. These tiny outputs are not production batch-size recommendations.
Sustained batch sizing, changing hot-key sets, compaction cost and current-turn
query cost remain necessary before finalizing the physical annotation layout.
In particular, this experiment does not prove that ordinary full-row compaction
will preserve the same body I/O savings.
