# Lance 13 BTree compatibility fixture

The `lance13-btree` directory was written by the real Rust Lance 13.0.0
index builder, using `lance13-btree-generator.rs`. Both native version-0 indexes
cover fragment 0; fragments 1 and 2 provide an uncovered tail. Six rows include
nullable status values. No Lance 14 pages were relabeled as version 0.

Generator build: lance-context commit e75e18705a24534ee466cf4ad54089d71f451173
with its original Cargo.lock and the adjacent generator installed as a core
example. Run it against a fresh output directory with Rust 1.99.0:

```sh
cargo run --locked -p lance-context-core --example legacy_btree_fixture -- /tmp/native13.lance
```

The report is in `lance13-btree.json`. Compatibility tests copy the fixture to
a temporary directory before appending, extending and merging indexes,
compacting, and reopening it. The committed fixture remains immutable.
