# Using the Datagen Checkpoint Store

`DatagenStore` is the durable checkpoint backend for a datagen experiment. It is
a single append-only Lance log: every write is one immutable event, and the
current state of any item is the *fold* of its events. Nothing is ever updated or
deleted. This document shows how a client (the `mai_datagen` executor, through
its pyo3 bindings) drives the store through a run — create, checkpoint, resume,
read — with runnable Rust snippets.

For the schema and the design rationale, see
[`specs/datagen-checkpoint-schema.md`](../../specs/datagen-checkpoint-schema.md).

## The model in one paragraph

An experiment is one `log.lance`. Each row is a `DatagenEvent` with an
`event_type`. An **item** is one stream of events sharing an `item_id`; a source
item is a **root**, and fan-out steps project **sub-items**, each its own stream.
To learn an item's state you read its events and call `fold_datagen_events`,
which replays them into a `FoldedDatagenItem` (fields, status, trajectory).
Because reconstruction is a pure fold over an append-only log, a crash can never
leave a torn write: an unacknowledged batch either persisted whole or not at all.

## Opening a store

```rust
use lance_context_core::{DatagenStore, DatagenStoreOptions};

// One writer per shard. Concurrent writers pass distinct shard ids.
let mut store = DatagenStore::open("s3://bucket/exp/log.lance").await?;

// Or with an explicit shard id (each writer owns its own MemWAL shard):
let mut store = DatagenStore::open_with_options(
    "s3://bucket/exp/log.lance",
    DatagenStoreOptions { shard_id: Some("worker-3".into()), ..Default::default() },
).await?;
```

## Item identity — composed on the client, no round-trip

Ids are structured values (`DatagenItemId`) stored as materialized path strings.
The client composes them purely; the store never allocates an id.

```rust
use lance_context_core::DatagenItemId;

let root = DatagenItemId::from_source_key("5");      // "5"
let child = root.child("solve_twice", 0);            // "5/solve_twice:0"
let grandchild = child.child("judge", 1);            // "5/solve_twice:0/judge:1"

assert_eq!(grandchild.origin_step(), Some("judge"));
assert_eq!(grandchild.branch_idx(), Some(1));
assert_eq!(grandchild.parent().unwrap(), child);
assert_eq!(grandchild.root(), root);
```

Because `root_item_id` and `parent_item_id` are denormalized onto every row,
reading a whole item tree is one filter (`events_for_root`), never a join.

## Writing events

An event is built with its provenance columns and appended. Field events plus
their `STEP_COMPLETED` marker for one step must go in a **single** call so a crash
cannot expose a half-checkpointed step — the batch is one durable generation.

```rust
use lance_context_core::{
    datagen_event_id, DatagenEvent, DatagenEventType, DatagenItemStatus,
    DatagenStepKind, DatagenValue, DATAGEN_SCHEMA_VERSION,
};

// 1. Announce the item once.
let created = DatagenEvent {
    event_id: datagen_event_id("5", "created", 0),
    item_id: "5".into(),
    root_item_id: "5".into(),
    parent_item_id: None,
    item_seq: 0,
    checkpoint_id: "created".into(),
    event_type: DatagenEventType::ItemCreated,
    status: Some(DatagenItemStatus::Running),
    schema_version: DATAGEN_SCHEMA_VERSION,
    ..blank_event()   // your helper that zero-fills the optional columns
};
store.append(&[created]).await?;

// 2. Checkpoint one step: its field delta + exactly one STEP_COMPLETED, atomically.
let score = DatagenEvent {
    event_id: datagen_event_id("5", "solve-0", 0),
    item_id: "5".into(),
    root_item_id: "5".into(),
    item_seq: 1,
    checkpoint_id: "solve-0".into(),
    event_type: DatagenEventType::FieldSet,
    step_name: Some("solve".into()),
    step_kind: Some(DatagenStepKind::Leaf),
    step_index: Some(0),
    enclosing_step: Some("solve_attempt".into()),
    field_name: Some("score".into()),
    field_type: Some("int".into()),
    codec_version: Some(1),
    value: Some(DatagenValue::Int(9)),
    schema_version: DATAGEN_SCHEMA_VERSION,
    ..blank_event()
};
let completed = DatagenEvent {
    event_id: datagen_event_id("5", "solve-0", 1),
    item_seq: 2,
    checkpoint_id: "solve-0".into(),
    event_type: DatagenEventType::StepCompleted,
    // same step_name / step_kind / step_index / enclosing_step as above
    ..score.clone()
};
store.append_checkpoint(&[score, completed]).await?;
```

`append_checkpoint` enforces "exactly one `STEP_COMPLETED` per batch";
`append` is the lower-level form used for lifecycle events (`ITEM_CREATED`,
`TERMINAL`, `FAILED`).

### Retries are safe

`event_id` is deterministic (`datagen_event_id(item_id, checkpoint_id, ordinal)`).
Replaying an ambiguously-acknowledged batch writes the same ids, and the fold
de-duplicates them — the second `append_checkpoint` of an identical batch is a
no-op in the folded result.

### Kind determines what you write

| Step kind | Emits |
|---|---|
| `Sequence`, `Loop` (drivers) | `STEP_STARTED` frame + `STEP_COMPLETED` |
| `MapReduce`, `Branch`, `SubPipeline` (fan-out) | only the reduce `STEP_COMPLETED`; sub-items are their own streams |
| `Conditional`, `Router` (selectors) | no row of their own — the chosen child sets `selector_step` |
| `Leaf` | `FIELD_SET` / `FIELD_APPEND` + `STEP_COMPLETED` |

## Finishing an item

```rust
// Success or filtered-out — a lifecycle terminal.
let terminal = DatagenEvent {
    event_type: DatagenEventType::Terminal,
    status: Some(DatagenItemStatus::Completed),   // or Filtered
    ..lifecycle_event("5", 3, "terminal")
};
store.append(&[terminal]).await?;

// A raised step — a failure-lens row. The item still folds to `running`.
let failed = DatagenEvent {
    event_type: DatagenEventType::Failed,
    status: Some(DatagenItemStatus::Failed),
    step_name: Some("score".into()),
    step_kind: Some(DatagenStepKind::Leaf),
    step_index: Some(1),
    error_type: Some("ValueError".into()),
    ..lifecycle_event("5", 3, "failed")
};
store.append(&[failed]).await?;
```

## Reading

### Fold one item

```rust
use lance_context_core::DatagenItemLookup;

match store.fold_item("5").await? {
    DatagenItemLookup::NeverStarted => { /* fresh: process from scratch */ }
    DatagenItemLookup::Found(item) => {
        // item.status, item.fields, item.trajectory, item.last_item_seq, item.last_attempt
        // Resume: continue writing at last_item_seq + 1, attempt last_attempt + 1.
    }
}
```

`NeverStarted` (no `ITEM_CREATED`) is the explicit fresh-vs-resume fork. A
`Found` item's `status` is always a *lifecycle* value (`running` / `completed`
/ `filtered`) — `failed` never appears here.

### Resume: the open frame

The trajectory records both started and completed positions. `started \ completed`
is the driver frame that was open when the process died — everything already
completed is skipped on re-run.

```rust
let item = store.fold_item("5").await?.folded().unwrap();
let open: Vec<_> = item.trajectory.started
    .difference(&item.trajectory.completed)
    .collect();   // the frame(s) to re-enter
```

### Whole item tree

```rust
// Every event under root "5", including all fan-out sub-items — one filter.
let events = store.events_for_root("5").await?;
```

### Bulk startup classification

```rust
// Classify many roots at once without folding fields (skip / resume / fresh).
let statuses = store.root_item_statuses(&["5", "6", "7"]).await?;
assert!(statuses.is_terminated(&DatagenItemId::from_source_key("5")));
```

### Failures (the failure lens)

```rust
let failures = store.item_failures("5").await?;   // 0..N, across attempts
for failure in &failures {
    println!("{} at {}", failure.error.error_type, failure.at.position.step.name);
}
let all = store.failures(Some("run-1")).await?;   // run-wide forensics
```

### Blobs are lazy

Field values that are blobs fold to a lazy `DatagenBlobValue { bytes: None, .. }`.
Materialize the bytes only when needed:

```rust
let bytes = store.get_blob(&blob_event_id).await?;   // O(single blob) take_rows
```

## Concurrency and maintenance

- One live owner writes a given item at a time; a new owner takes over with a new
  `writer_epoch` (fencing).
- Concurrent writers use distinct MemWAL shards. Reads union the base table and
  all flushed shards, so every instance sees every writer's events.
- Each writer periodically merges only its own generations into the base table
  (`cleanup_own_shard` / `spawn_periodic_cleanup`). Shared base-table compaction
  is scheduled by one elected maintenance worker per experiment.
