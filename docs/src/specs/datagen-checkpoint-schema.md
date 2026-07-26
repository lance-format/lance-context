# Datagen Checkpoint Log

## Decision

Each datagen experiment uses one Lance dataset:

```text
<durable-root>/datagen_checkpoints/<pipeline>/<experiment>/log.lance
```

The log is the only durable source of truth. Current state, failures, resume
cursors, and trajectories are all derived by folding events. This avoids
cross-dataset consistency hazards because Lance does not provide atomic
transactions across multiple datasets.

Optional query projections may be added later, but they must be disposable,
rebuildable, and excluded from checkpoint correctness.

## Event model

Every row is an immutable event:

- `ITEM_CREATED`
- `FIELD_SET`
- `FIELD_APPEND`
- `STEP_COMPLETED`
- `FAILED`
- `TERMINAL`

Every completed step emits a `STEP_COMPLETED` event, even when no field changed.
All field events and the completion marker for one step are written in the same
checkpoint batch.

`event_id` is a deterministic idempotency key. Retrying an ambiguously
acknowledged batch writes the same event ids, and the MemWAL LSM read path
de-duplicates them. `item_seq` is strictly increasing per item; two different
events at the same sequence are treated as a writer-fencing violation.

## Schema

| Column | Type | Purpose |
|---|---|---|
| `event_id` | string | Deterministic event identity and LSM primary key |
| `item_id` | string | Scoped item identity |
| `root_item_id` | string | Root of the projected item tree |
| `parent_item_id` | string? | Direct parent item |
| `item_seq` | int64 | Per-item event ordering |
| `checkpoint_id` | string | Atomic step-boundary identity |
| `event_type` | string | Event kind |
| `step_name` | string? | Step provenance |
| `step_index` | int64? | Static step position |
| `step_instance_id` | string? | Runtime step identity |
| `iteration` | int64? | Loop/branch iteration |
| `attempt` | int32 | Execution attempt |
| `run_id` | string | Run attribution |
| `writer_epoch` | string | Item ownership/fencing identity |
| `field_name` | string? | Changed field |
| `field_type` | string? | Stable codec id |
| `codec_version` | int32? | Codec compatibility version |
| `value_kind` | string? | `int`, `float`, `bool`, `str`, `json`, or `blob` |
| `value_i64` | int64? | Exact integer value |
| `value_f64` | float64? | Floating-point value |
| `value_bool` | bool? | Boolean value |
| `value_str` | large_string? | String value |
| `value_json` | large_string? | Canonical JSON value |
| `value_blob` | large_binary? | Sparse inline bytes |
| `payload_size` | int64? | Blob size |
| `payload_checksum` | string? | Blob integrity |
| `query_tags_json` | large_string? | Non-authoritative query tags |
| `terminal` | string? | `completed` or `filtered` |
| `error_type` | string? | Failure type |
| `error_dump` | large_string? | Serialized failure |
| `traceback` | large_string? | Failure traceback |
| `event_ts` | timestamp(us, UTC) | Event time |
| `schema_version` | int32 | Log schema compatibility version |

`value_blob` remains inline while MemWAL's LSM scanner cannot materialize
blob-v2 columns. Normal fold and trajectory reads project it out. Blob access
first locates `event_id` using lightweight columns and then calls `take_rows`
for the exact `_rowid`.

## Write and recovery invariants

1. A checkpoint batch is one durable MemWAL generation.
2. Log writes are append-only; retries reuse deterministic event ids.
3. One live owner writes a given item at a time. Ownership changes require a
   new `writer_epoch`.
4. Resume reads `item_id = X`, orders by `item_seq`, and folds the events.
5. `FIELD_SET` replaces a field; `FIELD_APPEND` accumulates it.
6. `STEP_COMPLETED` reconstructs the resume cursor.
7. `FAILED` and `TERMINAL` are ordinary log events, not separate datasets.
8. Blob bytes are loaded only when the corresponding lazy reference is used.

## Maintenance

Writers use distinct MemWAL shards. Reads union the base table and all flushed
shards, so any instance sees every writer's events. Each writer periodically
merges only its own generations into the base table. Shared base-table
compaction and index refresh must be scheduled by one elected maintenance
worker per experiment.

