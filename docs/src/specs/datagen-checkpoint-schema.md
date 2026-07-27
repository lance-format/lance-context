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

Every row is an immutable event. The current schema version is
`DATAGEN_SCHEMA_VERSION = 2`. There are seven event types:

- `ITEM_CREATED` — an item (stream) first appears, exactly once, `status = running`.
- `STEP_STARTED` — a driver frame (`Sequence`/`Loop`) opened. A structural marker.
- `FIELD_SET` — replaces a field's value (fold: last-writer-wins).
- `FIELD_APPEND` — accumulates onto a field (fold: append in order).
- `STEP_COMPLETED` — a checkpointed step boundary, carrying that step's field delta.
- `FAILED` — a step raised; a failure-lens row that leaves the item `running`.
- `TERMINAL` — the item finished, `status = completed | filtered`.

Every completed step emits a `STEP_COMPLETED` event, even when no field changed.
All field events and the completion marker for one step are written in the same
checkpoint batch. Only driver kinds emit `STEP_STARTED`; fan-out kinds
(`MapReduce`/`Branch`/`SubPipeline`) emit only their reduce `STEP_COMPLETED`;
selector kinds (`Conditional`/`Router`) emit no row of their own — the chosen
child records the selector in `selector_step`.

`event_id` is a deterministic idempotency key. Retrying an ambiguously
acknowledged batch writes the same event ids, and the MemWAL LSM read path
de-duplicates them. `item_seq` is strictly increasing per item; two different
events at the same sequence are treated as a writer-fencing violation.

### Item identity

An item id is a materialized path string owned by the store. A root id is the
executor's source key (`5`); a fan-out sub-item extends its parent with one
`step:idx` segment (`5/solve_twice:0`). Ids compose purely on the client, with
no store round-trip. `root_item_id` / `parent_item_id` are denormalized on every
row, so any subtree is a single filter with no join.

### Read lenses

State is read under two lenses:

- *lifecycle* — fold the events; `status` is `running` until a `TERMINAL`. Drives
  skip/resume/fresh classification and ignores `FAILED`.
- *failure* — read `FAILED` rows directly (forensics). An item may have 0..N
  failures across attempts while still folding to `running` under lifecycle.

## Schema

| Column | Type | Purpose |
|---|---|---|
| `event_id` | string | Deterministic event identity and LSM primary key |
| `item_id` | string | Scoped item identity (materialized path) |
| `root_item_id` | string | Root of the projected item tree (denormalized) |
| `parent_item_id` | string? | Direct parent item (denormalized) |
| `item_seq` | int64 | Per-item event ordering |
| `checkpoint_id` | string | Atomic step-boundary identity |
| `event_type` | string | Event kind (one of the seven above) |
| `step_name` | string? | Step provenance (globally-unique step name) |
| `step_kind` | string? | Composition kind (`sequence`, `loop`, `map_reduce`, `branch`, `sub_pipeline`, `conditional`, `router`, `leaf`, `root`) |
| `step_index` | int64? | Static step position (loop iteration = driver frame's index) |
| `enclosing_step` | string? | Name of the `Sequence`/`Loop` driver frame this step ran under |
| `selector_step` | string? | Name of the `Conditional`/`Router` that chose this step |
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
| `status` | string? | `running` / `completed` / `filtered` / `failed` (on ITEM_CREATED / TERMINAL / FAILED) |
| `error_type` | string? | Failure type |
| `error_dump` | large_string? | Serialized failure |
| `traceback` | large_string? | Failure traceback |
| `event_ts` | timestamp(us, UTC) | Event time |
| `schema_version` | int32 | Log schema compatibility version |

The `status` column replaces the earlier `terminal` column: it carries a stored
`running` (from `ITEM_CREATED`) rather than deriving it, plus `failed` for the
failure lens. Structured `step_kind` / `enclosing_step` / `selector_step` replace
the earlier opaque `step_instance_id` + `iteration` provenance, so a resume can
rebuild step coordinates instead of trusting a stored cursor blob.

`value_blob` remains inline while MemWAL's LSM scanner cannot materialize
blob-v2 columns. Normal fold and trajectory reads project it out. Blob access
first locates `event_id` using lightweight columns and then calls `take_rows`
for the exact `_rowid`.

## Write and recovery invariants

1. A checkpoint batch is one durable MemWAL generation.
2. Log writes are append-only; retries reuse deterministic event ids.
3. One live owner writes a given item at a time. Ownership changes require a
   new `writer_epoch`.
4. Resume reads `item_id = X`, orders by `item_seq`, and folds the events. An
   item with no `ITEM_CREATED` folds to `NeverStarted` (fresh vs. restore fork).
5. `FIELD_SET` replaces a field; `FIELD_APPEND` accumulates it. Mixing the two
   on one field is rejected.
6. `STEP_COMPLETED` reconstructs the resume cursor; `STEP_STARTED \ STEP_COMPLETED`
   (started minus completed) is the driver frame open at crash time.
7. `FAILED` and `TERMINAL` are ordinary log events, not separate datasets.
8. Blob bytes are loaded only when the corresponding lazy reference is used.

## Maintenance

Writers use distinct MemWAL shards. Reads union the base table and all flushed
shards, so any instance sees every writer's events. Each writer periodically
merges only its own generations into the base table. Shared base-table
compaction and index refresh must be scheduled by one elected maintenance
worker per experiment.
