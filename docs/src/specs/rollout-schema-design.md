# Design: A Native RolloutDB Schema for lance-context

**Status:** Draft
**Author:** Beinan Wang
**Date:** 2026-07-08
**Scope:** Add a purpose-built rollout schema to lance-context as a second first-class record type, so one library serves both multi-agent conversation context and RL rollout storage.

---

## 1. Summary

lance-context today stores one schema — `ContextRecord`, tuned for multi-agent conversation memory. This document adds a **second, independent schema** — `RolloutRecord` — designed from scratch for reinforcement-learning rollout storage.

We are **not** reusing `ContextRecord` or forcing rollout signals into its `metadata` escape hatch. RL training signals are first-class, high-volume, hot-path data; they deserve typed columns and their own table. lance-context becomes a **multi-schema store**: two purpose-built record types, each its own Lance dataset, sharing one set of infrastructure — versioning, blob offload, namespace partitioning, and the relationship graph.

---

## 2. What RolloutDB is, and why it matters

A **rollout** is one trajectory produced during RL training: a problem (prompt) is given to the policy model, the model generates a response — often multi-turn, with tool calls — a grader or verifier scores it, and the scored, token-aligned trajectory becomes a training example.

**RolloutDB is the system of record for that data.** It sits between two fleets:

```text
   generation fleet                RolloutDB                 learner
  (1000s of workers)   ─ append ▶  (this store)  ─ project ▶  (gradient steps)
   policy → trajectory              trajectories +            reads token-aligned
   grader → reward                  training signals          batches, updates policy
```

It is the backbone of the RL flywheel. Its hard requirements:

1. **High fan-in ingest** — thousands of rollout workers append trajectories concurrently.
2. **Cheap token-aligned reads** — the learner pulls only `[tokens, logprobs, advantage]` for a batch; it must not pay for text, artifacts, or grader logs it does not read.
3. **Exact reproducibility** — you must be able to say *precisely which rollouts trained which checkpoint*, and replay them.

Today MAI serves this with a Snowflake **star schema** (many tables joined on `ROLLOUT_ID`, fed by a Kafka-Connect streaming pipeline with chunk reassembly). It works, but reproducibility is expensive, the hot path scans a warehouse, and every write fans out across tables.

---

## 3. Why lance-context is the right choice

The properties RolloutDB needs are exactly the properties lance-context already has:

| RolloutDB need | lance-context primitive |
|---|---|
| **Reproducibility** — pin the exact training set per gradient step; A/B reward functions | **Native dataset versioning** — `checkout(v)` recovers an exact rollout set; `evaluate_versions` compares them. A star schema cannot snapshot cheaply. |
| **Cheap hot-path reads** — learner reads a few columns of millions of rows | **Columnar projection** — project `[output_tokens, output_logprobs, advantage]`; text and blobs are never touched on the scan. |
| **Large artifacts without bloating scans** | **Blob v2 offload** — bytes sink to an independent file; the row keeps a reference; column scans skip them. |
| **Atomic trajectory writes at high fan-in** | **Single-append** — one worker writes a whole trajectory (messages + signals + artifacts) in one atomic append. No multi-table fan-out, no `_CHUNK_INDEX` reassembly. |
| **Cross-rollout structure** — grades, prompt→N-responses groups | **Relationship graph** — real edges, not join keys. |
| **No warehouse, no pipeline** | **Embedded library over object storage** — workers write directly; no Snowflake, no streaming connector. |

The RL signals are the one thing lance-context lacks: **variable-length numeric arrays aligned to tokens.** That is the gap this schema fills.

---

## 4. Architecture: two schemas, one store

```text
                      ┌──────────────── lance-context ────────────────┐
                      │                                               │
   ContextRecord ─────┤  shared infrastructure:                       ├──── RolloutRecord
   (conversation)     │  • Lance native versioning                    │     (RL rollouts)
   its own dataset    │  • blob v2 payload offload                    │     its own dataset
   its own columns    │  • namespace / partition manifest             │     its own columns
                      │  • relationship graph                         │
                      │  • eval / export                              │
                      └───────────────────────────────────────────────┘
```

- Each schema is an **independent Lance dataset** with its own Arrow schema builder. They do not share columns and do not need to be schema-compatible.
- The **infrastructure is shared**: versioning, blob offload, namespace partitioning, relationships, and eval are schema-agnostic and serve both.
- Partitioning maps a rollout selector (e.g. `experiment`) to one physical dataset, replacing MAI's time-bucket cluster key.

This mirrors how the store already gates capability by construction — but at the schema level rather than the column-flag level.

---

## 5. The `RolloutRecord` schema

One row per **message** in a rollout (assistant turn, tool call, grade, or artifact). A trajectory is many rows sharing `rollout_id`; the N GRPO samples of one prompt share `problem_id`.

**Identity & grouping**

| Column | Arrow type | Purpose |
|---|---|---|
| `id` | `Utf8` | unique row id |
| `rollout_id` | `Utf8` | the trajectory |
| `problem_id` | `Utf8` | prompt / GRPO group key linking N samples of one prompt |
| `dataset` | `Utf8` | source dataset name |
| `sequence_order` | `Int32` | explicit intra-rollout ordering (`created_at` is not a reliable total order) |
| `role` | `Dictionary<Utf8>` | `assistant` / `tool` / `grade` / `artifact` / … |
| `created_at` | `Timestamp(µs)` | write time |

**Message content**

| Column | Arrow type | Purpose |
|---|---|---|
| `content` | `LargeUtf8` | message text |
| `content_type` | `Utf8` | MIME type |

**Claim-check offloaded message fields** *(oversized rollout-message fields, each an individually-projectable nullable column)*

| Column | Arrow type | Purpose |
|---|---|---|
| `model_input_string` | `LargeUtf8` | rendered model prompt string |
| `model_output_string` | `LargeUtf8` | raw model completion string |
| `rationale` | `LargeUtf8` | grader rationale |
| `problem_text` | `LargeUtf8` | source problem text |
| `user_metadata` | `LargeUtf8` | harness-supplied per-message metadata blob |

**Tokens** *(first-class variable-length arrays)*

| Column | Arrow type | Purpose |
|---|---|---|
| `input_tokens` | `List<Int32>` | prompt-side token ids |
| `output_tokens` | `List<Int32>` | generated token ids |
| `num_input_tokens` | `Int32` | prompt length |
| `num_output_tokens` | `Int32` | completion length |

**Training signals** *(first-class variable-length arrays — the gap we fill)*

| Column | Arrow type | Purpose |
|---|---|---|
| `output_logprobs` | `List<Float32>` | generation-time (old) logprobs — the PPO/GRPO ratio |
| `input_logprobs` | `List<Float32>` | prompt-side logprobs |
| `ref_logprobs` | `List<Float32>` | reference-model logprobs — the KL term (nullable; may live in the annotations dataset, §7) |
| `loss_mask` | `List<Int8>` | gradient only on model tokens (multi-turn / tool use) |
| `advantage` | `Float32` | group-normalized advantage (scalar; per-token GAE can graduate to `List<Float32>` later) |

**Reward**

| Column | Arrow type | Purpose |
|---|---|---|
| `reward` | `Float32` | final shaped reward |
| `raw_reward` | `Float32` | pre-shaping grader output |
| `grader_id` | `Utf8` | which grader/verifier produced the score |
| `score` | `Float32` | grader score |

**Training control & provenance**

| Column | Arrow type | Purpose |
|---|---|---|
| `include_in_training` | `Bool` | learner filter |
| `exclude_reason` | `Utf8` | why excluded, when it is |
| `policy_version` | `Utf8` | checkpoint that generated this trajectory |

**Graph, artifacts, escape hatch**

| Column | Arrow type | Purpose |
|---|---|---|
| `relationships` | `List<Struct{target_id, relation, weight}>` | cross-rollout grades, message→grade edges |
| `binary_payload` | `LargeBinary` *(blob v2)* | artifact bytes, physically offloaded (§6) |
| `payload_size` | `Int64` | artifact size |
| `payload_checksum` | `Utf8` | artifact checksum |
| `metadata` | `Utf8` *(JSON)* | harness metadata — the open-ended escape hatch, for genuinely unstructured fields only |

All signal and token columns are **nullable**: a grade row carries reward but no tokens; an assistant row carries tokens but no score. Trainers project only the columns they need; nothing else is read.

---

## 6. Artifacts: rows, not a side channel

Agent rollouts emit artifacts — code, tool logs, sandbox state, verifier output — that a numeric schema has no place for. We store them **in-table as their own rows**, versioned with the trajectory, physically offloaded via **blob v2** so scans never touch the bytes. This beats an external object-store reference on every axis that matters: single-append atomicity, `checkout(v)` recovers the exact artifact, and one retention/GC path instead of two.

**Decisions (locked):**

- **One row per artifact.** `role="artifact"`; `content_type` carries the file's own MIME; attached to its producing message via a `relationships` edge and to its rollout via `rollout_id`. Multiple artifacts per step = multiple rows.
- **`binary_payload` is a blob v2 column.** Bytes sink to an independent folder/file; the row keeps only the reference in-band. `payload_size` / `payload_checksum` carry size and checksum.
- **`artifact_type` / `filename`** live in `metadata` — no typed column yet.
- **No dedup** in this phase. Shared repo/sandbox checkouts across a group are stored per-row; content-addressing is a later option if reuse proves high.

### Shape

```text
# One rollout worker, one append — message and its artifact together:
append([
  {role: "assistant", content: "...", rollout_id: "r-42", sequence_order: 3,
   output_tokens: [...], output_logprobs: [...], advantage: 0.7},        # signal row
  {role: "artifact",  content_type: "text/x-python", rollout_id: "r-42",
   binary_payload: <bytes>,           # blob v2 → offloaded file
   payload_size: 1234, payload_checksum: "sha256:...",
   metadata: {filename: "solution.py", artifact_type: "code"},
   relationships: [{target_id: <message_id>, relation: "produced_by"}]},
])

# Learner:  project [output_tokens, output_logprobs, advantage] — blob column untouched
# Verifier: filter role="artifact", lazy-load blob on demand
```

---

## 7. Learner annotations: a companion dataset

Reference/learner logprobs are **re-annotated** for the same trajectory by successive learner checkpoints. Their write cadence and lifecycle differ from the rollout rows, so they get a small **companion dataset** keyed `(rollout_id, sequence_order, learner_iteration)` with a `learner_logprobs: List<Float32>` column.

**One trap:** `learner_iteration` is **not** a Lance dataset version. A dataset version is a snapshot of the whole table; `learner_iteration` is a per-segment annotation axis. Model it as a column, not a `checkout`.

---

## 8. Change surface

1. **New record type** — `RolloutRecord` (new module, e.g. `rollout.rs`) with the fields in §5.
2. **New schema builder** — an Arrow schema for the rollout dataset, `binary_payload` marked as a blob v2 column; parallels `schema_with_options` but is its own function.
3. **New store surface** — open/create/append for rollout datasets, reusing the versioning, namespace, blob, and relationship machinery unchanged.
4. **Batch conversion** — `RolloutRecord ⇄ Arrow RecordBatch` builders with presence guards, so appending a new column into an older dataset fails with a clear error.
5. **Companion annotations** — minimal schema + append path for the learner-annotation dataset (§7).

The existing `ContextRecord` path is untouched — this is additive.

---

## 9. Open items

- **Blob v2 encoding marker** — confirm the exact field-metadata key/value for V2 against the pinned Lance version before implementation.
- **Advantage granularity** — start scalar (`Float32`); promote to `List<Float32>` if per-token GAE is required.
- **Dedup** — deferred; revisit if intra-group artifact reuse is high.
