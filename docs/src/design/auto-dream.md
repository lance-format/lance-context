# Design: Offline Memory Consolidation ("auto-dream")

| | |
|---|---|
| **Status** | Draft — for discussion |
| **Issue** | [#99 — RFC: Offline memory consolidation ("auto-dream")](https://github.com/lance-format/lance-context/issues/99) |
| **Relates to** | [#96 — Curate records into trainable datasets](https://github.com/lance-format/lance-context/issues/96) · [#37 — Architecture discussion](https://github.com/lance-format/lance-context/issues/37) · [#16 — Compaction] |
| **References** | Generative Agents (reflection) · 2025 "sleep-time compute" line |
| **Scope** | Storage-native machinery + a user-supplied abstraction hook. No bundled models, no agent framework, no hard deletion. |

## 1. Summary

Long-running agents accumulate large volumes of low-level episodic memory. As the store
fills with redundant / near-duplicate entries, recall quality and cost degrade.
**Consolidation** ("auto-dream") summarizes many raw turns into a few higher-signal
memories, links each derived memory back to its sources, and **retires** (does not delete)
the now-redundant raw records — keeping online memory compact, auditable, and reversible.

This is deliberately *storage-native machinery only*: candidate selection, clustering,
write-back, lifecycle, and versioning live in `lance-context`; the **abstraction step is a
user-supplied hook**. We bundle no LLMs.

The central design decision this doc records: **auto-dream and #96 (curate/export) are the
same loop and should share one engine** — `select → transform → sink` — differing only in
*sink* (export writes JSONL; consolidation writes derived records back and supersedes
sources) and *trigger* (on-demand vs. idle/background).

## 2. Background

### 2.1 Where we are today

`lance-context` already holds the raw memory and every primitive consolidation needs:

- **embeddings** + cosine distance — already used in `export.rs` for near-duplicate
  **dedup** and **decontamination**; the same machinery clusters related memories.
- **relationships** (`relationships` column + `related()` traversal) — link a derived
  memory to the source records it summarizes.
- **lifecycle** (`upsert`/supersede, `update`/retire, `delete`/tombstone) — replace
  consolidated-away records non-destructively.
- **versioning** (`version()` / `checkout()`) — every pass is a new, inspectable,
  rollback-able version.
- **background compaction scheduler** (`should_compact`, `start_background_compaction`) —
  an existing pattern for bounded, opt-in background work.

What does **not** exist yet is the *write-back*: there is no API that, given a cluster,
inserts a derived record, links it to its sources, and retires those sources atomically.

### 2.2 What #99 asks

An opt-in process that: (1) selects redundant/related sets, (2) calls a user hook to
abstract each set into a consolidated record, (3) writes the result back and retires the
originals, (4) optionally runs on a schedule, (5) stays reversible and reproducible.

### 2.3 Why storage-native fits

The same bet as #96: consolidation maps directly onto existing storage features, so it
belongs next to the data rather than re-deriving it in the agent layer from flat exports.
"Curate in place + version" beats "export, transform externally, re-import".

## 3. Goals / Non-goals

**Goals**
- A shared `select → transform → sink` engine; consolidation is a second sink on the #96
  substrate.
- An opt-in, synchronous `consolidate(options, hook)` first; scheduler later.
- Non-destructive, reversible, reproducible: retire (never tombstone) sources; record
  params + source ids; pin/inspect by version.

**Non-goals**
- No bundled models / prompting — abstraction is a hook.
- No hard deletion for consolidation (lifecycle only).
- Not physical compaction (#16): that optimizes files; this is *semantic* consolidation.
- No background scheduler **on by default** until demand is validated (§10).

## 4. The consolidation loop, in brief

```
        ┌─────────── shared engine (with #96) ───────────┐
raw ──► select (filters + lifecycle + cluster) ──► transform (user hook) ──► sink
        └────────────────────────────────────────────────┘                   │
                                                          export (#96): JSONL + manifest
                                                          consolidate (#99): insert derived
                                                            + relationship-link + supersede
                                                            + version bump
```

## 5. Design

### 5.1 Principle: derive-and-supersede; one engine, two sinks

`export.rs` (from #96) already implements the **read side**:

- **candidate selection** — `list_filtered_with_options` (lifecycle-correct: drops
  tombstoned/expired/retired/superseded/contradicted) + metadata/quality filters;
- **clustering / redundancy** — cosine `dedup` / `decontaminate` over embeddings;
- **grouping/aggregation** — by `session_id` / `run_id` / `tenant` / external-id prefix,
  deterministically ordered;
- **provenance + manifest** — source ids, version, params.

Consolidation reuses all of that and replaces only the **sink**. Proposed refactor:
extract selection + clustering + grouping from `export.rs` into a shared `curate` module,
then implement both sinks on top of it. The transform hook differs only in signature.

### 5.2 Candidate selection & clustering

Two selection modes, both already expressible:

1. **Structural** — group by `session_id` / `run_id` / time window (existing `group_records`).
2. **Semantic** — cluster by embedding similarity. v1 = greedy near-duplicate collapse
   using the existing cosine helper, with an exposed similarity floor and `max_cluster_size`.

A cluster proceeds to abstraction only if it is **above a similarity floor** *and* **at or
above a minimum size** (don't summarize singletons).

### 5.3 Abstraction hook (user-supplied)

```
type ConsolidateHook = (cluster: &[ContextRecord]) -> Option<DerivedRecord>;
```

Given a cluster, the caller's hook returns a consolidated record (summary/abstraction), or
`None` to skip. `lance-context` invokes the hook and persists the result; it bundles no
models. (FFI note: in Python this is a callback; the engine batches clusters and calls the
hook per cluster between async write phases.)

### 5.4 Write-back: insert derived + link + supersede + version

For each accepted cluster with derived record `D` over sources `S_1..S_n`:

1. **Insert** `D` with `source="consolidated"`.
2. **Link** `D` to every `S_i` via `relationships` (`relation = "consolidates"`), so trace-back
   is a `related()` traversal.
3. **Retire** each `S_i` non-destructively — set `superseded_by_id = D.id` (and/or
   `retired_at` + `retired_reason = "consolidated:<D.id>"`). `is_hidden_by_lifecycle()`
   already hides records with `superseded_by_id`/`retired_at` from default reads.
4. **One version bump** for the whole pass.

> **API gap to fill:** today supersession is 1→1 (`upsert`) and `RecordPatch` does not expose
> `superseded_by_id`. Consolidation is N→1, so we need a small atomic primitive —
> "mark records `S_1..S_n` superseded-by `D` and insert `D`" — rather than reusing `upsert`
> per record. This is the main new storage operation auto-dream introduces.

### 5.5 Search interaction & traceability

- **Default reads / `search`**: consolidated (L0) records are visible; retired originals
  drop out of default recall (via `is_hidden_by_lifecycle`).
- **Traceability**: originals remain in the store and are reachable with
  `LifecycleQueryOptions { include_retired: true }` + the `consolidates` relationships —
  exactly the L0→L1→L2 evidence trace-back described in #37.
- So raw memory stays queryable for evidence; it just stops competing in normal recall.

### 5.6 Scheduling (deferred)

A background/idle pass would reuse the `should_compact` / `start_background_compaction`
pattern: bounded, cancellable, threshold/cadence-triggered. **Deferred** until the
synchronous API proves out and demand is validated (§10).

### 5.7 API sketch

```rust
pub struct ConsolidateOptions {
    pub filters: Option<RecordFilters>,
    pub lifecycle: LifecycleQueryOptions,      // default visible-only
    pub cluster_by: ClusterBy,                 // Structural(GroupBy) | Semantic { threshold, max_size }
    pub min_cluster_size: usize,               // skip singletons
    pub scope_boundary: ScopeBoundary,         // never cross session/tenant by default
    pub dry_run: bool,                         // report clusters without writing
}

pub struct ConsolidationManifest {
    pub version: u64,
    pub clusters: usize,
    pub derived: usize,
    pub retired_sources: usize,
    pub params: ConsolidateParamsSummary,
    pub derived_ids: Vec<String>,
}

impl ContextStore {
    pub async fn consolidate(
        &mut self,
        options: &ConsolidateOptions,
        hook: impl ConsolidateHook,
    ) -> LanceResult<ConsolidationManifest>;
}
```

`dry_run` returns the proposed clusters + counts without writing — a cheap way to tune
thresholds before committing a pass.

## 6. Shared engine with #96 — recommendation: **yes**

Build **one** selection/clustering substrate, **two** sinks. Concretely:

- Move `list_filtered_with_options`-based selection, the cosine cluster/dedup helpers, and
  `group_records` from `export.rs` into a shared `curate`/`cluster` module (pure refactor,
  no behavior change; #96 keeps passing).
- `export_training` becomes "select → format → JSONL sink"; `consolidate` becomes
  "select → hook → write-back sink".

This avoids two parallel near-duplicate implementations and means improvements to
clustering benefit both training-data curation and online consolidation.

## 7. Guardrails against over-abstraction

- **Reversible**: sources are retired, never deleted; a pass is a single version you can
  `checkout`/roll back.
- **Minimums**: similarity floor + `min_cluster_size` prevent trivial/lossy merges.
- **Scope boundaries**: never abstract across `session_id`/`tenant` by default.
- **Auditable**: `ConsolidationManifest` records params + source/derived ids; the
  `consolidates` relationships preserve the full lineage.
- **`dry_run`** before committing.

## 8. Interaction with existing features

- **Lifecycle**: reuses supersede/retire visibility; adds the N→1 supersede primitive (§5.4).
- **Relationships**: the lineage carrier; `migrate_relationships` already backfills the column.
- **Versioning**: each pass = one version; reproducible and inspectable.
- **Compaction (#16)**: orthogonal — semantic consolidation reduces row count; physical
  compaction reclaims files. They compose.
- **Curation/export (#96)**: shared engine (§6).
- **Namespaces (#94)**: consolidation runs per partition; cross-partition is out of scope.

## 9. Alternatives considered

- **Do it in the agent layer** — viable, but loses in-store versioning, lifecycle audit,
  and reuse of the #96 substrate; re-derives from flat data. Keep as the fallback for users
  who don't want in-store consolidation.
- **Hard-delete consolidated rows** — rejected: not reversible/auditable (matches #96 audit
  guidance: don't conflate curation/consolidation with record removal).
- **Two parallel engines (#96 + #99)** — rejected in favor of the shared engine (§6).

## 10. Open questions

- **Demand** — what concrete agent workflow needs *in-store* (vs. agent-layer)
  consolidation? Same validation bar as #96. Gate the scheduler on a real use case; ship the
  synchronous API first.
- **N→1 supersede primitive** — exact shape/atomicity (§5.4): extend `RecordPatch` vs. a
  dedicated `consolidate` write op.
- **Clustering quality** — is greedy cosine enough for v1, or do we need a real clustering
  pass (e.g. connected-components over a similarity graph)?
- **Hook batching / failure semantics** — partial-failure behavior if the hook errors on
  one cluster mid-pass (all-or-nothing per pass vs. best-effort with a report).

## 11. Phased rollout

1. **This doc** — record scope, the shared-engine decision, guardrails (RFC acceptance).
2. **Refactor** — extract shared selection/cluster/group module from `export.rs` (no
   behavior change).
3. **N→1 supersede primitive** — the one new storage op (§5.4), with tests.
4. **`consolidate(options, hook)`** — synchronous, opt-in, with `dry_run`; core + Python.
5. **Scheduler** — idle/cadence background pass, gated on validated demand (§10).

## 12. Acceptance criteria mapping (#99, RFC stage)

- [x] Scope / non-goals agreed (machinery + hook, no bundled models) — §3.
- [x] Shared-engine decision with #96 — **yes**, §6.
- [x] Short design doc under `docs/design/` — this file.
- [ ] Maintainer sign-off → then implementation tracked in follow-ups (§11).

## 13. References

- Issue #99 (this RFC), #96 (curate/export), #37 (architecture), #16 (compaction).
- `docs/design/partitioned-namespace.md` — companion design doc (same storage-native bet).
- Generative Agents (reflection); 2025 "sleep-time compute" line.
