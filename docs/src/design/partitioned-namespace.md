# Design: Partitioned-Namespace Layout for Scoped Agent Context

| | |
|---|---|
| **Status** | Draft — for discussion |
| **Issue** | [#90 — Explore partitioned namespace layout for scoped agent context](https://github.com/lance-format/lance-context/issues/90) |
| **Supersedes / refines** | [#37 — Change the management of data tables and add some interface exposure for agent](https://github.com/lance-format/lance-context/issues/37) |
| **References** | [Lance Partitioning Spec](https://lance.org/format/namespace/partitioning-spec/) · [Lance Namespace Spec](https://lance.org/format/namespace/) |
| **Scope** | Conventions + a thin helper API. No new storage engine, no catalog/database. |

## 1. Summary

`lance-context` today is a **single fat table**: one Lance dataset per context URI, holding
one `ContextRecord` schema that already carries scope-ish fields (`bot_id`, `session_id`,
`run_id`), a free-form `metadata` JSON blob, embeddings, relationships, and lifecycle
columns. Scoping is done entirely with **metadata filters inside that one dataset**.

This works well until a deployment needs *physical* isolation between scopes — separate
tenants, agents, environments, or content "directions" (memory vs. knowledge base vs.
skills) that want independent lifecycle, compaction, vector indexes, deletion, or access
control. The natural temptation (and the original direction in #37) is to grow into many
purpose-built tables (`ctx_agent`, `ctx_kb`, `ctx_meta`) plus an `information_schema`-style
registry. Maintainer guidance on #37 pushed back on that:

> I'm hesitating to make lance-context as a database with an information_schema.
> Maintaining the schema of each table would be a pain in the future.
> Any chance to use the partitioned namespace instead of making it as a database?

This document takes that suggestion seriously. It proposes that scoped isolation be
expressed as a **Lance Partitioned Namespace** over the **existing single `ContextRecord`
schema**, rather than as a family of bespoke tables with a hand-rolled catalog. A
partitioned namespace is, by definition, *"a collection of tables that share a common
schema … physically separated and independent, but logically related through partition
fields"* — which is exactly the shape we want: **one schema to maintain, many physically
isolated partitions, and an optional unified cross-partition view.**

The deliverable here is **conventions + a small, optional URI/namespace helper** — not a
catalog, not table families, not an `information_schema`.

## 2. Background

### 2.1 Where we are today

A `Context` is created over a single URI and is backed by exactly one Lance dataset:

```python
ctx = Context.create("s3://bucket/prefix/context.lance")
ctx.add("user", "…", bot_id="support-bot", session_id="incident-123",
        metadata={"tenant": "acme", "scope": "team", "source_uri": "docs://…"})
ctx.list(filters={"bot_id": "support-bot", "session_id": "incident-123",
                  "scope": "team"})
```

Scope is carried two ways:

- **Typed columns** that already exist on `ContextRecord`: `bot_id`, `session_id`,
  `run_id`, `role`, `content_type`, plus lifecycle columns.
- **Free-form `metadata` JSON** (e.g. `tenant`, `scope`, `tags`, `source_uri`), filtered
  by `list`/`search`/`retrieve`.

Everything lives in one dataset. That is simple and correct, and **must stay the default.**

### 2.2 What #37 was really asking for

The #37 thread had two distinct ideas tangled together:

1. **Layering (L0/L1/L2)** — summary / structured+vector / raw. This is a *per-record*
   property, orthogonal to isolation.
2. **"Different directions"** — memory, knowledge base, skills, tool calls — which the
   author modeled as separate table families (`ctx_agent` / `ctx_kb` / `ctx_meta`) and an
   `information_schema` (`ctx_meta`) to register them.

The maintainer's objection is specifically to #2: a growing set of tables, each with its
own schema, plus a registry we have to maintain, turns `lance-context` into a small
database. The cost is schema sprawl and migration pain.

### 2.3 Why a partitioned namespace fits

The Lance Partitioning Spec already solves "many physically separate tables that share one
schema and one lifecycle, queryable together when needed":

- All partitions **share a single namespace schema**; the spec *requires* that "all
  partition table schemas must be consistent with each other, as well as with the
  namespace schema." → **one schema to maintain**, which is the whole point of #37's
  pushback.
- The registry is the spec's own **`__manifest` table**, maintained by the namespace
  format — not a bespoke `information_schema` we own.
- Each partition is **a standard, independently accessible Lance `Dataset`** — so a single
  partition is exactly today's `Context`, unchanged.
- Cross-partition queries are supported via **partition pruning** over `__manifest`.

So we get physical isolation *and* a unified view, without becoming a database.

## 3. Goals / Non-goals

Mirrors #90.

**Goals**

- Recommend upstream-friendly **partition naming / spec conventions** for scoped context
  (tenant, agent, session, source, environment).
- Clarify **which scope dimensions belong in path/namespace layout vs. record metadata.**
- Sketch a **small helper API** to resolve/open a partition without a catalog.
- Show **when to use one context with metadata filters vs. separate partitions.**
- Give **migration guidance**: start with one dataset, split by scope later.
- Stay compatible with the single fat-table model and the #37 guidance.

**Non-goals**

- No `information_schema`, table-family management, SQL catalog, or database abstraction.
- `lance-context` does **not** own tenant authN/Z or policy enforcement.
- Not a replacement for metadata filters *inside* a single context dataset.
- No new storage engine; no change to the `ContextRecord` payload model.

## 4. Lance partitioned namespace, in brief

(Condensed from the [Partitioning Spec](https://lance.org/format/namespace/partitioning-spec/);
read it for the authoritative version.)

A partitioned namespace is a directory catalog whose `__manifest` table metadata declares
one or more **partition specs**. Logical layout:

```text
Root Namespace  (__manifest Lance table)
  metadata:
    schema          = <shared ContextRecord schema>
    partition_spec_v1 = [tenant, agent]
        │
        ▼
   Spec Version Level  ──  v1, v2, …   (one per partition-spec version)
        │
        ▼
   Partition Namespaces  ──  one level per partition field, in spec order
        │                    (names are random 16-char base36 ids)
        ▼
   Partition Table  ──  fixed name "dataset" = a normal Lance Dataset
```

Key properties we rely on:

- **Shared schema.** Field IDs (`lance:field_id`) are stable; partition specs reference
  source columns by id, so columns can be renamed without breaking partitioning.
- **Partition spec** = `{ id, fields: [{ field_id, source_ids, transform|expression,
  result_type }] }`. Built-in transforms: `identity`, `year`/`month`/`day`/`hour`,
  `bucket(N)`, `multi_bucket(N)`, `truncate(W)`; or a custom DataFusion `expression`.
- **Random-id partition namespaces.** Partition *values* are **not** put in the path
  (avoids `/`, `=`, `$` issues in tenant/source ids); they live in `__manifest` columns
  named `partition_field_{field_id}`.
- **`__manifest` is the registry.** Query engines push predicates onto its partition
  columns to prune to the matching `dataset` tables (no path parsing).
- **Partition evolution** via new spec versions (`v2`, …). Old data under `v1` stays
  queryable; no rewrite required.
- **Transactions.** A single partition is fully ACID (it is a Lance table). Cross-partition
  writes/reads are *not* atomic by default, but the spec defines an optional
  `read_version`/`read_branch`/`read_tag` commit protocol on `__manifest` for all-or-nothing
  multi-partition visibility.

## 5. Design

### 5.1 Principle: one schema, many partitions

Keep **one `ContextRecord` schema** as the namespace schema. Isolation is achieved by
*splitting rows across partitions*, never by giving each direction its own schema. This is
the single decision that keeps us out of "database with an information_schema" territory:

- "memory", "knowledge base", "skills", "tool calls" are **values of a partition field**
  (or scalar columns), **not** separate tables with separate schemas.
- The thing #37 called `ctx_meta` / `information_schema` is replaced by the spec's
  `__manifest` table, which we do not design or maintain.

### 5.2 Which dimensions go in the layout vs. the record

This is the core guidance #90 asks for.

> Put a dimension in the **partition layout** when you routinely scope an *entire* operation
> to one value of it, and you want that value's data physically isolated. Keep a dimension
> in **record fields/metadata** when it is high-cardinality, cross-cutting, or combined
> ad-hoc with other filters.

| Dimension | Typical cardinality | Recommended placement | Rationale |
|---|---|---|---|
| `tenant` / org | low–med | **Partition** (`identity`) | Hard isolation boundary; per-tenant delete/retention/access; queries almost always tenant-scoped. |
| `environment` (prod/stage) | tiny | **Partition** (`identity`) | Never want stage data bleeding into prod recall. |
| `agent` / bot (`bot_id`) | low–med | **Partition** (`identity`) *if* agents need independent indexes/lifecycle; else **column** | `bot_id` is already a column. Partition only when isolation matters. |
| `source` / `kind` (memory \| knowledge \| skill \| tool_call) | tiny (enum) | **Partition** (`identity`) *or* scalar column | The #37 "directions". Partition when lifecycles diverge (e.g. KB is bulk-rebuilt, memory is append/prune). |
| `user` | high | **Column** + metadata; **`bucket(N)`** if isolation needed | Don't create millions of partitions; hash into N buckets if you must partition. |
| `session_id` | very high | **Column** (`session_id`) | Already a column; fine-grained, cross-cut by other filters. |
| `run_id`, `role`, `content_type`, `tags`, `confidence`, `source_uri` | high / cross-cutting | **Column / `metadata`** | Ad-hoc filter material; never a directory boundary. |
| time | continuous | **Column** (`created_at`); `day`/`month` transform only for cold archival | Recall filters by range, not by exact partition; reserve time partitions for archive tiers. |

Rules of thumb:

1. **Few coarse boundaries → partition.** Many fine values → filter.
2. **Physical needs → partition.** If you need an independent vector index, independent
   compaction cadence, drop-to-delete, or per-scope ACLs at the storage layer, it must be a
   partition. Pure logical filtering can stay a column.
3. **Cap the partition count.** Keep the total number of `dataset` tables in the low
   thousands. For a dimension you need to isolate but whose cardinality is high, use
   `bucket(field, N)` instead of `identity`.
4. **A dimension you partition on must be a typed column** in the shared schema (see
   §5.5) — partition specs reference columns by field id, so they cannot read keys buried
   in the `metadata` JSON blob.

### 5.3 Recommended partition specs

Three conventions, smallest-useful first. Field ids are stable strings.

**A. Tenant-isolated (most common).**

```json
{
  "id": 1,
  "fields": [
    { "field_id": "tenant", "source_ids": [<id of tenant col>],
      "transform": { "type": "identity" }, "result_type": { "type": "utf8" } }
  ]
}
```

**B. Tenant → agent → source (multi-direction, answers #37).**

```json
{
  "id": 1,
  "fields": [
    { "field_id": "tenant", "source_ids": [<tenant>], "transform": {"type": "identity"}, "result_type": {"type": "utf8"} },
    { "field_id": "agent",  "source_ids": [<bot_id>], "transform": {"type": "identity"}, "result_type": {"type": "utf8"} },
    { "field_id": "source", "source_ids": [<source>], "transform": {"type": "identity"}, "result_type": {"type": "utf8"} }
  ]
}
```

`source ∈ {memory, knowledge, skill, tool_call}` — the #37 "directions" become partition
values, not tables. Order matters: outer levels are the coarsest, most-often-scoped
dimensions (the spec nests partition namespaces in field order).

**C. High-cardinality user isolation via bucketing.**

```json
{
  "id": 1,
  "fields": [
    { "field_id": "tenant", "source_ids": [<tenant>], "transform": {"type": "identity"}, "result_type": {"type": "utf8"} },
    { "field_id": "user_bucket", "source_ids": [<user_id>], "transform": {"type": "bucket", "num_buckets": 64}, "result_type": {"type": "int32"} }
  ]
}
```

Caps per-tenant partitions at 64 while still giving physical separation and enabling
storage-partitioned joins against other namespaces bucketed compatibly.

### 5.4 Path / URI conventions (caller-facing)

Per the spec, **physical** partition directories are random-id names like
`b4a3c2d1_v1$k7m2n9p4q8r5s3t6$dataset/`, and partition values live in `__manifest`. Callers
should never construct these by hand. What we standardize is the **caller-facing** address:

- A **namespace root URI** — the directory containing `__manifest`, e.g.
  `s3://bucket/acme-contexts/`. This replaces "one `.lance` dataset" as the top-level
  handle when partitioning is in use.
- A **partition selector** — a dict of `{field_id: value}` resolved against the spec, e.g.
  `{"tenant": "acme", "agent": "support-bot", "source": "memory"}`.

```text
s3://bucket/acme-contexts/                # namespace root (holds __manifest)
   └── (resolved by spec+manifest) ──►  one Lance "dataset" table == a normal Context
```

Because partition values are stored in `__manifest`, **tenant/source ids may contain any
characters** (`/`, `:`, spaces) without escaping — a real advantage over encoding scope
into the path ourselves.

### 5.5 Schema implication: promote partition keys to columns

To partition by a dimension, it must be a first-class column referenced by `source_ids`.
Status of the candidate keys on `ContextRecord` today:

- **Already columns** (usable now): `bot_id` (→ agent), `session_id`, `run_id`, `role`,
  `content_type`, `created_at`.
- **Currently in `metadata` JSON** (would need promotion to typed columns to partition on):
  `tenant`, `source`/`kind`, `environment`, `user_id`.

Recommendation: when (and only when) we implement partitioning, add nullable typed columns
for the dimensions a deployment wants to partition by (start with `tenant` and `source`).
This is additive and backward compatible — existing single datasets simply leave them null.
Keeping these as real columns also makes them filterable in `list`/`search` regardless of
whether they are partitioned, which is strictly better than the JSON blob.

### 5.6 Querying: single-partition fast path, cross-partition fan-out

- **Single partition (the 95% path).** Resolve the selector to one `dataset` and open it as
  today's `Context`. Full ACID, full speed, identical semantics to the current code. Vector
  search, hybrid `retrieve`, `list`, lifecycle — all unchanged, just scoped.
- **Cross-partition (the unified view).** Two sub-cases:
  - *Pruned scan*: a partial selector (e.g. `{"tenant": "acme"}`, all sources) → query
    `__manifest` for matching `dataset` paths, then fan out and merge. For vector search,
    run top-k per partition and merge by distance; for `retrieve`, fuse per-partition RRF
    results. This is partition pruning from the spec applied to recall.
  - *Full scan*: no selector → every partition. Discouraged for vector search at scale;
    intended for analytics/export.

Cross-partition reads are **not** isolated by default (each partition may be at a different
version). When a consistent multi-partition snapshot is required, use the spec's
`read_version`/`read_tag` columns on `__manifest`. We treat strongly-consistent
multi-partition transactions as a later, opt-in phase (§8).

### 5.7 Helper API sketch (thin resolver, not a catalog)

A small layer that (1) reads/writes `__manifest`, (2) resolves a selector to a `dataset`
URI, and (3) returns a normal `Context`. It deliberately exposes **no** create-table /
alter-schema / list-schemas surface — that would be the catalog we are avoiding.

```python
from lance_context.namespace import ContextNamespace, PartitionSpec

# Define once, at namespace creation. The schema is the single ContextRecord schema.
ns = ContextNamespace.create(
    "s3://bucket/acme-contexts/",
    partition_spec=PartitionSpec(fields=["tenant", "agent", "source"]),  # spec B
    storage_options={...},
)

# Open exactly one partition -> a normal Context (today's object, unchanged).
ctx = ns.context(tenant="acme", agent="support-bot", source="memory")
ctx.add("assistant", "runbook owner is the platform team", embedding=vec)

# Partial selector -> cross-partition, pruned via __manifest.
hits = ns.search(vec, limit=10, where={"tenant": "acme"})          # all agents+sources
hits = ns.retrieve(text="runbook owner", vector=vec, limit=5,
                   where={"tenant": "acme", "source": "knowledge"})

# Introspection is read-only and comes straight from __manifest (no info_schema we own).
for part in ns.partitions(where={"tenant": "acme"}):
    print(part.values, part.uri, part.version)
```

Rust mirror (sketch):

```rust
let ns = ContextNamespace::create(root_uri, PartitionSpec::new(&["tenant", "agent", "source"])).await?;
let mut ctx = ns.context(&[("tenant","acme"), ("agent","support-bot"), ("source","memory")]).await?;
ctx.add(&records).await?;

let hits = ns.search(&query, 10, &[("tenant","acme")]).await?; // fan-out + merge
```

`ContextNamespace` is optional sugar; opening a single partition directly by its resolved
dataset URI must keep working so nothing depends on the helper.

## 6. When to partition vs. filter inside one context

A decision aid for users (the doc #90 asks for):

| Situation | Recommendation |
|---|---|
| One app, one team, < ~10M records, scopes only differ logically | **One `Context`**, use `metadata`/column filters. Don't partition. |
| Multiple tenants needing data isolation, independent deletion/retention, or per-tenant ACLs | **Partition by `tenant`.** |
| KB is bulk re-imported on a schedule while chat memory is append+prune | **Partition by `source`** so you can rebuild/compact/index each independently. |
| Millions of users, each needing isolation | **`bucket(user_id, N)`**, not a partition per user. |
| You filter by `session_id`, `tags`, `role`, time range, `confidence` | **Keep as columns/metadata**; never a partition. |
| You need vector indexes tuned differently per scope | **Partition** (each `dataset` has its own index). |
| You occasionally need a cross-scope view | Fine — that's the unified namespace; use a partial/empty selector. |

Default remains: **start with one context and metadata filters.** Reach for partitions only
when a *physical* property (isolation, lifecycle, index, deletion, ACL) demands it.

## 7. Migration & evolution

### 7.1 Single dataset → partitioned namespace

Existing single-dataset `Context`s keep working with no change; the namespace layer is
purely additive. To adopt partitioning:

1. **New deployments**: create a `ContextNamespace` with `partition_spec_v1`; write through
   it from day one.
2. **Existing single dataset**: stand up a namespace root, register the current dataset as
   the initial partition (e.g. the `default`/legacy scope) in `__manifest`, and route new
   scoped writes to new partitions. No bulk rewrite of existing rows is required to *start*;
   backfilling the new partition-key columns can be incremental (rows without the key sort
   into a null/`default` partition).

### 7.2 Partition spec evolution

Use spec versions exactly as the spec intends:

- Start at `partition_spec_v1 = [tenant]`.
- Later add agent/source: introduce `partition_spec_v2 = [tenant, agent, source]`. New
  writes land under `v2/`; `v1/` data stays queryable; cross-version queries reconcile via
  each version's spec. **No migration of old partitions needed.**

Because partition fields are referenced by stable field id, renaming the underlying column
later does not break existing specs.

## 8. Interaction with existing features

- **Versioning / time-travel**: each partition is a normal Lance dataset, so `checkout`,
  `snapshot`, `fork` work per partition unchanged. Namespace-wide consistent snapshots are
  the opt-in `read_version`/`read_tag` mechanism (Phase 3).
- **Compaction**: per partition (a key benefit — compact hot memory partitions without
  touching a large static KB).
- **Vector / FTS indexes**: per partition; tune independently. Cross-partition search merges
  per-partition results.
- **Delete / `forget`**: still a versioned logical tombstone within a partition; dropping an
  entire partition becomes a clean physical delete path for "forget this whole tenant".
- **`storage_options`**: unchanged; applies to the namespace root and inherited by
  partitions.
- **Relationships / GraphRAG**: edges are within-record and unaffected. Note: relationship
  *traversal* is naturally partition-local; cross-partition edges resolve through the
  fan-out path. Call this out as a known limitation, not a feature, for v1.

## 9. Alternatives considered

1. **Table families + `information_schema` (original #37).** Rejected per maintainer
   guidance: per-table schemas and a hand-maintained registry are ongoing maintenance and
   migration cost — i.e. building a database.
2. **Metadata filters only (status quo, forever).** Insufficient when a scope needs physical
   isolation: no independent deletion/retention, no per-scope index tuning, no storage-level
   ACL, and one giant dataset to compact. Great default, wrong ceiling.
3. **Many independent `.lance` datasets with an app-side map.** Loses the shared-schema
   guarantee, the `__manifest` registry, partition pruning, and cross-scope query — every
   consumer reinvents discovery. The partitioned namespace *is* the standardized version of
   this, so we should use it rather than re-implement it badly.
4. **Partition by every dimension (incl. session/user).** Rejected: directory/partition
   explosion. Hence the §5.2 cardinality rules and `bucket(N)`.

## 10. Open questions

- **Default `source`/`kind` vocabulary**: do we bless an enum (`memory`/`knowledge`/`skill`/
  `tool_call`) or leave it free-form? Leaning: document the convention, don't enforce.
- **Cross-partition vector search ranking**: top-k-per-partition + merge is simple but reads
  every selected partition. Is a coarse L0-style summary partition (ties back to #37's L0)
  worth it as a pre-filter? Probably a later optimization.
- **Manifest dependency**: do we depend on `lance-namespace` clients, or implement the
  minimal `__manifest` read/resolve ourselves to keep the dependency surface small? Affects
  Phase 2 scope.
- **Helper ergonomics**: is `ContextNamespace` the right primitive, or should `Context.open`
  simply accept a `partition={…}` kwarg over a namespace root?

## 11. Phased rollout

Incremental, docs-first, each phase independently shippable (matches #90's acceptance
criteria):

- **Phase 0 (this doc).** Conventions agreed: dimension placement (§5.2), partition specs
  (§5.3), URI conventions (§5.4), when-to-partition guidance (§6).
- **Phase 1 — read/resolve helper.** Promote `tenant`/`source` to typed columns; a thin
  `ContextNamespace` that creates `__manifest`, resolves a full selector to one `dataset`,
  and returns today's `Context`. Single-partition only. No semantics change for existing
  users.
- **Phase 2 — cross-partition reads.** Partition pruning over `__manifest`; fan-out +
  merge for `search`/`retrieve`/`list`; read-only `partitions()` introspection.
- **Phase 3 — multi-partition consistency (opt-in).** `read_version`/`read_tag` commit
  protocol for consistent cross-partition snapshots; whole-partition drop as physical
  `forget`.

## 12. Acceptance criteria mapping (#90)

| #90 criterion | Where addressed |
|---|---|
| Propose upstream-friendly partition naming conventions | §5.3, §5.4 |
| Clarify path/namespace layout vs. record metadata | §5.2 (decision table + rules) |
| Follow-up implementation can be incremental (docs first / small helper) | §11 |
| Compatible with single-table model and #37 guidance | §5.1, §5.5, §8, §9 |

## 13. References

- Lance Partitioning Spec — <https://lance.org/format/namespace/partitioning-spec/>
- Lance Namespace Spec — <https://lance.org/format/namespace/>
- Issue #37 — table management & agent interfaces (and maintainer guidance)
- Issue #90 — explore partitioned namespace layout for scoped agent context
