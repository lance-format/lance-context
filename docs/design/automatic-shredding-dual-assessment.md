# Automatic shredding: baseline and Arrow Variant

Measured on 2026-11-17 using the repaired baseline and the separate Arrow Variant
implementation in one release native extension. Both paths pass the correctness
checks below. Variant provides the requested standard representation and automatic
layout replacement, but is slower and uses more memory in these synthetic cases.
The baseline remains available; these results do not justify replacing it by default.

The required implementation is `VariantDocumentStore`, based on canonical Arrow
Variant. `DocumentStore` remains the separately maintained baseline. Both choose
fields automatically; callers provide selection limits, not extraction paths.

| Behavior | DocumentStore baseline | VariantDocumentStore |
| --- | --- | --- |
| Stored representation | Generated typed columns + residual JSON | ID + one `arrow.parquet.variant` column |
| Shredding/reconstruction | lance-context JSON codec | Official Arrow Variant kernels |
| Field selection | Bounded growth of existing layout | Bounded reselection; incoming string paths can replace old fields |
| Layout changes | Add columns; historical text can need promotion | Atomic full rewrite through Arrow unshred/reshred |
| ID lookup | Maintained BTree index | Maintained BTree index |
| Text search | Native typed-text indexes + residual string scan | Native standard typed-child indexes + residual/literal-key scan |
| Equality | Existing baseline logical-field implementation | Projected Arrow path extraction and comparison; no predicate pushdown |

## What the Variant implementation does

1. Profile a bounded sample of incoming JSON and infer a typed layout.
2. Convert values with Arrow's Variant builder, retaining unsigned integer
   precision and explicit nulls.
3. Invoke Arrow's `shred_variant`. Preserve incompatible rows in the canonical
   residual to prevent Arrow casting from silently changing JSON values/types.
4. Store standard metadata/value/typed_value children in Lance. The dominant
   extracted text occurs once in the current record; indexes and historical
   versions retain their normal independent storage costs.
5. For full reads, invoke Arrow unshred/JSON conversion. For path reads, project
   only the requested subtree and ancestor residuals, unshred that projection,
   then call Arrow's path kernel. This preserves literal keys and type changes.
6. If automatic reselection changes the schema, rewrite existing rows and append
   new rows in a single non-rebasing transaction. Pinned readers retain old data.

## Dependency changes

The new Variant adapter adds no Lance vendor edits. It uses Arrow 58.3-compatible
Variant crates and pins a small metadata-validation fix in the Arrow fork for
legal empty object keys. It does not copy Arrow sources into lance-context.
The existing baseline's 13-line encoder correction remains in the dependency.
The approximately 598k vendored Rust source lines predate that feature.

## Boundaries that affect performance

- The current strict-type guard falls back at whole-row granularity. One
  incompatible selected scalar can leave a large text field in the root residual.
  The mixed-type benchmark measures this case explicitly.
- Layout migration costs O(existing data); retained snapshots also retain old
  files. A write-triggered rewrite is synchronous, not background maintenance.
- Residual search can scan all residuals on rare/no-hit queries. Indexed hits rank
  first and fallback hits have score zero; the combined ranking is not global BM25.
- Native nested text indexing is limited to plain ASCII identifier components.
  Strings under empty, dotted, quoted or other keys remain searchable via scanning.
- Point-read/index maintenance happens after the durable data commit. Failure is
  logged without falsely reporting a committed append as failed; explicit
  preparation retries index maintenance.
- Native encoding targets 4 MiB/1024 rows, with one oversized record allowed.
  This is not a process memory cap. Python `add_iter` has per-batch atomicity;
  previous yielded batches survive later source or encoding errors.
- This is an embedded Rust/Python API, without new Rocketkeep service/WAL wiring.

## Delivery and validation

Parent: [#832](https://github.com/lance-format/lance-context/issues/832).
Small implementation increments:

| PR | Increment |
| --- | --- |
| [#913](https://github.com/lance-format/lance-context/pull/913) | Baseline: avoid unnecessary historical text backfills |
| [#914](https://github.com/lance-format/lance-context/pull/914) | Official Arrow Variant kernels and automatic selection |
| [#915](https://github.com/lance-format/lance-context/pull/915) | Baseline: residual string search after layout exhaustion/type drift |
| [#916](https://github.com/lance-format/lance-context/pull/916) | Baseline: ID BTree maintenance and projected point reads |
| [#917](https://github.com/lance-format/lance-context/pull/917) | Baseline: byte-targeted batches and streaming Python ingestion |
| [#918](https://github.com/lance-format/lance-context/pull/918) | Persistent Variant store and atomic automatic migration |
| [#919](https://github.com/lance-format/lance-context/pull/919) | Variant queries, native nested text indexes and Python API |

The full combined local check script passed 1,433 Rust tests (418 ignored) and
236 Python tests (11 skipped, one expected failure), plus workspace format/Clippy
and Ruff/Pyright. That full run preceded the final seven-line writer-option fix.
After that fix, all 13 Variant and 18 baseline native tests, both feature-enabled
physical I/O regressions, targeted Clippy and 16 release-extension Python tests
passed. The final acceptance increment separately checks inherited Variant
streaming partial commits and enables the Variant physical I/O test in CI.

Tests cover standard-array interoperability, exact numeric values, literal/empty
keys, missing versus explicit null, snapshots, stale writers, automatic layout
replacement, residual search, native index replacement and byte-targeted writes.
The I/O tests prove that compatible small-field reads avoid unrelated 4 MiB text;
they do not claim that whole-row residual fallback avoids that text.

## Comparable release measurements

All 24 dual-backend runs and 16 additional repair runs passed their assertions.
Both backends used the same release extension, source
`c39e1519` (full revision and binary SHA-256 in the raw metadata), Rust 1.99.0 and
Lance 14.0.2. The merged production source differs only in the placement of the
baseline ingestion module declaration; its contents and dependencies are unchanged.
Each case used a fresh process and dataset. Runs were serial, with
no concurrent local compilation; backend order rotated. This was a shared
16-vCPU Linux host, not an isolated production-capacity test. Thread-count and
Lance-memory environment overrides were unset for both backends.

The dual comparison has three runs per backend/case and 15 warmed read samples
per run. Search has one warmup and five samples. Tables show median run medians;
ingestion and preparation use the median of three single measurements. Raw data
retain every sample, including outliers. Several initial ingestion samples were
much slower (up to 10.123 s); ingestion medians alone are not stability evidence.
The supplemental ID-index scale probe has one fresh process per size and 30
rotating ID samples, not three independent processes per size.

Text repeats a deterministic 4,096-token vocabulary. The resulting extremely
small compressed sizes are **not representative compression predictions**.
Retained directory sizes include all data, indexes, manifests and history at the
end of each case, without cleanup. They are not live-value-only storage counts.
RSS is whole-process peak, including Python fixtures/native conversions; ingestion
RSS is captured before verification reads. Reads use local warm caches, with no
cold-cache or remote-storage claim. JSON equality checks distinguish booleans
from integers and verify reconstructed values for selected rows and snapshots.

| Case | Backend | Ingest s (range) | Full read ms | Small typed path ms | Residual tags path ms | Retained MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10,000 × 128 B | baseline | 0.061 (0.059–5.024) | 0.638 | 0.566 | 0.514 | 1.177 |
| 10,000 × 128 B | variant | 0.100 (0.099–4.800) | 0.916 | 0.789 | 0.643 | 1.189 |
| 8 × 20 MiB | baseline | 0.390 (0.377–10.123) | 28.296 | 4.917 | 0.660 | 0.089 |
| 8 × 20 MiB | variant | 0.510 (0.504–0.530) | 45.637 | 5.001 | 0.785 | 0.093 |
| 8 × 20 MiB, type drift | baseline | 0.379 (0.373–0.387) | 27.805 | 4.936 | 0.668 | 0.089 |
| 8 × 20 MiB, type drift | variant | 0.491 (0.489–0.511) | 45.384 | 5.088 | 0.771 | 0.093 |

For large compatible documents, Variant full reads are 45.637 ms versus
28.296 ms for the baseline (about 1.61×). Its small paths retain sub-full-read
cost, but this implementation does not establish a general read-speed benefit.
The small-row case adds roughly 0.28 ms to a full point read.

### Text search and incompatible rows

The mixed case changes the last record's selected integer `sequence` to boolean
while retaining its 20 MiB text. The strict guard keeps that entire Variant row
in residual storage. The baseline can continue shredding its compatible text.

| Case | Backend | Prepare text s | Unique text match ms | Residual match ms | No match ms | Whole-process peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| small | baseline | 0.055 | 6.139 | 0.905 | 6.066 | 156.6 |
| small | variant | 0.060 | 19.620 | 3.050 | 19.300 | 166.1 |
| large | baseline | 1.607 | 1.073 | 0.731 | 0.715 | 780.1 |
| large | variant | 1.612 | 1.415 | 1.049 | 1.056 | 984.7 |
| mixed | baseline | 1.656 | 1.226 | 0.770 | 0.755 | 780.4 |
| mixed | variant | 1.408 | 154.633 | 153.203 | 154.088 | 946.0 |

The mixed Variant searches take about 153–155 ms versus about 0.8–1.2 ms for the
baseline. A rare or absent hit scans residuals to fill remaining result slots,
so even a native-indexed hit can pay fallback cost. This is a material limitation.
Reading just `sequence` from the incompatible row takes 5.011 ms versus 1.099 ms;
reading its `tags` takes 2.019 ms versus 0.504 ms. These are measured separately
so the compatible-row median cannot hide fallback behavior.

The independent baseline residual-search probe fills its 32-column budget before
adding 10,000 one-KiB text records. Search remains correct: a unique residual match
takes 80.640 ms and no match 78.949 ms, versus 3.915/2.995 ms with one extra budget
slot allowing a native text index. Correct residual coverage is not free.

### Automatic evolution and sparse compression

After 1,000 historical records fill 32 numeric paths, one new record introduces
about 20 MiB of text and changes a numeric field to string. Variant automatically
reselects text paths and atomically rewrites old rows. The baseline retains its
existing bounded columns and searches the new text through residual fallback.
This deliberately compares the actual evolution policies, not identical layouts.

| Backend | Evolving add s | Body selected | First body search s | Retained MiB | Whole-process peak MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline | 0.140 | false | 0.050 | 0.092 | 251.1 |
| variant | 0.145 | true | 0.252 | 0.549 | 315.1 |

Exploratory runs exposed roughly 21.5 MB of retained Variant storage in this case:
historical null children plus a later large string became a raw constant page.
The regression test stores 4 MB of repeated text and fails before the fix at
4,112,059 bytes. Preserving encoded batch boundaries through existing Lance writer
options fixes migration and ordinary append without another vendor edit. The
final 20 MiB case retains about 0.55 MiB. More small pages and historical rewritten
files still cost space: Variant retains more bytes than this baseline scenario.

### ID indexing and streaming memory

The baseline scale probe compares indexed and unindexed snapshots referencing
the same data files. The index adds a separate durable version.

| Rows | Full unindexed ms | Full indexed ms | Residual path unindexed ms | Residual path indexed ms |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1.571 | 0.762 | 1.336 | 0.602 |
| 100,000 | 2.816 | 1.194 | 2.500 | 0.588 |
| 1,000,000 | 9.873 | 1.179 | 8.638 | 0.561 |

Streaming probes use eight 20 MiB records, three fresh processes per backend/mode.
The timer includes fixture generation in both modes. Streaming lowers retained
input/conversion memory but makes multiple commits; it does not preserve the
atomic mode's single-append visibility contract.

| Backend | Mode | Ingest s | Ingestion peak RSS MiB |
| --- | --- | ---: | ---: |
| baseline | atomic | 0.526 | 511.2 |
| baseline | stream | 0.562 | 354.5 |
| variant | atomic | 0.655 | 593.0 |
| variant | stream | 0.688 | 434.9 |

Streaming reduces ingestion peak RSS by about 31% for the baseline and 27% for
Variant in this fixture, with a modest measured latency increase. The native
batch target remains a batching policy, not a process memory limit.

## Reproduction and follow-ups

Raw measurements and reproduction commands are in
[the benchmark evidence directory](../benchmarks/automatic-shredding-dual-2026-11/README.md).
The earlier [2026-11-13 results](../benchmarks/automatic-shredding-2026-11/README.md)
compare the custom baseline with serialized/duplicated JSON; they are not Variant
performance evidence.

Prioritize field-level lossless fallback and reducing Variant-to-JSON conversion
cost before recommending Variant for latency-sensitive mixed-type workloads.
Automatic layout migration is synchronous and can rewrite O(existing data);
production use also needs workload-specific migration policy, history cleanup,
remote-storage validation and service integration. Neither current backend
provides background migration or a production-capacity qualification here.
