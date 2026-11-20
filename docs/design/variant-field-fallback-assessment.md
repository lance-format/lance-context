# Arrow Variant field-level fallback repair

The repair in [#939](https://github.com/lance-format/lance-context/pull/939)
keeps compatible sibling fields shredded when another selected field changes type.
For example, `sequence: true` in an integer layout retains the boolean in the
standard field residual while `body` stays in a standard typed child and its text
index. It uses Arrow's shredding, unshredding and value builders; the baseline,
dependencies and Lance vendor sources are unchanged.

## Measurements

Measured on 2026-11-20 (UTC).

Measured source: `5095a05aa96a61f825f47b99c523a945ea1c6ed7`. Release native SHA-256:
`b16592890169f5167565e0fcc49081ad19823b0f27b9986e7b36861b8de56297`. All 30 fresh-process runs validated full values,
projected values and search results. The paired mixed case uses 8 records with
20 MiB of synthetic text each and a boolean in the final record's integer field.
Each process takes 15 full/path samples and 5 samples for each search, after warmup.
The table shows the median of three process medians, with their min–max range.
Times are milliseconds.

| Mixed-type search | Original Variant | Repaired Variant |
| --- | ---: | ---: |
| Last record's text token | 173.742 (165.442–200.714) | 1.555 (1.540–1.887) |
| Residual text token | 161.165 (159.409–181.544) | 1.224 (1.088–1.383) |
| Absent token | 168.075 (158.590–172.249) | 1.165 (1.093–1.255) |

The independent dual suite measures both backends in the same repaired binary.
Its mixed-type last-record search is 1.611 ms baseline versus 2.014 ms Variant
(medians from different processes than the paired table above). The severe
whole-row fallback penalty is removed, but the baseline still has lower latency
in that comparison. Full-record reads still include Variant reconstruction and
conversion:

| Full read | Baseline | Repaired Variant |
| --- | ---: | ---: |
| Small records | 0.748 (0.703–0.904) | 1.157 (1.060–1.302) |
| 20 MiB compatible records | 31.933 (30.926–39.604) | 57.444 (49.835–59.901) |
| 20 MiB mixed-type records | 37.700 (30.625–38.689) | 51.889 (49.136–52.918) |

[Raw evidence](../benchmarks/variant-field-fallback-2026-11/README.md) retains all
samples, ingestion/index preparation costs, memory observations, storage sizes,
and the evolution scenario. Do not infer that Variant is faster for every
operation or should replace the baseline by default.

## Other costs

The same-binary dual suite still favors the baseline for ingestion and memory.
For the compatible 20 MiB case, median ingestion including ID indexing is
0.468 s baseline versus 0.597 s Variant; ingestion peak RSS is 512.7 versus
574.5 MiB. For mixed types, ingestion is 0.474 versus 0.652 s and ingestion peak
RSS is 512.5 versus 593.1 MiB. These peaks include caller-owned Python documents.
Text-index preparation is measured separately in the raw results.

Outliers remain in the evidence: one compatible Variant ingestion took 4.650 s,
and one mixed baseline text-index preparation took 6.290 s. No samples were
removed. Across the paired original/repaired mixed runs, median ingestion is
0.619 versus 0.613 s; this small difference is not evidence of an ingestion
speedup. The fix targets the severe residual search penalty.

## Validation and limits

The exact implementation head passed the full Rust CI job (1,556 passed test
executions, zero failures, 424 ignored; executions are not unique tests), all
16 Variant unit/store cases, and the Python CI suite (245 passed, 2 skipped,
1 expected failure). The measured release extension also passed local Python
validation (236 passed, 11 skipped, 1 expected failure); the extra local skips
are server-dependent cases covered in CI. Workspace format/Clippy, Python
format/lint/type checks and the feature-enabled physical I/O test passed.

Regression tests cover exact type/value reconstruction, missing versus null,
large unsigned integers, nested/literal keys, sliced and already-shredded arrays,
malformed external binary input, and indexed sibling hits after reopening. The
physical I/O regression uses a 4 MiB high-entropy body and a boolean/integer
mismatch: small-field reads and warm no-hit search each fetch less than 1 MiB,
while the full read fetches more than 1 MiB.

These are warm local reads on a shared host. The highly repetitive benchmark text
is useful for reproducing the regression but is not a realistic compression
prediction. The I/O regression uses different, high-entropy text. Three process
repetitions are not a statistical claim about production latency. The paired
comparison rotates original/repaired order and uses the same Python dependencies;
the original extension was built from the previously measured `c39e1519` source.
Variant query/binding code and Lance dependencies between that source and the
repair's base are unchanged; unrelated WAL changes are outside the measured path.

Existing whole-row residuals remain readable, but opening a dataset or preparing
an index does not rewrite them. Newly written rows and later automatic layout
migrations use the repaired shredding behavior. Layout migrations are still
synchronous full rewrites. Residual-only text and unsupported index paths can
still require scanning; this change does not add global BM25 ranking or predicate
pushdown. Full-document conversion remains a separate optimization opportunity.
