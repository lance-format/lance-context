# Variant field-level fallback: release evidence

See the [assessment](../../design/variant-field-fallback-assessment.md).
`dual.json` contains 24 runs (two backends, four scenarios, three repetitions).
`paired-mixed.json` contains six alternating original/repaired Variant runs.
`summary.json` aggregates dual-run medians and retains ranges.

Build the measured source's release Python extension as described in the
[original benchmark guide](../automatic-shredding-dual-2026-11/README.md), then run:

```bash
.venv/bin/python ../scripts/benchmarks/shredding/run.py --suite dual --output /tmp/variant-repaired-dual
.venv/bin/python ../scripts/benchmarks/shredding/summarize.py /tmp/variant-repaired-dual
```

For the paired case, use `dual_probe.py --backend variant --scenario mixed
--rows 8 --body-bytes 20971520 --samples 15 --output NEW_DIRECTORY` with the
original `c39e1519` and repaired release extensions. Use a fresh dataset/process
for each run, alternate backend order, and repeat three times. The raw metadata
records source revisions and native-library hashes. Keep compilation separate
from measurements. The original extension uses the stable abi3 interface; both
arms in this run use the same Python interpreter and dependencies.

All runs have `validated=true`. Raw process RSS includes caller-owned documents;
retained storage includes snapshots and indexes. Synthetic repetitive text and
warm local caching limit generalization to other workloads.
