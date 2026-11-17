# Repaired baseline and Arrow Variant: release evidence

Measured on 2026-11-17. Read the
[assessment](../../design/automatic-shredding-dual-assessment.md) for conclusions,
methodology, limitations and the implementation PRs.

- [dual.json](dual.json): 24 fresh-process runs, two backends × four cases × three
  repetitions. Every read/search sample and storage/memory result is retained.
- [repairs.json](repairs.json): 12 atomic/streaming runs, indexed versus unindexed
  point reads at 10k/100k/1M rows, and residual-search coverage/cost.
- [summary.json](summary.json): dual-run median/min/max aggregates; it does not
  replace the raw samples.

Both files record the measured source revision and release native library SHA-256.
The native source is `c39e1519` on
[`beinan/lance-context`](https://github.com/beinan/lance-context/commit/c39e1519).
The implementation was assembled from the small PR increments for equivalent
testing; this measurement revision is not a claim that those PRs were merged.
The probe code is in [scripts/benchmarks/shredding](../../../scripts/benchmarks/shredding).

Build the current checkout's release native extension with the project's Python
test environment, including `pylance` for index inspection. For example, from the
repository root:

```bash
cd python
uv sync --extra tests --extra dev --no-install-project
VIRTUAL_ENV="$PWD/.venv" uv tool run --from maturin maturin develop --release
uv pip install --python .venv/bin/python pylance
.venv/bin/python ../scripts/benchmarks/shredding/run.py --suite dual --output /tmp/shredding-dual-new
.venv/bin/python ../scripts/benchmarks/shredding/summarize.py /tmp/shredding-dual-new
.venv/bin/python ../scripts/benchmarks/shredding/run.py --suite repairs --output /tmp/shredding-repairs-new
```

Each output directory must be new. The runner records source identity, hashes the
native extension without retaining the whole binary in parent memory, and rotates
backend order. It retains datasets, JSON results and per-process stderr; remove
those temporary datasets only after saving the results you need. Run the suites
serially without concurrent compilation. These commands measure the checkout you
build; they do not automatically restore the historical measurement revision.

All 40 measured runs completed and passed their assertions. The text is highly
repetitive synthetic data, so its extreme compression ratios are not representative.
Ingestion includes ID-index maintenance; streaming includes multiple commits.
Whole-process RSS includes caller-owned Python objects. Warm local reads are not
remote-storage or cold-cache measurements. Some ingestion samples are large
outliers; the raw results and assessment preserve their ranges.

The two earlier exploratory runs are excluded: one was incomplete and inflated
child RSS by reading the native library into parent memory; the second overlapped
compilation and preceded the sparse-value compression fix. Neither supplies
numbers in these final files.
