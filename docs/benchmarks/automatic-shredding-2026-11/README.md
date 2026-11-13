# Automatic shredding: release benchmark results

Measured on 2026-11-13. The benchmark is in [#869](https://github.com/lance-format/lance-context/pull/869); the parent roadmap is [#832](https://github.com/lance-format/lance-context/issues/832).

The fixes remove two measured regressions: residual filters no longer fetch unrelated extracted text, and a one-row backfill page no longer expands compressed large text into a raw constant value. In the 8 × 20 MiB case, the filter median falls from 247.086 ms to 0.430 ms and requests 720 bytes instead of 39,797,813 bytes. Current data after maintenance falls from 160.004 MiB to 37.957 MiB. All 54 baseline/candidate runs passed their reconstruction, projection, filtering, search, historical promotion and reopen assertions.

The resulting tradeoffs remain workload-dependent:

- Current data is about half the duplicated-JSON arm, including after backfill. Full-document reads are 28.312 ms versus 31.857 ms for the huge case; body-only reads are approximately 28.1 ms in both arms. Small/medium full reads retain roughly 0.2–0.3 ms of overhead versus duplicated JSON.
- First FTS preparation still rewrites the dataset synchronously. For huge records it takes 3.356 s, versus 1.764 s for duplicated JSON, and layout-growth preparation takes another 3.460 s. Compression fixes storage expansion but increases this growth-phase median versus the old raw-page rewrite (2.121 s).
- Retaining old data and index versions means the huge final directory is 114.272 MiB, versus 76.107 MiB for duplicated JSON. This experiment performs no history cleanup. Live-record savings therefore do not imply lower retained-directory usage.
- Huge-case process peak RSS falls from 1,405.055 MiB before the fixes to 1,045.512 MiB, but remains above the duplicated arm's 731.473 MiB. Medium-case RSS is also higher than before (298.090 versus 267.711 MiB). Batching targets are not process memory caps.

This completes a local synthetic comparison and regression check, not a production capacity validation. First-search maintenance, history retention, point-read indexing, and background/WAL service integration need workload-specific treatment.

## Workload and build

| Case | Records | Text bytes per record | Total input text |
|---|---:|---:|---:|
| small | 2,048 | 1,024 | 2 MiB |
| medium | 128 | 262,144 | 32 MiB |
| huge | 8 | 20,971,520 | 160 MiB |

Each record has a dominant `body`, a numeric `sequence`, and a residual `tags` array. The tags predicate matches the last row, forcing a full scan. A rare historical `late_text` is promoted by a later append. Duplicated JSON materializes both text columns upfront, so its growth preparation is a no-op; automatic shredding discovers the second text path and must prepare historical data. These are deliberately separate preparation workloads, not equivalent schema-evolution implementations.

Both baseline arms use serialized JSON in `LargeUtf8`, not Lance JSONB or Arrow Variant. A deterministic 4,096-token synthetic vocabulary determines compression and tokenization costs. The two benchmark binaries each contain all three representations. They share Rust 1.99.0, release optimization, Lance 14.0.2 and the same dependency tree; the candidate changes document scanning and the singleton encoding guard. Baseline production source is `decdc4411`, with benchmark source subsequently committed as `fa5fe9a7`; the measured candidate is `abd6ba4e3238e80dcb0faff40112e3d1cec646e4`. Exact revisions, compiler versions and binary SHA-256 values are in the raw files.

The shared Linux host exposes 16 CPUs with a 15-CPU cgroup quota and 56 GiB memory limit. Each process uses four Tokio workers, `LANCE_CPU_THREADS=4`, `RAYON_NUM_THREADS=4`, `LANCE_IO_THREADS=8` and `LANCE_MEM_POOL_SIZE=1073741824`. Runs are sequential with no concurrent local compilation. Arm order rotates across repetitions: JSON/duplicated/shredded, duplicated/shredded/JSON, shredded/JSON/duplicated.

Reads use warm OS caches. Reopen resets handle caches, not the OS cache; no cold-disk result is claimed. No arm has an ID index. I/O counters report requested object-store bytes, not device traffic. RSS includes fixture generation, ingestion, validation and index construction for the entire fresh process, so it is not a per-operation allocation measurement. JSON-only FTS has no equivalent native index and is omitted. Index bytes include retained obsolete indexes; current-data bytes include only files referenced by the latest manifest.

Measurements use three independent processes per case and representation, each with 15 latency samples. Tables show the median of the three run medians (or the median of the three single preparation/ingestion measurements). Raw samples and storage counters are in [baseline.json](baseline.json) and [candidate.json](candidate.json).

## Candidate query and ingestion times

| Case | Representation | Ingest ms | Full read p50 ms | Body read p50 ms | Residual filter p50 ms | FTS p50 ms |
|---|---|---:|---:|---:|---:|---:|
| small | json | 20.817 | 0.901 | 0.841 | 2.616 | — |
| small | duplicated | 23.249 | 0.877 | 0.865 | 2.720 | 0.866 |
| small | shredded | 19.274 | 1.090 | 1.003 | 1.204 | 0.637 |
| medium | json | 260.235 | 1.200 | 1.247 | 93.022 | — |
| medium | duplicated | 297.402 | 1.175 | 1.123 | 90.375 | 0.883 |
| medium | shredded | 202.122 | 1.460 | 1.245 | 0.438 | 0.624 |
| huge | json | 1,313.678 | 31.735 | 31.507 | 495.566 | — |
| huge | duplicated | 1,446.582 | 31.857 | 28.128 | 481.986 | 0.866 |
| huge | shredded | 997.002 | 28.312 | 28.104 | 0.430 | 0.622 |

## Candidate preparation, retention and memory

| Case | Representation | Initial FTS prep ms | Growth prep ms | Final current data MiB | All retained data MiB | All index MiB | Total directory MiB | Process peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | json | — | — | 0.603 | 0.603 | 0.000 | 0.604 | 99.605 |
| small | duplicated | 52.379 | 0.005 | 1.146 | 1.146 | 0.578 | 1.729 | 124.395 |
| small | shredded | 72.194 | 73.346 | 0.580 | 1.728 | 1.169 | 2.913 | 130.707 |
| medium | json | — | — | 7.759 | 7.759 | 0.000 | 7.759 | 199.574 |
| medium | duplicated | 423.063 | 0.005 | 15.508 | 15.508 | 0.534 | 16.047 | 248.508 |
| medium | shredded | 722.470 | 722.994 | 7.757 | 23.270 | 1.066 | 24.352 | 298.090 |
| huge | json | — | — | 37.955 | 37.955 | 0.000 | 37.956 | 700.309 |
| huge | duplicated | 1,764.265 | 0.007 | 75.910 | 75.910 | 0.193 | 76.107 | 731.473 |
| huge | shredded | 3,356.300 | 3,459.980 | 37.957 | 113.872 | 0.384 | 114.272 | 1,045.512 |

## Shredded before/after

| Case | Build | Residual filter p50 ms | Filter requested bytes (median sample) | Final current data MiB | Initial FTS prep ms | Growth prep ms |
|---|---|---:|---:|---:|---:|---:|
| small | baseline | 96.170 | 587,288.000 | 0.577 | 196.412 | 225.025 |
| small | candidate | 1.204 | 33,736.000 | 0.580 | 72.194 | 73.346 |
| medium | baseline | 50.185 | 8,130,896.000 | 7.757 | 678.626 | 691.677 |
| medium | candidate | 0.438 | 5,280.000 | 7.757 | 722.470 | 722.994 |
| huge | baseline | 247.086 | 39,797,813.000 | 160.004 | 3,170.766 | 2,121.473 |
| huge | candidate | 0.430 | 720.000 | 37.957 | 3,356.300 | 3,459.980 |

## Candidate read tails and reopen

These are medians of the three per-run p95 observations (15 samples per run); they are descriptive and have limited tail resolution. Reopen is the median of one first read per process, with a warm OS cache.

| Case | Representation | Full read p95 ms | Body read p95 ms | Filter p95 ms | FTS p95 ms | First read after reopen ms |
|---|---|---:|---:|---:|---:|---:|
| small | json | 1.188 | 1.007 | 3.835 | — | 1.226 |
| small | duplicated | 1.011 | 0.994 | 3.900 | 1.126 | 1.266 |
| small | shredded | 1.182 | 1.079 | 1.341 | 0.696 | 1.638 |
| medium | json | 1.841 | 1.576 | 102.655 | — | 1.907 |
| medium | duplicated | 1.882 | 1.282 | 109.668 | 1.026 | 1.702 |
| medium | shredded | 1.720 | 1.704 | 0.570 | 0.735 | 1.909 |
| huge | json | 48.568 | 38.540 | 530.829 | — | 31.769 |
| huge | duplicated | 48.864 | 29.148 | 532.274 | 0.941 | 30.967 |
| huge | shredded | 46.706 | 30.155 | 0.837 | 0.687 | 34.005 |

## Reproduce

At each source revision, build the release example with `cargo +1.99.0 build --locked --release -p lance-context-core --example automatic_shredding --features lance-io-metrics`. Use the worker environment above, fresh output directories and three repetitions of each case/arm. For example:

```bash
LANCE_CPU_THREADS=4 RAYON_NUM_THREADS=4 LANCE_IO_THREADS=8 \
LANCE_MEM_POOL_SIZE=1073741824 \
target/release/examples/automatic_shredding --mode shredded \
  --output /tmp/shredded-huge-repeat-1 --rows 8 --body-bytes 20971520 \
  --samples 15 --threads 4
```

Substitute `json` and `duplicated` and the case dimensions above; rotate arm order and preserve each `result.json`. If `CARGO_TARGET_DIR` is set, use its release executable. See the [benchmark design](../../design/automatic-shredding.md#reproducible-performance-comparison) for metric definitions and validation behavior. The raw files here preserve individual samples for all 54 runs rather than only the summary tables.
