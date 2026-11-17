"""Equivalent JSON operations against the same compiled native extension.

Use a fresh process and dataset per backend/scenario/repetition. All latency
samples follow one untimed warmup; these are local warm-cache measurements.
Peak RSS is whole-process through ingestion, including caller-owned fixtures.
"""

import argparse
import json
import resource
import statistics
import time
from pathlib import Path
from lance_context import DocumentStore, VariantDocumentStore

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["baseline", "variant"], required=True)
parser.add_argument(
    "--scenario", choices=["point", "large", "mixed", "evolution"], required=True
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--body-bytes", type=int, default=128)
parser.add_argument("--samples", type=int, default=20)
args = parser.parse_args()
backend = DocumentStore if args.backend == "baseline" else VariantDocumentStore
store = backend.create(args.output)

patterns = [
    " ".join(f"word{(j * 17 + seed) % 4096}" for j in range(4096)) + " "
    for seed in range(8)
]


def document(i):
    pattern = patterns[i % len(patterns)]
    text = (pattern * (args.body_bytes // len(pattern) + 1))[: args.body_bytes]
    sequence = True if args.scenario == "mixed" and i == args.rows - 1 else i
    return {
        "sequence": sequence,
        "body": text + f" recordtoken{i}",
        "tags": ["residualword"],
    }


def equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equal(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


result = vars(args).copy()
result["output"] = str(args.output)
if args.scenario == "evolution":
    seed = {f"field{i}": i for i in range(32)}
    historical = [dict(seed, field0=i) for i in range(args.rows)]
    historical_ids = store.add(historical)
    first = historical_ids[0]
    version = store.version()
    start = time.perf_counter()
    record = {
        "body": "lateotter " * max(1, args.body_bytes // 10),
        "field0": "typedrift",
    }
    last = store.add([record])[0]
    result["evolving_add_seconds"] = time.perf_counter() - start
    result["body_selected"] = any(
        field["path"] == ["body"] for field in store.layout()["fields"]
    )
    assert equal(store.get(first), seed) and equal(store.get(last), record)
    result["search_seconds"] = {}
    for query in ["lateotter", "typedrift"]:
        start = time.perf_counter()
        assert last in [hit["id"] for hit in store.search(query)]
        result["search_seconds"][query] = time.perf_counter() - start
    store.checkout(version)
    assert equal(store.get(first), seed)
    store.refresh()
else:
    documents = [document(i) for i in range(args.rows)]
    start = time.perf_counter()
    ids = store.add(documents)
    result["ingest_seconds_including_id_index"] = time.perf_counter() - start
    result["ingestion_process_peak_rss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss
    # Verify complete values before timing, including residual containers.
    for i in {0, args.rows // 2, args.rows - 1}:
        assert equal(store.get(ids[i]), documents[i])
    samples = {"full_read_ms": [], "typed_path_ms": [], "residual_path_ms": []}
    warmup_position = args.rows // 2
    assert equal(store.get(ids[warmup_position]), documents[warmup_position])
    assert equal(
        store.get_path(ids[warmup_position], ["sequence"]),
        documents[warmup_position]["sequence"],
    )
    assert store.get_path(ids[warmup_position], ["tags"]) == ["residualword"]
    for i in range(args.samples):
        position = (i * 104729 + args.rows // 2) % args.rows
        for name, operation, expected in [
            ("full_read_ms", lambda: store.get(ids[position]), documents[position]),
            (
                "typed_path_ms",
                lambda: store.get_path(ids[position], ["sequence"]),
                documents[position]["sequence"],
            ),
            (
                "residual_path_ms",
                lambda: store.get_path(ids[position], ["tags"]),
                ["residualword"],
            ),
        ]:
            start = time.perf_counter()
            actual = operation()
            elapsed = (time.perf_counter() - start) * 1000
            assert equal(actual, expected)
            samples[name].append(elapsed)
    result["latency"] = {
        name: {"median": statistics.median(values), "samples": values}
        for name, values in samples.items()
    }
    if args.scenario == "mixed":
        result["incompatible_row_paths"] = {}
        for path in [["sequence"], ["tags"]]:
            expected = documents[-1][path[0]]
            assert equal(store.get_path(ids[-1], path), expected)
            latencies = []
            for _ in range(args.samples):
                start = time.perf_counter()
                actual = store.get_path(ids[-1], path)
                latencies.append((time.perf_counter() - start) * 1000)
                assert equal(actual, expected)
            result["incompatible_row_paths"][path[0]] = {
                "median_ms": statistics.median(latencies),
                "samples_ms": latencies,
            }
    start = time.perf_counter()
    store.prepare_text_search()
    result["prepare_text_seconds"] = time.perf_counter() - start
    result["search"] = {}
    for query in [f"recordtoken{args.rows - 1}", "residualword", "absentneedle"]:
        store.search(query)
        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            hits = store.search(query)
            latencies.append(time.perf_counter() - start)
        result["search"][query] = {
            "median_seconds": statistics.median(latencies),
            "samples_seconds": latencies,
            "count": len(hits),
        }
        if query.startswith("recordtoken"):
            assert [hit["id"] for hit in hits] == [ids[-1]]
        elif query == "residualword":
            assert len(hits) == min(10, args.rows)
        else:
            assert hits == []
result["retained_directory_bytes"] = sum(
    path.stat().st_size for path in args.output.rglob("*") if path.is_file()
)
result["final_process_peak_rss_kib"] = resource.getrusage(
    resource.RUSAGE_SELF
).ru_maxrss
result["version"] = store.version()
result["validated"] = True
result["cache"] = "local warm cache; no cold/remote-storage claim"
print(json.dumps(result))
