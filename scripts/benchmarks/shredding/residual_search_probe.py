"""Measure the disclosed indexed-first residual-search fallback cost."""

import argparse
import json
import statistics
import time
from pathlib import Path
from lance_context import DocumentStore

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rows", type=int, default=10000)
parser.add_argument("--body-bytes", type=int, default=1024)
parser.add_argument("--samples", type=int, default=5)
args = parser.parse_args()
results = {}
for label, budget in [("indexed", 33), ("residual", 32)]:
    store = DocumentStore.create(args.output / label, policy={"max_columns": budget})
    store.add([{f"metadata_{i:02}": i for i in range(32)}])
    documents = [
        {
            "body": ("common payload " * (args.body_bytes // 15 + 1))[: args.body_bytes]
            + (" uniqueneedle" if i == args.rows - 1 else "")
        }
        for i in range(args.rows)
    ]
    ids = store.add(documents)
    store.prepare_text_search()
    assert any(field["path"] == ["body"] for field in store.layout()["fields"]) == (
        label == "indexed"
    )
    cases = {}
    for query in ("uniqueneedle", "nonexistentneedle"):
        samples = []
        for _ in range(args.samples):
            start = time.perf_counter()
            hits = store.search(query, limit=10)
            samples.append((time.perf_counter() - start) * 1000)
            assert [hit["id"] for hit in hits] == (
                [ids[-1]] if query == "uniqueneedle" else []
            )
        cases[query] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    results[label] = cases
print(
    json.dumps(
        {
            "rows": args.rows,
            "body_bytes": args.body_bytes,
            "limit": 10,
            "results": results,
            "cache": "first sample unwarmed; later samples warm",
            "validated": True,
        }
    )
)
