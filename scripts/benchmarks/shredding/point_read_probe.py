"""Compare indexed/unindexed snapshots over identical files in a fresh process."""

import argparse
import json
import statistics
import time
from pathlib import Path
import lance
from lance_context import DocumentStore

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--samples", type=int, default=30)
args = parser.parse_args()
store = DocumentStore.create(args.output)
start = time.perf_counter()
ids = store.add(
    [
        {"sequence": i, "body": f"document number {i}", "tags": ["small"]}
        for i in range(args.rows)
    ]
)
ingest = time.perf_counter() - start
indexed_version = store.version()
unindexed_version = indexed_version - 1
assert any(
    index["name"] == "document_id_btree"
    for index in lance.dataset(str(args.output), version=indexed_version).list_indices()
)
assert not any(
    index["name"] == "document_id_btree"
    for index in lance.dataset(
        str(args.output), version=unindexed_version
    ).list_indices()
)
results = {}
for label, version in [("unindexed", unindexed_version), ("indexed", indexed_version)]:
    store.checkout(version)
    store.get(ids[-1])
    full, path = [], []
    for i in range(args.samples):
        position = (i * 104729 + args.rows // 2) % args.rows
        start = time.perf_counter()
        value = store.get(ids[position])
        full.append((time.perf_counter() - start) * 1000)
        assert value["sequence"] == position
        start = time.perf_counter()
        value = store.get_path(ids[position], ["tags"])
        path.append((time.perf_counter() - start) * 1000)
        assert value == ["small"]
    results[label] = {
        "full_read_median_ms": statistics.median(full),
        "residual_path_median_ms": statistics.median(path),
        "full_samples_ms": full,
        "path_samples_ms": path,
    }
print(
    json.dumps(
        {
            "rows": args.rows,
            "ingest_seconds_including_id_index": ingest,
            "samples": args.samples,
            "warm_cache": True,
            "results": results,
            "validated": True,
        }
    )
)
