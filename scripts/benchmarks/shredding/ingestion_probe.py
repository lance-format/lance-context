"""Run each mode in a fresh process against the same rebuilt native extension."""

import argparse
import json
import resource
import time
from pathlib import Path
from lance_context import DocumentStore, VariantDocumentStore

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["baseline", "variant"], default="baseline")
parser.add_argument("--mode", choices=["atomic", "stream"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rows", type=int, default=8)
parser.add_argument("--body-bytes", type=int, default=20 * 1024 * 1024)
args = parser.parse_args()


def document(i):
    # Deterministic varied text; compression and tokenization are synthetic.
    pattern = " ".join(f"word{(j * 17 + i) % 4096}" for j in range(4096)) + " "
    text = (pattern * (args.body_bytes // len(pattern) + 1))[: args.body_bytes]
    return {"sequence": i, "body": text, "tags": ["ingestion"]}


backend = DocumentStore if args.backend == "baseline" else VariantDocumentStore
store = backend.create(args.output)
start = time.perf_counter()
if args.mode == "atomic":
    ids = store.add([document(i) for i in range(args.rows)])
else:
    ids = []
    for batch_ids in store.add_iter(document(i) for i in range(args.rows)):
        ids.extend(batch_ids)
elapsed = time.perf_counter() - start
write_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
for i, record_id in enumerate(ids):
    assert store.get_path(record_id, ["sequence"]) == i
assert store.get(ids[-1]) == document(args.rows - 1)
print(
    json.dumps(
        {
            "backend": args.backend,
            "mode": args.mode,
            "rows": args.rows,
            "body_bytes": args.body_bytes,
            "ingest_seconds": elapsed,
            "ingestion_peak_rss_kib": write_rss,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "version": store.version(),
            "validated": True,
        }
    )
)
