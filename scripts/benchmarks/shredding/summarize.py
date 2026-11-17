import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
assert (root / "COMPLETE").exists()
records = [
    json.loads(p.read_text()) for p in root.glob("*.json") if p.name != "metadata.json"
]
assert len(records) == 24 and all(x["validated"] for x in records)
summary = {}
for scenario in ["point", "large", "mixed", "evolution"]:
    summary[scenario] = {}
    for backend in ["baseline", "variant"]:
        runs = [
            x for x in records if x["scenario"] == scenario and x["backend"] == backend
        ]
        assert len(runs) == 3

        def aggregate(values):
            return {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "runs": values,
            }

        result = {
            "retained_mib": aggregate(
                [x["retained_directory_bytes"] / 2**20 for x in runs]
            ),
            "process_peak_mib": aggregate(
                [x["final_process_peak_rss_kib"] / 1024 for x in runs]
            ),
        }
        if scenario == "evolution":
            result["evolving_add_seconds"] = aggregate(
                [x["evolving_add_seconds"] for x in runs]
            )
            result["body_selected"] = [x["body_selected"] for x in runs]
            result["first_search_seconds"] = {
                query: aggregate([x["search_seconds"][query] for x in runs])
                for query in runs[0]["search_seconds"]
            }
        else:
            result["ingest_seconds_including_id_index"] = aggregate(
                [x["ingest_seconds_including_id_index"] for x in runs]
            )
            result["ingestion_peak_mib"] = aggregate(
                [x["ingestion_process_peak_rss_kib"] / 1024 for x in runs]
            )
            result["latency_ms"] = {
                name: aggregate([x["latency"][name]["median"] for x in runs])
                for name in runs[0]["latency"]
            }
            result["prepare_text_seconds"] = aggregate(
                [x["prepare_text_seconds"] for x in runs]
            )
            result["search_seconds"] = {
                query: aggregate([x["search"][query]["median_seconds"] for x in runs])
                for query in runs[0]["search"]
            }
            if scenario == "mixed" and "incompatible_row_paths" in runs[0]:
                result["incompatible_row_ms"] = {
                    path: aggregate(
                        [x["incompatible_row_paths"][path]["median_ms"] for x in runs]
                    )
                    for path in runs[0]["incompatible_row_paths"]
                }
        summary[scenario][backend] = result
print(json.dumps(summary, indent=2))
