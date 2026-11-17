"""Run both shredding backends in serial fresh processes using one native build.

Run with the Python environment containing the release lance-context extension
and pylance. Do not compile or run another benchmark concurrently.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--suite", choices=["dual", "repairs"], default="dual")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
scripts = Path(__file__).resolve().parent
repo = scripts.parents[2]
binary = Path(
    subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import lance_context._internal as native; print(native.__file__)",
        ],
        text=True,
    ).strip()
)
with binary.open("rb") as source:
    digest = hashlib.file_digest(source, "sha256").hexdigest()
metadata = {
    "python": sys.executable,
    "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip(),
    "native_library": str(binary),
    "native_sha256": digest,
    "load_average_at_start": os.getloadavg(),
    "suite": args.suite,
    "environment": "serial fresh processes; synthetic repetitive text; local warm cache",
}
(args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))


def run(name, script, arguments):
    result_path = args.output / (name + ".json")
    with (
        result_path.open("w") as result,
        (args.output / (name + ".stderr")).open("w") as error,
    ):
        subprocess.run(
            [
                sys.executable,
                str(scripts / script),
                "--output",
                str(args.output / (name + ".lance")),
                *arguments,
            ],
            stdout=result,
            stderr=error,
            check=True,
        )
    assert json.loads(result_path.read_text())["validated"]
    print(name + ": validated", flush=True)


if args.suite == "dual":
    for repetition in range(3):
        arms = (
            ["baseline", "variant"] if repetition % 2 == 0 else ["variant", "baseline"]
        )
        for scenario, rows, body in [
            ("point", 10000, 128),
            ("large", 8, 20 * 1024 * 1024),
            ("mixed", 8, 20 * 1024 * 1024),
            ("evolution", 1000, 20 * 1024 * 1024),
        ]:
            for backend in arms:
                run(
                    f"{scenario}-{backend}-{repetition}",
                    "dual_probe.py",
                    [
                        "--backend",
                        backend,
                        "--scenario",
                        scenario,
                        "--rows",
                        str(rows),
                        "--body-bytes",
                        str(body),
                        "--samples",
                        "15",
                    ],
                )
else:
    for repetition in range(3):
        arms = [
            ("baseline", "atomic"),
            ("variant", "atomic"),
            ("baseline", "stream"),
            ("variant", "stream"),
        ]
        if repetition % 2:
            arms.reverse()
        for backend, mode in arms:
            run(
                f"ingestion-{backend}-{mode}-{repetition}",
                "ingestion_probe.py",
                ["--backend", backend, "--mode", mode],
            )
    for rows in [10000, 100000, 1000000]:
        run(f"point-{rows}", "point_read_probe.py", ["--rows", str(rows)])
    run("residual-search", "residual_search_probe.py", [])
(args.output / "COMPLETE").write_text("All runs completed and validated.\n")
