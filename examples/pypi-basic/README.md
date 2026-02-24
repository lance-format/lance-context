# Lance Context PyPI Example

This example demonstrates how to install the `lance-context` package (0.2.4 or newer) from PyPI and work with a context store using the Python API.

## Setup

Ensure Python 3.11 or newer is available locally.
This example runs directly from the `src/` directory and only installs `lance-context` from PyPI.

### Using uv

```bash
cd examples/pypi-basic
uv venv
source .venv/bin/activate
uv pip install "lance-context>=0.2.4"
```

### Using pip

```bash
cd examples/pypi-basic
python -m venv .venv
source .venv/bin/activate
pip install "lance-context>=0.2.4"
```

## Run the demo

```bash
python src/context_example/main.py
```

The script creates a Lance dataset under `.artifacts/` (ignored by git) and appends a short travel-planning conversation. It prints the current version, demonstrates time-travel by checking out an earlier version, and shows how you can reopen the dataset path to continue appending in future runs.
