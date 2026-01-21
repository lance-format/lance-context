# Lance Context PyPI Example

This example project demonstrates how to install the `lance-context` package from PyPI and work with a context store using the Python API.

## Setup

Ensure Python 3.11 or newer is available locally.

### Using uv

```bash
cd examples/pypi-basic
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Using pip

```bash
cd examples/pypi-basic
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the demo

```bash
uv run context-demo
# or
python -m context_example.main
```

The script creates a Lance dataset under `.artifacts/` (ignored by git) and appends a short travel-planning conversation. It prints the current version, demonstrates time-travel by checking out an earlier version, and shows how you can reopen the dataset path to continue appending in future runs.
