# Examples

Runnable, self-contained examples for `lance-context`. Each lives in its own
directory with a `README.md` and its own dependency manifest. Start at the top
and work down.

| Example | What you'll learn | Needs |
|---------|-------------------|-------|
| [`pypi-basic`](./pypi-basic) | Install from PyPI, create a `Context`, add entries, and time-travel through versions. The 5-minute quickstart. | PyPI install only |
| [`multi-session`](./multi-session) | Concurrent, multi-bot writes with **MemWAL** sharding and per-session isolation. | PyPI install only |
| [`eval-quality`](./eval-quality) | Measure retrieval quality with the built-in eval harness (`evaluate` / `evaluate_versions`): labeled query sets, recall/precision/MRR/nDCG, and A/B-ing a change. Fully offline. | PyPI install only |
| [`mcp-claude-code`](./mcp-claude-code) | Expose a context store to an agent over **MCP** and drive it from Claude Code, with a bundled skill and end-to-end tests. | MCP client (e.g. Claude Code) |

## Running an example

Each example is a standalone project. With [`uv`](https://docs.astral.sh/uv/):

```bash
cd examples/pypi-basic
uv run context-demo
```

or with a plain virtualenv:

```bash
cd examples/pypi-basic
python -m venv .venv && source .venv/bin/activate
pip install -e .
context-demo          # see the example's README for its entry point
```

See each example's own `README.md` for the exact command and what to expect.
