# Lance Context MCP + Claude Code Example

This example runs an MCP server that exposes a `lance-context` store as a
simple memory and knowledge base for Claude Code.

## Setup

Ensure Python 3.11+ is available locally.

### Using uv

```bash
cd examples/mcp-claude-code
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Using pip

```bash
cd examples/mcp-claude-code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the MCP server

```bash
python server/lance_context_mcp.py --uri .artifacts/claude_context.lance
```

The server creates a Lance dataset under `.artifacts/` if it doesn't exist.

## Connect Claude Code

From the example directory, add the MCP server to Claude Code (project scope):

```bash
claude mcp add --transport stdio --scope project lance-context -- \
  .venv/bin/python server/lance_context_mcp.py --uri .artifacts/claude_context.lance
```

This writes a `.mcp.json` file in the current directory. Verify with:

```bash
claude mcp list
```

If you launch Claude Code via `aifx agent run claude`, run it from this
directory after adding the MCP server so it can pick up the project-scoped
configuration.

Now you can use tools like `add_entry`, `list_entries`, `search_entries`,
`stats`, and `checkout_version` inside Claude Code.

## End-to-End Workflow (Claude Code + lance-context)

This walkthrough uses Claude Code plus the lance-context MCP server to fix a
real packaging build failure in a UV project.

### Step 1: Reproduce the build failure

From the example root:

```bash
cd examples/mcp-claude-code/e2e/uv-build-broken
uv build
```

Expected failure (excerpt):

```text
OSError: Readme file does not exist: README.md
```

### Step 2: Launch Claude Code with MCP access

Ensure the MCP server is running (see "Run the MCP server"). Then start Claude
Code from `examples/mcp-claude-code` so it picks up `.mcp.json`:

```bash
aifx agent run claude
```

### Step 3: Ask Claude to fix the build and store context

Example prompt (paste into Claude Code):

```text
We need to fix `examples/mcp-claude-code/e2e/uv-build-broken`.
1) Run `uv build` to confirm the failure.
2) Store the error in lance-context with `add_entry` (session_id: build).
3) Fix the project so `uv build` succeeds.
4) Re-run `uv build`, then store the fix summary in lance-context (session_id: build).
```

### Step 4: Verify the fix

After Claude updates the project, run:

```bash
uv build
```

The build should succeed. The expected fix is to point the `readme` field in
`pyproject.toml` at `docs/README.md`. A reference solution lives at:

```
examples/mcp-claude-code/e2e/uv-build-fixed
```

## MCP Smoke Test (Optional)

If you want to verify the MCP server programmatically without Claude Code:

```bash
python mcp_smoke_test.py --keep
```

This script drives the MCP server over stdio, writes sample entries, and keeps
the dataset so you can inspect `.artifacts/e2e_context.lance`.

## Skill example

For teams using Codex skills, an example skill definition lives at:

```
examples/mcp-claude-code/skills/mcp-lance-context
```

It is provided as a reference for how to document MCP tooling and Claude Code
setup in a reusable skill.

## Resetting the store

To start fresh, stop the server and delete `.artifacts/claude_context.lance`.
