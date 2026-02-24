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

Now you can use tools like `add_entry`, `list_entries`, `search_entries`,
`stats`, and `checkout_version` inside Claude Code.

## Skill example

For teams using Codex skills, an example skill definition lives at:

```
examples/mcp-claude-code/skills/mcp-lance-context
```

It is provided as a reference for how to document MCP tooling and Claude Code
setup in a reusable skill.

## Resetting the store

To start fresh, stop the server and delete `.artifacts/claude_context.lance`.
