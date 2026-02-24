---
name: mcp-lance-context
description: Build or update the lance-context MCP server and Claude Code example that uses lance-context as a memory/knowledge store.
---

# MCP Lance Context

## Overview
Use this skill when adding or updating the MCP server for lance-context or the
Claude Code example that demonstrates memory/knowledge workflows.

## Files to touch
- `examples/mcp-claude-code/server/lance_context_mcp.py`
- `examples/mcp-claude-code/README.md`
- `examples/mcp-claude-code/requirements.txt`

## Workflow
1. Keep MCP tool outputs JSON-serializable (convert datetimes to strings).
2. Add or adjust tools that write to `Context.add()` and read via `Context.list()`.
3. Update the README instructions and verify CLI commands stay accurate.
4. If dependencies change, update `requirements.txt`.

## Claude Code setup
Use project-scoped MCP configuration with `claude mcp add --transport stdio --scope project`.
This writes a `.mcp.json` file in the project directory.
