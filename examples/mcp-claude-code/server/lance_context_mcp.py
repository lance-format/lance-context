from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from lance_context import Context
from mcp.server.fastmcp import FastMCP

DEFAULT_DB_NAME = "claude_context.lance"

mcp = FastMCP("Lance Context")

_CTX: Context | None = None


def _default_uri() -> str:
    example_root = Path(__file__).resolve().parents[1]
    artifacts_dir = example_root / ".artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return str(artifacts_dir / DEFAULT_DB_NAME)


def _require_context() -> Context:
    if _CTX is None:
        raise RuntimeError("Context not initialized. Call init_context() first.")
    return _CTX


def init_context(uri: str) -> Context:
    global _CTX
    if _CTX is None:
        _CTX = Context.create(uri)
    return _CTX


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    created_at = record.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    return {
        "id": record.get("id"),
        "role": record.get("role"),
        "text": record.get("text"),
        "content_type": record.get("content_type"),
        "created_at": created_at,
        "session_id": record.get("session_id"),
        "bot_id": record.get("bot_id"),
    }


def _stats(ctx: Context) -> dict[str, Any]:
    return {
        "uri": ctx.uri(),
        "branch": ctx.branch(),
        "entries": ctx.entries(),
        "version": ctx.version(),
    }


@mcp.tool()
def add_entry(
    role: str,
    content: str,
    *,
    session_id: str | None = None,
    bot_id: str | None = None,
) -> dict[str, Any]:
    """Store a memory or knowledge entry in the context store."""
    ctx = _require_context()
    ctx.add(role, content, session_id=session_id, bot_id=bot_id)
    return _stats(ctx)


@mcp.tool()
def list_entries(limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
    """List stored entries for recall."""
    ctx = _require_context()
    records = ctx.list(limit=limit, offset=offset)
    return [_serialize_record(record) for record in records]


@mcp.tool()
def search_entries(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Naive text search over stored entries (case-insensitive substring match)."""
    ctx = _require_context()
    needle = query.casefold()
    matches: list[dict[str, Any]] = []
    for record in ctx.list():
        text = record.get("text") or ""
        if needle in text.casefold():
            matches.append(_serialize_record(record))
            if len(matches) >= limit:
                break
    return matches


@mcp.tool()
def checkout_version(version_id: int) -> dict[str, Any]:
    """Checkout a prior version of the context store."""
    ctx = _require_context()
    ctx.checkout(version_id)
    return _stats(ctx)


@mcp.tool()
def stats() -> dict[str, Any]:
    """Return dataset stats (entries, version, branch, uri)."""
    ctx = _require_context()
    return _stats(ctx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCP server exposing lance-context as memory and knowledge base."
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("LANCE_CONTEXT_URI"),
        help="Lance context dataset URI (defaults to .artifacts/claude_context.lance)",
    )
    args = parser.parse_args()

    uri = args.uri or _default_uri()
    init_context(uri)
    mcp.run()


if __name__ == "__main__":
    main()
