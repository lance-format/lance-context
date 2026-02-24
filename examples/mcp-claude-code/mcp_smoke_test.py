from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class ToolError(RuntimeError):
    pass


def _example_root() -> Path:
    return Path(__file__).resolve().parent


def _default_uri() -> Path:
    return _example_root() / ".artifacts" / "e2e_context.lance"


def _coerce_text_blocks(blocks: Iterable[Any]) -> str | None:
    for block in blocks:
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            return block.text
    return None


def _payload_from_result(result: Any) -> Any | None:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    text = _coerce_text_blocks(getattr(result, "content", []))
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _require_dict(payload: Any, tool_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ToolError(f"{tool_name} returned unexpected payload: {payload!r}")
    return payload


def _extract_list(payload: Any, tool_name: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("result"), list):
        return payload["result"]
    raise ToolError(f"{tool_name} returned unexpected payload: {payload!r}")


async def _call_tool(session: ClientSession, name: str, arguments: dict[str, Any] | None = None) -> Any:
    result = await session.call_tool(name, arguments)
    if result.isError:
        message = _coerce_text_blocks(result.content) or "unknown error"
        raise ToolError(f"{name} failed: {message}")
    return result


def _print_step(title: str) -> None:
    print(f"- {title}")


async def run_e2e(uri: Path, keep: bool) -> None:
    if uri.exists() and not keep:
        shutil.rmtree(uri)

    uri.parent.mkdir(parents=True, exist_ok=True)

    server = StdioServerParameters(
        command=sys.executable,
        args=["server/lance_context_mcp.py", "--uri", str(uri)],
        cwd=str(_example_root()),
    )

    try:
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                _print_step("Initialized MCP session")

                stats_result = await _call_tool(session, "stats")
                stats_payload = _require_dict(_payload_from_result(stats_result), "stats")
                initial_version = stats_payload["version"]
                initial_entries = stats_payload["entries"]
                _print_step(f"Initial stats: entries={initial_entries}, version={initial_version}")

                _print_step("Adding memory + knowledge entries")
                await _call_tool(
                    session,
                    "add_entry",
                    {
                        "role": "user",
                        "content": "Remember: I like single-origin coffee.",
                        "session_id": "profile",
                    },
                )
                await _call_tool(
                    session,
                    "add_entry",
                    {
                        "role": "assistant",
                        "content": "The enterprise support SLA is 24 hours.",
                        "session_id": "policy",
                    },
                )
                await _call_tool(
                    session,
                    "add_entry",
                    {
                        "role": "assistant",
                        "content": "Project Nebula rollout is scheduled for May 2026.",
                        "session_id": "roadmap",
                    },
                )

                stats_after_add = _require_dict(
                    _payload_from_result(await _call_tool(session, "stats")), "stats"
                )
                if stats_after_add["entries"] != 3:
                    raise ToolError("Expected 3 entries after initial adds")
                _print_step(f"Stats after adds: entries={stats_after_add['entries']}")

                listed = _extract_list(
                    _payload_from_result(await _call_tool(session, "list_entries", {"limit": 10})),
                    "list_entries",
                )
                if len(listed) != 3:
                    raise ToolError(f"Expected 3 list_entries results, got {len(listed)}")
                _print_step("Listed entries successfully")

                matches = _extract_list(
                    _payload_from_result(await _call_tool(session, "search_entries", {"query": "coffee"})),
                    "search_entries",
                )
                if not any("coffee" in (entry.get("text") or "").lower() for entry in matches):
                    raise ToolError("Expected search_entries to find 'coffee'")
                _print_step("Search returned expected memory match")

                _print_step("Adding a temporary entry to test checkout")
                await _call_tool(
                    session,
                    "add_entry",
                    {
                        "role": "assistant",
                        "content": "Temporary note: remove me after rollback.",
                        "session_id": "temp",
                    },
                )
                stats_after_temp = _require_dict(
                    _payload_from_result(await _call_tool(session, "stats")), "stats"
                )
                if stats_after_temp["entries"] != 4:
                    raise ToolError("Expected 4 entries after temp add")

                _print_step(f"Checkout version {stats_after_add['version']}")
                await _call_tool(session, "checkout_version", {"version_id": stats_after_add["version"]})
                stats_after_checkout = _require_dict(
                    _payload_from_result(await _call_tool(session, "stats")), "stats"
                )
                if stats_after_checkout["entries"] != 3:
                    raise ToolError("Expected 3 entries after checkout")

                _print_step("E2E assertions passed")
    finally:
        if not keep and uri.exists():
            shutil.rmtree(uri)

    if keep:
        print(f"E2E OK. Dataset: {uri}")
    else:
        print("E2E OK. Dataset removed (use --keep to inspect).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP end-to-end test against lance-context.")
    parser.add_argument(
        "--uri",
        type=Path,
        default=_default_uri(),
        help="Lance dataset URI for the test.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the dataset on disk after the test.",
    )
    args = parser.parse_args()
    asyncio.run(run_e2e(args.uri, args.keep))


if __name__ == "__main__":
    main()
