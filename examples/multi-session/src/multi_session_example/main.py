"""Multi-bot, multi-session context store with MemWAL sharding.

Demonstrates how lance-context uses MemWAL to shard writes by
(bot_id, session_id), enabling concurrent agents to write to the
same dataset without contention. Each unique pair gets its own
write-ahead log region.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from lance_context import Context


def simulate_coding_assistant(ctx: Context, session_id: str) -> None:
    """A coding assistant bot handling a debug session."""
    bot_id = "coding-assistant"

    ctx.add(
        "system",
        "You are a senior engineer helping debug production issues.",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "user",
        "Our API latency spiked to 2s after the last deploy. P99 was 200ms before.",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "assistant",
        "Let's check the deployment diff and correlate with the latency timeline. "
        "Can you share the Grafana dashboard link and the commit range?",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "user",
        "Found it — we added an N+1 query in the user-preferences loader.",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "assistant",
        "Good catch. Batch the preference lookups with a single IN query. "
        "Here's the fix: `SELECT * FROM prefs WHERE user_id IN (...)`",
        bot_id=bot_id,
        session_id=session_id,
    )


def simulate_research_assistant(ctx: Context, session_id: str) -> None:
    """A research assistant bot exploring a topic across turns."""
    bot_id = "research-assistant"

    ctx.add(
        "system",
        "You are a research assistant specializing in distributed systems.",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "user",
        "Compare write-ahead logs vs copy-on-write for versioned datasets.",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "assistant",
        "WAL appends mutations sequentially and replays on read — great for write-heavy "
        "workloads. COW snapshots the full state on each write — better for read-heavy "
        "patterns. Lance's MemWAL combines both: WAL for fast ingestion with periodic "
        "compaction into COW fragments.",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "user",
        "How does sharding help with concurrent writes?",
        bot_id=bot_id,
        session_id=session_id,
    )
    ctx.add(
        "assistant",
        "Each (bot_id, session_id) pair maps to a deterministic shard via UUID v5. "
        "Writers to different shards never contend on the same WAL files, so multiple "
        "agents can ingest concurrently without coordination.",
        bot_id=bot_id,
        session_id=session_id,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = project_root / ".artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    dataset_path = artifacts_dir / f"multi_session_{uuid4().hex[:8]}.lance"
    ctx = Context.create(dataset_path.as_posix())
    print(f"Created context store at {dataset_path}")

    # Two bots write to the same dataset with different sessions.
    # MemWAL automatically shards writes by (bot_id, session_id),
    # so each pair gets its own WAL region — no contention.
    coding_session = "debug-incident-42"
    research_session = "wal-deep-dive"

    simulate_coding_assistant(ctx, coding_session)
    v1 = ctx.version()
    print(f"After coding session: version={v1}, entries={ctx.entries()}")

    simulate_research_assistant(ctx, research_session)
    v2 = ctx.version()
    print(f"After research session: version={v2}, entries={ctx.entries()}")

    # Add a second coding session (same bot, different session = different shard)
    simulate_coding_assistant(ctx, "debug-incident-43")
    v3 = ctx.version()
    print(f"After second coding session: version={v3}, entries={ctx.entries()}")

    # Time-travel: roll back to see only the first coding session
    ctx.checkout(v1)
    print(f"\nRolled back to version {v1}: entries={ctx.entries()}")

    # Restore latest
    ctx.checkout(v3)
    print(f"Restored to version {v3}: entries={ctx.entries()}")

    print(
        "\nAll writes were sharded by (bot_id, session_id) via MemWAL. "
        "Each pair wrote to its own WAL region independently."
    )


if __name__ == "__main__":
    main()
