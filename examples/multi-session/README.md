# Multi-Session Context Example

Demonstrates how `lance-context` uses **MemWAL** (Memory Write-Ahead Log) to handle concurrent, multi-bot context storage with automatic shard isolation.

## What this shows

- **Multiple bots** (coding-assistant, research-assistant) writing to the **same dataset**
- **Automatic sharding** by `(bot_id, session_id)` — each pair gets its own WAL region
- **No contention** between concurrent writers targeting different shards
- **Time-travel** with `checkout()` across all shards

## How MemWAL sharding works

When you call `ctx.add(..., bot_id="X", session_id="Y")`, lance-context:

1. Hashes `(bot_id, session_id)` into a deterministic UUID v5 shard ID
2. Routes the write to that shard's WAL region
3. Periodically compacts WAL regions into optimized Lance fragments

This means multiple agents can write simultaneously without coordination — each writes to its own isolated WAL region.

## Run

```bash
pip install lance-context>=0.2.5
cd examples/multi-session
pip install -e .
multi-session-demo
```

## Expected output

```
Created context store at .artifacts/multi_session_XXXXXXXX.lance
After coding session: version=N, entries=5
After research session: version=N, entries=9
After second coding session: version=N, entries=14
Rolled back to version N: entries=5
Restored to version N: entries=14
All writes were sharded by (bot_id, session_id) via MemWAL. Each pair wrote to its own WAL region independently.
```
