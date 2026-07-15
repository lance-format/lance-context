# lance-context

**A storage engine for AI data — agent memory and RL training data — built on [Lance](https://lancedb.github.io/lance/).**

Modern AI systems produce two kinds of data that are awkward to store well:

1. **Agent memory** — the running history of a chat or agent: text, images, tool
   calls, and their embeddings, which you later search over to recall context.
2. **RL training data** — the trajectories, rewards, and logprobs produced when
   you train models with reinforcement learning (GRPO, RLVR, PPO, ...).

`lance-context` stores both in one place. It gives you a durable, columnar,
versioned table you can append to, search, filter, and time-travel through —
without standing up a database server (though a server is available if you want
one).

## Why use it

- **Two use cases, one engine.** A `Context` store for agent memory and a
  `RolloutStore` (a purpose-built **RolloutDB**) for RL rollouts. Same storage
  format, same versioning, same cloud backends.
- **Multimodal.** Store text, images, and binary blobs next to their embeddings
  and typed metadata — the raw bytes are kept, not just a pointer.
- **Search built in.** Run vector search, full-text search, or hybrid retrieval
  directly on the table. No separate vector database.
- **Versioned.** Every write creates a new immutable snapshot. Roll back, branch,
  or reproduce an exact state with `checkout(version)`.
- **Runs anywhere.** Local files, or S3 / GCS / Azure via a simple
  `storage_options` dict. Embedded in your process, or behind an HTTP server.

## Install

```bash
pip install lance-context
```

## Quick start

### 1. Agent memory

Store what your agent sees and says, then search it back.

```python
from lance_context import Context

ctx = Context.create("memory.lance")

# Add a message with an embedding so it's searchable.
ctx.add(
    "user",
    "Where should I travel in spring?",
    embedding=[0.1, 0.2, 0.3],  # from your own embedding model
    metadata={"tenant": "acme", "tags": ["travel"]},
)

# Semantic search returns the closest records.
hits = ctx.search([0.1, 0.2, 0.3], limit=5)
print(hits[0]["text"])

# Time-travel: go back to an earlier version of the store.
v = ctx.version()
ctx.add("assistant", "How about Japan?")
ctx.checkout(v)  # the second message is no longer visible
```

`Context` also supports images and other binary payloads, metadata filtering,
hybrid (text + vector) retrieval, batch ingestion, and object-storage backends.
See [`examples/`](./examples) for runnable projects.

### 2. RL rollouts (a "RolloutDB")

Store training trajectories — one row per step (an assistant turn, a tool call,
a grade, or an artifact). Only `id` and `rollout_id` are required; everything
else (tokens, logprobs, rewards, advantages) is optional and filled in as your
pipeline computes it. We call this shape a **RolloutDB**: a versioned,
columnar store purpose-built for RL rollout data, the same way a vector DB is
built for embeddings.

```python
from lance_context import RolloutStore

store = RolloutStore.open("rollouts.lance")

# One step of a trajectory. `id` is auto-generated if you omit it.
store.add({
    "rollout_id": "traj-1",   # the trajectory this row belongs to
    "problem_id": "prompt-7", # groups the N samples of one prompt (for GRPO)
    "role": "assistant",
    "content": "The answer is 42.",
    "reward": 1.0,
    "policy_version": "ckpt-100",
})

for row in store.list():
    print(row["rollout_id"], row["reward"])

# Exact-match filters are combined with AND and applied before pagination.
training_rows = store.list(filters={
    "policy_version": "ckpt-100",
    "include_in_training": True,
    "role": "assistant",
})
```

In a real training run, generation workers and the learner talk to a shared
store over HTTP. Use `AsyncRolloutStore` from async code so writes don't block
your event loop:

```python
from lance_context import AsyncRolloutStore

store = await AsyncRolloutStore.connect_or_create("http://localhost:8080", "rl-run-1")
await store.add({"rollout_id": "traj-1", "role": "assistant", "reward": 1.0})
```

## Rust

The engine is written in Rust and usable directly:

```rust
use lance_context::ContextStore;

let mut store = ContextStore::open("memory.lance").await?;
store.add(&[record]).await?;
println!("version {}", store.version());
```

See `crates/lance-context-core` for the full `ContextRecord` shape and the
`RolloutRecord` schema.

## Project layout

```
crates/lance-context-core    # Rust engine: Context + Rollout stores (no Python deps)
crates/lance-context-api     # Shared request/response types (DTOs)
crates/lance-context-server  # HTTP server for remote access
crates/lance-context-client  # HTTP client for the server
crates/lance-context         # Re-export crate for downstream clients
python/                      # Python bindings (PyO3) + tests
examples/                    # Runnable example projects
```

## Storage backends

Point any store at cloud storage with `storage_options`:

```python
ctx = Context.create(
    "s3://my-bucket/memory.lance",
    storage_options={
        "aws_access_key_id": "...",
        "aws_secret_access_key": "...",
        "aws_region": "us-east-1",
    },
)
```

GCS (`gs://`) and Azure (`az://`) work the same way. When `storage_options` is
omitted, standard environment variables (`AWS_ACCESS_KEY_ID`, ...) are used.

## Examples

The [`examples/`](./examples) directory has self-contained, runnable projects:

- [`pypi-basic`](./examples/pypi-basic) — 5-minute quickstart
- [`multi-session`](./examples/multi-session) — concurrent multi-bot writes
- [`eval-quality`](./examples/eval-quality) — measuring retrieval quality
- [`mcp-claude-code`](./examples/mcp-claude-code) — serving a store over MCP

See the [examples index](./examples/README.md) for the full list.

## Development

```bash
make venv      # create python/.venv using uv
make install   # editable install with test extras
make test      # run the Python test suite
cargo test --manifest-path crates/lance-context-core/Cargo.toml
```

Linting and type checks: `python/.venv/bin/ruff check python/`,
`python/.venv/bin/pyright`, and `cargo fmt -- --check`.

## Contributing

1. Fork and clone the repository.
2. Create a feature branch off `main`.
3. Make your change, add tests, and run the checks above.
4. Open a Pull Request with a clear summary.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
