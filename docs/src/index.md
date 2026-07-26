# lance-context

**A storage engine for AI data — agent memory and RL training data — built on
[Lance](https://lancedb.github.io/lance/).**

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

Then head to the [Quickstart](guide/quickstart.md).

## Project layout

```
crates/lance-context-core    # Rust engine: Context + Rollout stores (no Python deps)
crates/lance-context-api     # Shared request/response types (DTOs)
crates/lance-context-server  # HTTP server for remote access
crates/lance-context-client  # HTTP client for the server
crates/lance-context-master  # Control plane, scheduler, and admin UI
crates/lance-context         # Re-export crate for downstream clients
python/                      # Python bindings (PyO3) + tests
deploy/kubernetes/           # Kubernetes master examples
examples/                    # Runnable example projects
```

## Examples

The [`examples/`](https://github.com/lance-format/lance-context/tree/main/examples)
directory has self-contained, runnable projects:

- [`pypi-basic`](https://github.com/lance-format/lance-context/tree/main/examples/pypi-basic) — 5-minute quickstart
- [`multi-session`](https://github.com/lance-format/lance-context/tree/main/examples/multi-session) — concurrent multi-bot writes
- [`eval-quality`](https://github.com/lance-format/lance-context/tree/main/examples/eval-quality) — measuring retrieval quality
- [`mcp-claude-code`](https://github.com/lance-format/lance-context/tree/main/examples/mcp-claude-code) — serving a store over MCP

## License

Licensed under the Apache License, Version 2.0. See
[LICENSE](https://github.com/lance-format/lance-context/blob/main/LICENSE) for
details.
