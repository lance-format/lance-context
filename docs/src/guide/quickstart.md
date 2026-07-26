# Quickstart

## Install

```bash
pip install lance-context
```

## Agent memory

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

## Next steps

- [Rollouts (RolloutDB)](rollouts.md) — storing RL training data
- [Storage backends](storage.md) — S3, GCS, Azure
