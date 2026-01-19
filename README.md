# lance-context

Multimodal, versioned context storage for agentic workflows built on top of [Lance](https://lancedb.github.io/lance/).

Lance Context gives AI agents a durable memory that can store text, binary payloads (images, Arrow tables, etc.), and semantic embeddings in a single columnar table. Every append produces a new Lance dataset version, so you can time-travel to prior checkpoints, branch off experiments, or reproduce conversations. The project ships with both a Rust API and a thin, Pythonic wrapper that integrates easily with orchestration frameworks.

## Why another context store?

Key motivations inspired by the broader Lance roadmap<sup>[1](https://github.com/lance-format/lance/discussions/5716)</sup>:

- **Multimodal first** – store text, images, and structured data together, keeping the original bytes plus typed metadata.
- **Version aware** – each append creates an immutable snapshot, enabling time-travel, branching, and auditability for long-running agents.
- **Searchable semantics** – embeddings are managed alongside content so you can run Lance vector search without leaving the dataset.
- **Columnar performance** – backed by the Lance file format, giving fast analytics, compaction, and cloud-friendly storage.

## Features

- Unified schema for agent messages (`ContextRecord`) with optional embeddings and metadata.
- Automatic versioning via Lance manifests with `checkout(version)` support.
- Python API (`lance_context.api.Context`) aligned with the Rust implementation.
- Integration tests that exercise real persistence, image serialization, and version rollbacks.

## Project layout

```
crates/lance-context-core  # Pure Rust context engine (no Python deps)
crates/lance-context       # Re-export crate consumed by downstream clients/bindings
python/                    # PyO3 bindings, wheel build, and pytest suite
python/tests/              # High-level integration tests
```

## Getting started

Install the Python package (wheel publishing coming soon):

```bash
pip install lance-context
```

Then follow the usage examples below to create a `Context`, append entries, and time-travel through versions.

## Usage

### Python

```python
from pathlib import Path
from lance_context.api import Context

uri = Path("context.lance").as_posix()
ctx = Context.create(uri)

# Add multimodal entries
ctx.add("user", "Where should I travel in spring?")

from PIL import Image
image = Image.new("RGB", (2, 2), color="teal")
ctx.add("assistant", image)

print("Current version:", ctx.version())

# Time-travel to prior state
first_version = ctx.version()
ctx.add("assistant", "Let me fetch suggestions…")
ctx.checkout(first_version)

print("Entries after checkout:", ctx.entries())
```

### Rust

```rust
use lance_context::{ContextStore, ContextRecord, StateMetadata};
use chrono::Utc;

# tokio_test::block_on(async {
let mut store = ContextStore::open("context.lance").await?;
let record = ContextRecord {
    id: "run-1-1".into(),
    run_id: "run-1".into(),
    created_at: Utc::now(),
    role: "user".into(),
    state_metadata: Some(StateMetadata {
        step: Some(1),
        active_plan_id: None,
        tokens_used: None,
        custom: None,
    }),
    content_type: "text/plain".into(),
    text_payload: Some("hello world".into()),
    binary_payload: None,
    embedding: None,
};
store.add(&[record]).await?;
println!("Current version {}", store.version());
# Ok::<(), Box<dyn std::error::Error>>(())
# })?;
```

## Testing

- `make test` – Python pytest suite (including persistence integration tests).
- `cargo test --manifest-path crates/lance-context-core/Cargo.toml` – Rust unit tests.
- `python/.venv/bin/ruff check python/` and `python/.venv/bin/pyright` – linting/type checks.

## Roadmap

We are tracking future enhancements as GitHub issues:

- [Support S3-backed context stores](https://github.com/lance-format/lance-context/issues/14)
- [Add relationship column for GraphRAG workflows](https://github.com/lance-format/lance-context/issues/15)
- [Background compaction for Lance fragments](https://github.com/lance-format/lance-context/issues/16)

Contributions are welcome—feel free to comment on the issues above or open your own proposals.

## Contributing

1. Fork and clone the repository.
2. Create a feature branch off `main`.
3. Set up the development environment:
   ```bash
   make venv      # creates python/.venv using uv
   make install   # installs the package in editable mode with test extras
   make test      # runs pytest (python/tests/)
   cargo test --manifest-path crates/lance-context-core/Cargo.toml
   ```
4. Run linting/type checks: `python/.venv/bin/ruff check python/`, `python/.venv/bin/pyright`, and `~/.cargo/bin/cargo fmt -- --check`.
5. Open a Pull Request with a clear summary of the change.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
