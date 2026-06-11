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
- GraphRAG-friendly `relationships` column for directed edges such as
  `{"target_id": "...", "relation": "cites", "weight": 0.75}`.
- Automatic versioning via Lance manifests with `checkout(version)` support.
- Background compaction to optimize storage and read performance.
- Remote persistence on any `object_store` backend (S3, GCS, Azure Blob, ...)
  via the generic `storage_options` dict, aligned with `lance` and `lance-graph`.
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

Install the core Python package:

```bash
pip install lance-context
```

The default install supports context records, metadata, persistence, and
retrieval without installing the Python `lance-graph` package. Graph/Cypher
integrations can opt in explicitly:

```bash
pip install "lance-context[graph]"
```

If you need direct Python-side Lance/LanceDB inspection of the datasets written
by `lance-context`, install the Lance Python packages extra:

```bash
pip install "lance-context[lance-python]"
```

Then follow the usage examples below to create a `Context`, append entries, and time-travel through versions.

### Python wheels

Release builds publish source distributions plus prebuilt wheels for:

- `manylinux_2_28_x86_64`
- `manylinux_2_28_aarch64`
- `macosx_11_0_arm64`

Other platforms can still install from the source distribution when a Rust
toolchain, maturin, and protobuf compiler are available.

## Usage

### Python

```python
from pathlib import Path
from lance_context.api import Context

uri = Path("context.lance").as_posix()
ctx = Context.create(uri)

# Add multimodal entries
ctx.add(
    "user",
    "Where should I travel in spring?",
    external_id="conversation-2026-03-01#turn-1",
    metadata={
        "tenant": "example-org",
        "scope": "travel-planning",
        "source_uri": "chat://conversation-2026-03-01",
        "tags": ["travel", "preference"],
    },
)
print(ctx.get(external_id="conversation-2026-03-01#turn-1"))
ctx.delete(external_id="conversation-2026-03-01#turn-1")
assert ctx.get(external_id="conversation-2026-03-01#turn-1") is None

# Scoped recall and provenance-oriented metadata
runbook_embedding = [0.0] * 1536
ctx.add(
    "assistant",
    "The runbook owner is the platform team.",
    embedding=runbook_embedding,
    bot_id="support-bot",
    session_id="incident-123",
    relationships=[
        {
            "target_id": "docs://runbooks/service-a",
            "relation": "cites",
            "weight": 0.92,
        },
        {"target_id": "service://service-a", "relation": "describes"},
    ],
    metadata={
        "tenant": "example-org",
        "scope": "team",
        "source_uri": "docs://runbooks/service-a",
        "tags": ["runbook", "ownership"],
        "confidence": 0.92,
    },
)
records = ctx.list(
    filters={
        "bot_id": "support-bot",
        "session_id": "incident-123",
        "scope": "team",
        "tags": {"contains": "runbook"},
    }
)
hits = ctx.search(
    runbook_embedding,
    limit=10,
    filters={"tenant": "example-org", "content_type": "text/plain"},
    include_relationships=True,
)
service_context = ctx.related("service://service-a", relation="describes")

from PIL import Image
image = Image.new("RGB", (2, 2), color="teal")
ctx.add("assistant", image)

print("Current version:", ctx.version())

# Batch append source chunks in one storage operation
ctx.add_many([
    {
        "role": "source",
        "content": "Chunk 1 from a runbook",
        "content_type": "text/markdown",
        "session_id": "runbook-import",
        "relationships": [
            {"target_id": "service://service-a", "relation": "describes"}
        ],
    },
    {
        "role": "source",
        "content": "Chunk 2 from the same runbook",
        "content_type": "text/markdown",
        "session_id": "runbook-import",
    },
])

# Time-travel to prior state
first_version = ctx.version()
ctx.add("assistant", "Let me fetch suggestions…")
ctx.checkout(first_version)

print("Entries after checkout:", ctx.entries())

# Remote persistence on any object_store backend uses a generic `storage_options`
# dict, matching the conventions used by `lance` and `lance-graph`.
#
# Amazon S3 (and S3-compatible endpoints like MinIO / moto):
ctx = Context.create(
    "s3://my-bucket/context.lance",
    storage_options={
        "aws_access_key_id": "minioadmin",
        "aws_secret_access_key": "minioadmin",
        "aws_region": "us-east-1",
        "aws_endpoint_url": "http://localhost:9000",  # optional
        "aws_allow_http": "true",                      # optional
    },
)
# Environment variables (AWS_ACCESS_KEY_ID, ...) are picked up by lance when
# `storage_options` isn't provided; pass overrides only when you need them.

# Google Cloud Storage:
ctx = Context.create(
    "gs://my-bucket/context.lance",
    storage_options={
        # Pick one: inline service-account JSON, path to the JSON file, or ADC.
        "google_service_account_key": service_account_json,
        # "google_service_account_path": "/path/to/sa.json",
        # "google_application_credentials": "/path/to/adc.json",
    },
)

# Azure Blob Storage:
ctx = Context.create(
    "az://my-container/context.lance",
    storage_options={
        "azure_storage_account_name": "...",
        "azure_storage_account_key": "...",
    },
)

# Background Compaction - optimize storage and read performance
ctx = Context.create(
    "context.lance",
    enable_background_compaction=True,  # Enable automatic compaction
    compaction_interval_secs=300,       # Check every 5 minutes
    compaction_min_fragments=10,        # Trigger when 10+ fragments exist
    quiet_hours=[(22, 6)],              # Skip compaction 10pm-6am
)

# Manual compaction control
for i in range(100):
    ctx.add("user", f"message {i}")  # Creates many small fragments

# Check compaction status
stats = ctx.compaction_stats()
print(f"Fragments: {stats['total_fragments']}")

# Manually trigger compaction
metrics = ctx.compact()
print(f"Compaction removed {metrics['fragments_removed']} fragments")
```

`delete()` and its alias `forget()` write a versioned tombstone for the target
record and return `False` if the id is already absent. Default `list()`,
`get()`, and `search()` calls hide tombstoned records, but this is logical
forgetting rather than guaranteed physical erasure: older dataset versions and
underlying files may still contain the original payload until retention or
physical cleanup policies remove them.

### Rust

```rust
use lance_context::{ContextStore, ContextRecord, Relationship, StateMetadata};
use chrono::Utc;

# tokio_test::block_on(async {
let mut store = ContextStore::open("context.lance").await?;
let record = ContextRecord {
    id: "run-1-1".into(),
    external_id: None,
    run_id: "run-1".into(),
    created_at: Utc::now(),
    role: "user".into(),
    state_metadata: Some(StateMetadata {
        step: Some(1),
        active_plan_id: None,
        tokens_used: None,
        custom: None,
    }),
    metadata: None,
    relationships: vec![Relationship {
        target_id: "service://service-a".into(),
        relation: "mentions".into(),
        weight: None,
    }],
    expires_at: None,
    retention_policy: None,
    lifecycle_status: "active".into(),
    retired_at: None,
    retired_reason: None,
    supersedes_id: None,
    superseded_by_id: None,
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

- ~~[Support S3-backed context stores](https://github.com/lance-format/lance-context/issues/14)~~ ✅ **Implemented**
- ~~[Support standard storage_options / GCS](https://github.com/lance-format/lance-context/issues/45)~~ ✅ **Implemented**
- [Add relationship column for GraphRAG workflows](https://github.com/lance-format/lance-context/issues/15)
- ~~[Background compaction for Lance fragments](https://github.com/lance-format/lance-context/issues/16)~~ ✅ **Implemented**

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
