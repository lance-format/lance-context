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

# Cross-modal retrieval: plug in a multi-modal (e.g. CLIP-style) embedder that
# maps text and media into one shared space. Images are auto-embedded via
# `embed_media`; a text query (embedded via `embed_texts`) then retrieves them.
# lance-context bundles no models — supply your own provider.
clip_ctx = Context.create("multimodal.lance", embedding_dim=512,
                          embedding_provider=my_clip_provider)  # implements MultiModalEmbeddingProvider
clip_ctx.add("user", image_bytes, content_type="image/png", external_id="img-1")
results = clip_ctx.search("a photo of a cat")  # text query -> image results
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

# Multi-modal-friendly reads: skip large media bytes for metadata/search
# queries, then fetch a record's bytes on demand.
lean = ctx.list(filters={"content_type": "image/png"}, include_binary=False)
image_bytes = ctx.get_blob(lean[0]["id"])
hits = ctx.search(query_embedding, include_binary=False)  # no bytes pulled into results

# Hybrid retrieval combines lexical recall, vector recall, and existing filters
# over the same context records.
hybrid_hits = ctx.retrieve(
    text="service-a runbook owner",
    vector=runbook_embedding,
    limit=5,
    filters={"tenant": "example-org", "scope": "team"},
)
print(hybrid_hits[0]["matched_channels"], hybrid_hits[0]["score"])

from PIL import Image
image = Image.new("RGB", (2, 2), color="teal")
ctx.add("assistant", image)

print("Current version:", ctx.version())

# External media references: keep large media in object storage (GCS/S3/local)
# and reference it from a record by a typed URI, instead of inlining the bytes.
object_uri = "gs://my-bucket/media/diagram-001.png"
ctx.put_payload(object_uri, image_bytes)  # offload bytes via the context's storage_options
ctx.add(
    "assistant",
    "incident timeline diagram",  # inline caption; the media stays in the bucket
    content_type="image/png",
    external_id="diagram-001",
    payload_uri=object_uri,
    payload_size=len(image_bytes),
)
# list/search/get return the reference without fetching the bytes; resolve on demand:
record = ctx.get(external_id="diagram-001")
media_bytes = ctx.fetch_payload(record["id"])


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

# Bulk insert-or-replace by external_id (idempotent re-ingestion). New
# external_ids are inserted; existing ones are replaced (the previous row is
# superseded), all in one storage operation. Returns one result per record.
results = ctx.upsert_many([
    {
        "role": "source",
        "content": "Chunk 1, revised",
        "content_type": "text/markdown",
        "external_id": "doc-77#chunk-1",
    },
    {
        "role": "source",
        "content": "Brand new chunk",
        "content_type": "text/markdown",
        "external_id": "doc-77#chunk-2",
    },
])
print([(r["inserted"], r["replaced_id"]) for r in results])

# Deferred embeddings: raw-first capture, enrich later.
#
# Bulk ingestion often needs to persist source chunks immediately and compute
# embeddings asynchronously (large documents, rate-limited or remote embedding
# providers). Append the raw text first with a stable external_id, then have a
# worker patch in the embedding once it is ready. A record without an embedding
# is durably stored but excluded from vector search until it is enriched.
ctx.add_many([
    {
        "role": "source",
        "content": "Deferred chunk",
        "external_id": "doc-77#chunk-1",
        "metadata": {"embedding_status": "pending"},
    },
])

# ...later, from your own worker/queue/batch job:
vector = [0.0] * 1536
ctx.update(
    external_id="doc-77#chunk-1",
    embedding=vector,                       # attach the freshly computed vector
    metadata={"embedding_status": "ready"},
)
# The enriched record now shows up in vector search and hybrid retrieve.

# Raw interaction-log ingestion: map raw rows into ContextRecord fields while
# preserving the original row under metadata["raw_record"]. Use mode="upsert"
# with stable external_id values for idempotent re-ingestion.
ctx.ingest_jsonl(
    "chat-log.jsonl",
    field_map={
        "external_id": "event_id",
        "role": "speaker",
        "content": "message",
        "session_id": "conversation_id",
        "run_id": "trace_id",
        "created_at": "timestamp",
    },
    defaults={"tenant": "example-org"},
    mode="upsert",
    batch_size=500,
)

# OpenAI-style messages can be ingested directly as raw records.
ctx.ingest_messages(
    [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What changed?"},
    ],
    session_id="conversation-2026-03-01",
    external_id_prefix="conversation-2026-03-01",
)

# Time-travel to prior state
first_version = ctx.version()
ctx.add("assistant", "Let me fetch suggestions…")
ctx.checkout(first_version)

print("Entries after checkout:", ctx.entries())

<<<<<<< HEAD
# Curate stored records into a trainable dataset and export it as JSONL plus a
# reproducible manifest. Curation (lifecycle-correct filtering, semantic dedup,
# decontamination against a holdout set, reward thresholding) runs before
# export; reward/preference signals are read from each record's `metadata`
# (`reward`, `reward_source`, `group_id`, `label`, `rank`).
#
# SFT, grouped by session into conversations (rejection-sampling via min_reward):
manifest = ctx.export_training(
    "training/sft.jsonl",
    task="sft",
    group_by="session_id",
    filters={"tenant": "acme", "source": "memory"},
    min_reward=0.7,            # keep only high-reward completions
    dedup_threshold=0.02,      # collapse near-duplicates (cosine)
    version=ctx.version(),     # pin for reproducibility
    emit_stats=True,           # also write sft.jsonl.stats.json (counts, tokens, exclusions)
)
print("SFT examples:", manifest["counts"]["examples"])

# Preference (DPO paired / KTO unpaired / judge-ranked) and RL rollout shapes:
ctx.export_training("training/pref.jsonl", task="preference", preference_form="paired")
ctx.export_training("training/rollout.jsonl", task="rollout")  # GRPO/RLVR groups

# Reproducible, group-disjoint train/eval split (no session leaks across the
# boundary; same seed reproduces the partition). Writes sft.train.jsonl +
# sft.eval.jsonl, each with its own manifest.
ctx.export_training(
    "training/sft.jsonl",
    task="sft",
    split={"eval_fraction": 0.1, "by": "session_id", "seed": 42},
)

# Measure retrieval quality against a labeled query set. Each query lists the
# relevant records by stable external_id (with optional graded relevance), and
# the report carries recall@k / precision@k / MRR / nDCG@k / hit-rate plus a
# manifest (version, k, mode, distance_metric) for reproducibility.
report = ctx.evaluate(
    [
        {
            "query_id": "q1",
            "vector": query_embedding,            # vector channel
            "relevant": [{"external_id": "doc-77#chunk-1", "grade": 1.0}],
        },
    ],
    k=10,
    mode="vector",                                # or "hybrid" for text+vector
)
print("recall@10:", report["aggregate"]["recall"])

# A/B the same query set across two dataset versions (regression detection that
# a stateless vector DB can't do) and read per-metric deltas.
ab = ctx.evaluate_versions(query_set, baseline_version, candidate_version, k=10)
print("nDCG delta:", ab["deltas"]["ndcg"])


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
    payload_uri: None,
    payload_size: None,
    payload_checksum: None,
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
- ~~[External media references — store large media in object storage by typed URI](https://github.com/lance-format/lance-context/issues/115)~~ ✅ **Implemented**

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
