# Feature: Add embedding support to `Context.add()` API

## Problem

Currently, the `Context.add()` API stores records with `embedding: None` hardcoded ([lib.rs#L147](https://github.com/lance-format/lance-context/blob/main/python/src/lib.rs#L147)). This means:

1. **`search()` is unusable** - Vector similarity search requires embeddings, but users have no way to store them
2. **The embedding field exists but is inaccessible** - `ContextRecord` has an `embedding: Option<Vec<f32>>` field, but the Python API doesn't expose it

## Proposed Solution

Based on industry patterns (LanceDB, ChromaDB, Pinecone), support **both** auto-embedding and manual embedding:

### Option A: Auto-embedding (recommended default)

Allow users to configure an embedding function at Context creation time:

```python
ctx = Context.open(
    "my-context",
    embedding_function="sentence-transformers/all-MiniLM-L6-v2"
)
ctx.add("user", "Hello world")  # Auto-generates embedding
ctx.search("greeting")  # Works!
```

**Pros:**
- Simpler API - users don't think about embeddings
- Consistent embeddings between ingestion and query
- Aligns with LanceDB's embedding function registry pattern

**Cons:**
- Requires embedding model infrastructure
- Less control over model version/parameters

### Option B: Manual embedding parameter

Add an optional `embedding` parameter to `add()`:

```python
embedding = my_model.encode("Hello world")
ctx.add("user", "Hello world", embedding=embedding)
```

**Pros:**
- Full control over embedding model
- Works offline / with custom models
- Simpler implementation

**Cons:**
- More boilerplate for users
- Risk of embedding mismatches between add and search

### Recommendation: Support both

```python
# Auto-embedding (if configured)
ctx = Context.open("my-context", embedding_function="...")
ctx.add("user", "Hello world")  # Auto-embeds

# Manual override (always available)
ctx.add("user", "Hello world", embedding=custom_vector)

# No embedding (current behavior, for non-searchable entries)
ctx.add("user", "Hello world")  # embedding=None if no function configured
```

## Implementation Notes

### Changes needed:

1. **Rust core (`store.rs`)**: Already supports embeddings - no changes needed
2. **PyO3 bindings (`lib.rs`)**: Add optional `embedding: Option<Vec<f32>>` parameter to `add()`
3. **Python API (`api.py`)**: Expose `embedding` parameter, add embedding function configuration
4. **Embedding registry**: Could leverage LanceDB's existing `get_registry()` or build a simpler version

### Questions for discussion:

- [ ] Should auto-embedding be opt-in or opt-out?
- [ ] Which embedding providers should we support out of the box?
- [ ] Should we auto-embed at query time too (for `search(text)` instead of `search(vector)`)?
- [ ] How do we handle schema migration for existing contexts without embeddings?

## Related

- Current `search()` implementation requires a vector: `search(query: Any, limit: int | None = None)`
- `ContextRecord` struct already has embedding field: `pub embedding: Option<Vec<f32>>`
