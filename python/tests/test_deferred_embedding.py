"""End-to-end tests for deferred embedding workflows (issue #88).

Raw-first ingestion: append source chunks without embeddings, then enrich each
record with an embedding later via ``update()``. The enriched record must then
participate in vector search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def _embedding(value: float) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = value
    return vector


def test_update_attaches_embedding_by_external_id(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    # Raw-first capture: persist the source chunk immediately, no embedding yet.
    ctx.add(
        "user",
        "raw source chunk",
        external_id="doc-1#chunk-1",
        metadata={"embedding_status": "pending"},
    )

    raw = ctx.get(external_id="doc-1#chunk-1")
    assert raw is not None
    assert raw["embedding"] is None

    # Without an embedding the record is invisible to vector search.
    assert ctx.search(_embedding(1.0), limit=10) == []

    # Enrich-later: a worker computes the embedding and patches it in.
    result = ctx.update(
        external_id="doc-1#chunk-1",
        embedding=_embedding(1.0),
        metadata={"embedding_status": "ready"},
    )
    assert result["updated"] is True
    assert result["record"]["embedding"] == _embedding(1.0)
    # Raw payload is preserved across the enrich update.
    assert result["record"]["text"] == "raw source chunk"

    # The enriched record now participates in vector search.
    hits = ctx.search(_embedding(1.0), limit=10)
    assert [hit["external_id"] for hit in hits] == ["doc-1#chunk-1"]


def test_update_attaches_embedding_by_id(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add("user", "raw source chunk")
    raw = ctx.list()[0]
    assert raw["embedding"] is None

    result = ctx.update(id=raw["id"], embedding=_embedding(0.0))
    assert result["updated"] is True
    assert result["record"]["embedding"] == _embedding(0.0)

    hits = ctx.search(_embedding(0.0), limit=10)
    assert len(hits) == 1
    assert hits[0]["id"] == result["record"]["id"]


def test_embedding_only_is_a_valid_patch(tmp_path: Path) -> None:
    """An embedding-only patch must be accepted (no other field required)."""
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add("user", "raw source chunk", external_id="doc-2#chunk-1")
    result = ctx.update(external_id="doc-2#chunk-1", embedding=_embedding(1.0))
    assert result["updated"] is True
    assert result["record"]["embedding"] == _embedding(1.0)


def test_bulk_raw_first_then_enrich(tmp_path: Path) -> None:
    """add_many() raw chunks, then enrich each by external_id."""
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add_many(
        [
            {"role": "user", "content": "chunk a", "external_id": "doc-3#chunk-1"},
            {"role": "user", "content": "chunk b", "external_id": "doc-3#chunk-2"},
        ]
    )
    assert ctx.search(_embedding(1.0), limit=10) == []

    for ext_id, pivot in (("doc-3#chunk-1", 0.0), ("doc-3#chunk-2", 1.0)):
        ctx.update(external_id=ext_id, embedding=_embedding(pivot))

    hits = ctx.search(_embedding(1.0), limit=10)
    assert {hit["external_id"] for hit in hits} == {"doc-3#chunk-1", "doc-3#chunk-2"}
    # The exact match ranks first.
    assert hits[0]["external_id"] == "doc-3#chunk-2"
