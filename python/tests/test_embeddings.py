"""Tests for the pluggable embedding provider registry.

Uses a stub provider so no external dependencies are needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from lance_context.api import Context
from lance_context.embeddings import EmbeddingProvider, _build_provider

# ---------------------------------------------------------------------------
# Stub provider
# ---------------------------------------------------------------------------


class StubProvider:
    """Deterministic fake that returns [index, 0.0] per text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def dims(self) -> int:
        return 2

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(i), 0.0] for i in range(len(texts))]


def _ctx_with_provider(provider: StubProvider) -> Context:
    ctx = Context.__new__(Context)
    ctx._inner = _DummyInner()  # type: ignore[attr-defined]
    ctx._embedding_provider = provider
    return ctx


def _ctx_no_provider() -> Context:
    ctx = Context.__new__(Context)
    ctx._inner = _DummyInner()  # type: ignore[attr-defined]
    ctx._embedding_provider = None
    return ctx


class _DummyInner:
    """Minimal inner stub that records calls."""

    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []
        self.add_many_calls: list[list[dict[str, Any]]] = []
        self.search_calls: list[tuple[Any, ...]] = []
        self.upsert_calls: list[dict[str, Any]] = []

    def add(  # noqa: PLR0913
        self,
        role: str,
        content: Any,
        data_type: Any,
        embedding: Any,
        bot_id: Any,
        session_id: Any,
        external_id: Any,
        metadata_json: Any,
        expires_at: Any = None,
        retention_policy: Any = None,
        lifecycle_status: Any = None,
        retired_at: Any = None,
        retired_reason: Any = None,
        supersedes_id: Any = None,
        superseded_by_id: Any = None,
        relationships_json: Any = None,
    ) -> None:
        self.add_calls.append(
            {"role": role, "content": content, "embedding": embedding}
        )

    def add_many(self, records: list[dict[str, Any]]) -> None:
        self.add_many_calls.append(records)

    def search(
        self,
        vector: list[float],
        limit: Any,
        filters_json: Any,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
    ) -> list[Any]:
        self.search_calls.append(
            (
                vector,
                limit,
                filters_json,
                include_expired,
                include_retired,
                include_relationships,
            )
        )
        return []

    def upsert(  # noqa: PLR0913
        self,
        role: str,
        content: Any,
        data_type: Any,
        embedding: Any,
        bot_id: Any,
        session_id: Any,
        external_id: Any,
        metadata_json: Any,
        expires_at: Any = None,
        retention_policy: Any = None,
        lifecycle_status: Any = None,
        retired_at: Any = None,
        retired_reason: Any = None,
        relationships_json: Any = None,
        key: str = "external_id",
    ) -> dict[str, Any]:
        self.upsert_calls.append(
            {"role": role, "content": content, "embedding": embedding}
        )
        return {
            "inserted": True,
            "replaced_id": None,
            "version": 1,
            "record": {
                "id": "x",
                "external_id": external_id,
                "run_id": "r",
                "bot_id": None,
                "session_id": None,
                "role": role,
                "content_type": data_type,
                "text_payload": content,
                "binary_payload": None,
                "embedding": embedding,
                "created_at": "2024-01-01T00:00:00Z",
                "state_metadata": None,
                "metadata": None,
                "relationships": [],
                "expires_at": None,
                "retention_policy": None,
                "lifecycle_status": "active",
                "retired_at": None,
                "retired_reason": None,
                "supersedes_id": None,
                "superseded_by_id": None,
            },
        }


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_stub_satisfies_protocol():
    assert isinstance(StubProvider(), EmbeddingProvider)


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


def test_add_auto_embeds_text():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.add("user", "hello world")
    assert provider.calls == [["hello world"]]
    assert ctx._inner.add_calls[0]["embedding"] == [0.0, 0.0]  # type: ignore[attr-defined]


def test_add_manual_embedding_takes_precedence():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.add("user", "hello", embedding=[9.0, 9.0])
    assert provider.calls == []
    assert ctx._inner.add_calls[0]["embedding"] == [9.0, 9.0]  # type: ignore[attr-defined]


def test_add_no_provider_leaves_embedding_none():
    ctx = _ctx_no_provider()
    ctx.add("user", "hello")
    assert ctx._inner.add_calls[0]["embedding"] is None  # type: ignore[attr-defined]


def test_add_binary_content_not_auto_embedded():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.add("user", b"\x00\x01\x02", content_type="application/octet-stream")
    assert provider.calls == []
    assert ctx._inner.add_calls[0]["embedding"] is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# add_many()
# ---------------------------------------------------------------------------


def test_add_many_batch_embeds_text_records():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.add_many(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    )
    # Both texts sent in one provider call.
    assert provider.calls == [["first", "second"]]
    records = ctx._inner.add_many_calls[0]  # type: ignore[attr-defined]
    assert records[0]["embedding"] == [0.0, 0.0]
    assert records[1]["embedding"] == [1.0, 0.0]


def test_add_many_skips_records_with_manual_embedding():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.add_many(
        [
            {"role": "user", "content": "first", "embedding": [5.0, 5.0]},
            {"role": "assistant", "content": "second"},
        ]
    )
    # Only the second record is sent for embedding.
    assert provider.calls == [["second"]]
    records = ctx._inner.add_many_calls[0]  # type: ignore[attr-defined]
    assert records[0]["embedding"] == [5.0, 5.0]
    assert records[1]["embedding"] == [0.0, 0.0]


def test_add_many_no_provider_leaves_embeddings_unchanged():
    ctx = _ctx_no_provider()
    ctx.add_many([{"role": "user", "content": "hello"}])
    records = ctx._inner.add_many_calls[0]  # type: ignore[attr-defined]
    assert records[0]["embedding"] is None


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_auto_embeds_string_query():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.search("spring travel")
    assert provider.calls == [["spring travel"]]
    vector_passed = ctx._inner.search_calls[0][0]  # type: ignore[attr-defined]
    assert vector_passed == [0.0, 0.0]


def test_search_vector_query_bypasses_provider():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.search([0.1, 0.2])
    assert provider.calls == []
    vector_passed = ctx._inner.search_calls[0][0]  # type: ignore[attr-defined]
    assert vector_passed == [0.1, 0.2]


def test_search_string_query_no_provider_raises():
    ctx = _ctx_no_provider()
    with pytest.raises(TypeError):
        ctx.search("spring travel")


# ---------------------------------------------------------------------------
# upsert()
# ---------------------------------------------------------------------------


def test_upsert_auto_embeds_text():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.upsert("user", "updated content", external_id="doc-1")
    assert provider.calls == [["updated content"]]
    assert ctx._inner.upsert_calls[0]["embedding"] == [0.0, 0.0]  # type: ignore[attr-defined]


def test_upsert_manual_embedding_takes_precedence():
    provider = StubProvider()
    ctx = _ctx_with_provider(provider)
    ctx.upsert("user", "updated content", external_id="doc-1", embedding=[7.0, 7.0])
    assert provider.calls == []
    assert ctx._inner.upsert_calls[0]["embedding"] == [7.0, 7.0]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _build_provider registry
# ---------------------------------------------------------------------------


def test_build_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        _build_provider({"provider": "does-not-exist"})


def test_build_provider_missing_key_raises():
    with pytest.raises(ValueError, match="'provider' key"):
        _build_provider({})
