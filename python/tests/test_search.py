import json
from datetime import datetime, timezone
from typing import Any

import pytest
from lance_context.api import (
    Context,
    _coerce_vector,
    _normalize_record,
    _normalize_retrieve_hit,
    _normalize_search_hit,
)


class DummyInner:
    def __init__(self) -> None:
        self.search_calls: list[tuple[list[float], int | None, str | None]] = []
        self.search_lifecycle_calls: list[tuple[bool, bool]] = []
        self.search_relationship_calls: list[bool] = []
        self.retrieve_calls: list[
            tuple[
                str | None,
                list[float] | None,
                int | None,
                str | None,
                bool,
                bool,
                bool,
                str,
            ]
        ] = []
        self.list_calls: list[tuple[int | None, int | None, str | None]] = []
        self.list_lifecycle_calls: list[tuple[bool, bool]] = []
        self.list_projection_calls: list[tuple[bool, bool]] = []
        self.search_projection_calls: list[tuple[bool, bool]] = []
        self.related_calls: list[tuple[str, str | None, int | None, bool, bool]] = []
        self.get_calls: list[tuple[str | None, str | None]] = []
        self.delete_calls: list[tuple[str | None, str | None]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.lifecycle_add_calls: list[dict[str, Any]] = []
        self.relationship_add_calls: list[str | None] = []
        self.state_metadata_add_calls: list[dict[str, Any] | None] = []
        self.add_calls: list[
            tuple[
                str,
                Any,
                str | None,
                list[float] | None,
                str | None,
                str | None,
                str | None,
                str | None,
            ]
        ] = []
        self.add_many_calls: list[list[dict[str, Any]]] = []

    def add(
        self,
        role: str,
        content: Any,
        data_type: str | None,
        embedding: list[float] | None,
        bot_id: str | None,
        session_id: str | None,
        tenant: str | None,
        source: str | None,
        external_id: str | None,
        run_id: str | None,
        created_at: str | None,
        state_metadata: dict[str, Any] | None,
        metadata_json: str | None,
        expires_at: str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: str | None = None,
        retired_reason: str | None = None,
        supersedes_id: str | None = None,
        superseded_by_id: str | None = None,
        relationships_json: str | None = None,
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
    ):
        self.add_calls.append(
            (
                role,
                content,
                data_type,
                embedding,
                bot_id,
                session_id,
                external_id,
                metadata_json,
            )
        )
        self.lifecycle_add_calls.append(
            {
                "expires_at": expires_at,
                "retention_policy": retention_policy,
                "lifecycle_status": lifecycle_status,
                "retired_at": retired_at,
                "retired_reason": retired_reason,
                "supersedes_id": supersedes_id,
                "superseded_by_id": superseded_by_id,
            }
        )
        self.relationship_add_calls.append(relationships_json)
        self.state_metadata_add_calls.append(state_metadata)

    def upsert(
        self,
        role: str,
        content: Any,
        data_type: str | None,
        embedding: list[float] | None,
        bot_id: str | None,
        session_id: str | None,
        tenant: str | None,
        source: str | None,
        external_id: str | None,
        run_id: str | None,
        created_at: str | None,
        state_metadata: dict[str, Any] | None,
        metadata_json: str | None,
        expires_at: str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: str | None = None,
        retired_reason: str | None = None,
        relationships_json: str | None = None,
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
        key: str = "external_id",
    ):
        self.upsert_calls.append(
            {
                "role": role,
                "content": content,
                "data_type": data_type,
                "embedding": embedding,
                "bot_id": bot_id,
                "session_id": session_id,
                "external_id": external_id,
                "metadata_json": metadata_json,
                "expires_at": expires_at,
                "retention_policy": retention_policy,
                "lifecycle_status": lifecycle_status,
                "retired_at": retired_at,
                "retired_reason": retired_reason,
                "relationships_json": relationships_json,
                "key": key,
            }
        )
        return {
            "inserted": False,
            "replaced_id": "old-id",
            "version": 7,
            "record": {
                "id": "new-id",
                "external_id": external_id,
                "run_id": "run-2",
                "bot_id": bot_id,
                "session_id": session_id,
                "role": role,
                "content_type": data_type or "text/plain",
                "text_payload": content if isinstance(content, str) else None,
                "binary_payload": None,
                "embedding": embedding,
                "created_at": "2024-01-03T12:00:00Z",
                "state_metadata": None,
                "metadata": json.loads(metadata_json) if metadata_json else None,
                "relationships": (
                    json.loads(relationships_json) if relationships_json else []
                ),
                "expires_at": expires_at,
                "retention_policy": retention_policy,
                "lifecycle_status": lifecycle_status or "active",
                "retired_at": retired_at,
                "retired_reason": retired_reason,
                "supersedes_id": "old-id",
                "superseded_by_id": None,
            },
        }

    def update(
        self,
        id: str | None,
        external_id: str | None,
        bot_id: str | None,
        session_id: str | None,
        tenant: str | None,
        source: str | None,
        metadata_json: str | None,
        relationships_json: str | None,
        expires_at: str | None,
        retention_policy: str | None,
        lifecycle_status: str | None,
        retired_at: str | None,
        retired_reason: str | None,
        embedding: list[float] | None = None,
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
    ):
        self.update_calls.append(
            {
                "id": id,
                "external_id": external_id,
                "bot_id": bot_id,
                "session_id": session_id,
                "metadata_json": metadata_json,
                "relationships_json": relationships_json,
                "expires_at": expires_at,
                "retention_policy": retention_policy,
                "lifecycle_status": lifecycle_status,
                "retired_at": retired_at,
                "retired_reason": retired_reason,
                "embedding": embedding,
            }
        )
        if id == "missing" or external_id == "missing":
            return {
                "updated": False,
                "replaced_id": None,
                "version": 7,
                "record": None,
            }
        return {
            "updated": True,
            "replaced_id": "old-id",
            "version": 8,
            "record": {
                "id": "new-id",
                "external_id": external_id,
                "run_id": "run-2",
                "bot_id": bot_id,
                "session_id": session_id,
                "role": "user",
                "content_type": "text/plain",
                "text_payload": "stable content",
                "binary_payload": None,
                "embedding": None,
                "created_at": "2024-01-03T12:00:00Z",
                "state_metadata": None,
                "metadata": json.loads(metadata_json) if metadata_json else None,
                "relationships": (
                    json.loads(relationships_json) if relationships_json else []
                ),
                "expires_at": expires_at,
                "retention_policy": retention_policy,
                "lifecycle_status": lifecycle_status or "active",
                "retired_at": retired_at,
                "retired_reason": retired_reason,
                "supersedes_id": "old-id",
                "superseded_by_id": None,
            },
        }

    def get(self, id: str | None, external_id: str | None):
        self.get_calls.append((id, external_id))
        if id == "rec-1" or external_id == "source-1":
            return self.list(None, None, None)[0]
        return None

    def delete(self, id: str | None, external_id: str | None):
        self.delete_calls.append((id, external_id))
        return id == "rec-1" or external_id == "source-1"

    def search(
        self,
        vector: list[float],
        limit: int | None,
        filters_json: str | None,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
        include_binary: bool = True,
        include_embedding: bool = True,
    ):
        self.search_calls.append((vector, limit, filters_json))
        self.search_lifecycle_calls.append((include_expired, include_retired))
        self.search_relationship_calls.append(include_relationships)
        self.search_projection_calls.append((include_binary, include_embedding))
        hit = {
            "id": "rec-1",
            "external_id": "source-1",
            "run_id": "run-1",
            "bot_id": "support_bot",
            "session_id": None,
            "role": "user",
            "content_type": "text/plain",
            "text_payload": "hello",
            "binary_payload": None,
            "embedding": [0.1, 0.2],
            "distance": 0.12,
            "created_at": "2024-01-01T12:00:00Z",
            "state_metadata": {"step": 1},
            "metadata": {"scope": "team", "tags": ["runbook"]},
            "expires_at": None,
            "retention_policy": None,
            "lifecycle_status": "active",
            "retired_at": None,
            "retired_reason": None,
            "supersedes_id": None,
            "superseded_by_id": None,
        }
        if include_relationships:
            hit["relationships"] = [
                {"target_id": "doc-1#chunk-1", "relation": "cites", "weight": 0.75}
            ]
        return [hit]

    def retrieve(
        self,
        text: str | None,
        vector: list[float] | None,
        limit: int | None,
        filters_json: str | None,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
        fusion: str = "rrf",
    ):
        self.retrieve_calls.append(
            (
                text,
                vector,
                limit,
                filters_json,
                include_expired,
                include_retired,
                include_relationships,
                fusion,
            )
        )
        hit = self.search(
            vector or [0.0, 0.0],
            limit,
            filters_json,
            include_expired,
            include_retired,
            include_relationships,
        )[0]
        hit.pop("distance", None)
        hit["score"] = 0.032
        hit["vector_distance"] = 0.12 if vector is not None else None
        hit["text_score"] = 1.0 if text else None
        hit["matched_channels"] = [
            channel
            for channel, enabled in (
                ("vector", vector is not None),
                ("text", text is not None),
            )
            if enabled
        ]
        return [hit]

    def add_many(self, records: list[dict[str, Any]]):
        self.add_many_calls.append(records)

    def list(
        self,
        limit: int | None,
        offset: int | None,
        filters_json: str | None,
        include_expired: bool = False,
        include_retired: bool = False,
        include_binary: bool = True,
        include_embedding: bool = True,
    ):
        self.list_calls.append((limit, offset, filters_json))
        self.list_lifecycle_calls.append((include_expired, include_retired))
        self.list_projection_calls.append((include_binary, include_embedding))
        return [
            {
                "id": "rec-1",
                "external_id": "source-1",
                "run_id": "run-1",
                "bot_id": "support_bot",
                "session_id": "user_1",
                "role": "user",
                "content_type": "text/plain",
                "text_payload": "hello",
                "binary_payload": None,
                "embedding": [0.1, 0.2],
                "created_at": "2024-01-01T12:00:00Z",
                "state_metadata": {"step": 1},
                "metadata": {"scope": "team", "tags": ["runbook"]},
                "expires_at": None,
                "retention_policy": None,
                "lifecycle_status": "active",
                "retired_at": None,
                "retired_reason": None,
                "supersedes_id": None,
                "superseded_by_id": None,
            },
            {
                "id": "rec-2",
                "external_id": None,
                "run_id": "run-1",
                "bot_id": None,
                "session_id": None,
                "role": "assistant",
                "content_type": "text/plain",
                "text_payload": "world",
                "binary_payload": None,
                "embedding": None,
                "created_at": "2024-01-02T12:00:00Z",
                "state_metadata": None,
                "metadata": None,
                "expires_at": None,
                "retention_policy": None,
                "lifecycle_status": "active",
                "retired_at": None,
                "retired_reason": None,
                "supersedes_id": None,
                "superseded_by_id": None,
            },
        ]

    def related(
        self,
        target_id: str,
        relation: str | None,
        limit: int | None,
        include_expired: bool = False,
        include_retired: bool = False,
    ):
        self.related_calls.append(
            (target_id, relation, limit, include_expired, include_retired)
        )
        record = self.list(None, None, None)[0]
        record["relationships"] = [
            {"target_id": target_id, "relation": relation or "cites", "weight": None}
        ]
        return [record]


def _only_add_call(dummy: DummyInner):
    assert len(dummy.add_calls) == 1
    return dummy.add_calls[0]


def test_coerce_vector_from_list():
    assert _coerce_vector([1, 2.5]) == [1.0, 2.5]


def test_coerce_vector_rejects_invalid():
    with pytest.raises(TypeError):
        _coerce_vector("invalid")


def test_normalize_search_hit_converts_timestamp():
    result = _normalize_search_hit(
        {
            "id": "rec-2",
            "external_id": None,
            "created_at": "2024-01-01T00:00:00Z",
            "content_type": "text/plain",
            "text_payload": None,
            "binary_payload": None,
            "embedding": None,
            "distance": 0.5,
            "run_id": "run-2",
            "role": "assistant",
            "state_metadata": None,
        }
    )
    assert isinstance(result["created_at"], datetime)


def test_context_search_formats_results():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    hits = ctx.search([0.5, 0.4], limit=3)

    assert dummy.search_calls == [([0.5, 0.4], 3, None)]
    assert hits[0]["id"] == "rec-1"
    assert hits[0]["text"] == "hello"
    assert hits[0]["binary"] is None
    assert hits[0]["metadata"] == {"scope": "team", "tags": ["runbook"]}
    assert hits[0]["relationships"] == []
    assert isinstance(hits[0]["created_at"], datetime)


def test_context_search_forwards_filters():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.search([0.5, 0.4], filters={"bot_id": "support_bot", "scope": "team"})

    filters_json = dummy.search_calls[0][2]
    assert filters_json is not None
    assert json.loads(filters_json) == {"bot_id": "support_bot", "scope": "team"}


def test_context_search_passes_lifecycle_flags():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.search([0.5, 0.4], include_expired=True, include_retired=True)

    assert dummy.search_lifecycle_calls == [(True, True)]


def test_context_search_can_include_relationships():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    hits = ctx.search([0.5, 0.4], include_relationships=True)

    assert dummy.search_relationship_calls == [True]
    assert hits[0]["relationships"] == [
        {"target_id": "doc-1#chunk-1", "relation": "cites", "weight": 0.75}
    ]


def test_context_retrieve_forwards_hybrid_arguments():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    hits = ctx.retrieve(
        text="POLICY-123 service-a",
        vector=[0.5, 0.4],
        limit=3,
        filters={"bot_id": "support_bot", "scope": "team"},
        include_expired=True,
        include_retired=True,
        include_relationships=True,
    )

    assert dummy.retrieve_calls == [
        (
            "POLICY-123 service-a",
            [0.5, 0.4],
            3,
            '{"bot_id":"support_bot","scope":"team"}',
            True,
            True,
            True,
            "rrf",
        )
    ]
    assert hits[0]["score"] == 0.032
    assert hits[0]["vector_distance"] == 0.12
    assert hits[0]["text_score"] == 1.0
    assert hits[0]["matched_channels"] == ["vector", "text"]
    assert hits[0]["relationships"] == [
        {"target_id": "doc-1#chunk-1", "relation": "cites", "weight": 0.75}
    ]


def test_context_retrieve_accepts_text_only():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    hits = ctx.retrieve(text="runbook")

    assert dummy.retrieve_calls[0][0] == "runbook"
    assert dummy.retrieve_calls[0][1] is None
    assert hits[0]["matched_channels"] == ["text"]


def test_context_retrieve_requires_text_or_vector():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="requires text or vector"):
        ctx.retrieve()


def test_context_retrieve_rejects_unknown_fusion():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="supports only 'rrf'"):
        ctx.retrieve(text="runbook", fusion="weighted")


def test_normalize_record_without_distance():
    result = _normalize_record(
        {
            "id": "rec-1",
            "external_id": "source-1",
            "created_at": "2024-01-01T00:00:00Z",
            "content_type": "text/plain",
            "text_payload": "hello",
            "binary_payload": None,
            "embedding": None,
            "run_id": "run-1",
            "role": "user",
            "state_metadata": None,
        }
    )
    assert "distance" not in result
    assert result["text"] == "hello"
    assert result["relationships"] == []
    assert isinstance(result["created_at"], datetime)


def test_normalize_record_with_relationships():
    result = _normalize_record(
        {
            "id": "rec-1",
            "external_id": None,
            "created_at": "2024-01-01T00:00:00Z",
            "content_type": "text/plain",
            "text_payload": "hello",
            "binary_payload": None,
            "embedding": None,
            "run_id": "run-1",
            "role": "user",
            "state_metadata": None,
            "relationships": [
                {"target_id": "service-a", "relation": "mentions", "weight": None}
            ],
        }
    )

    assert result["relationships"] == [
        {"target_id": "service-a", "relation": "mentions", "weight": None}
    ]


def test_normalize_retrieve_hit_with_scores():
    result = _normalize_retrieve_hit(
        {
            "id": "rec-1",
            "external_id": None,
            "created_at": "2024-01-01T00:00:00Z",
            "content_type": "text/plain",
            "text_payload": "hello",
            "binary_payload": None,
            "embedding": None,
            "run_id": "run-1",
            "role": "user",
            "state_metadata": None,
            "score": 0.032,
            "vector_distance": 0.12,
            "text_score": 1.0,
            "matched_channels": ["vector", "text"],
        }
    )

    assert result["score"] == 0.032
    assert result["vector_distance"] == 0.12
    assert result["text_score"] == 1.0
    assert result["matched_channels"] == ["vector", "text"]


def test_context_list_returns_entries():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    entries = ctx.list(limit=10, offset=5)

    assert dummy.list_calls == [(10, 5, None)]
    assert len(entries) == 2
    assert entries[0]["id"] == "rec-1"
    assert entries[0]["text"] == "hello"
    assert entries[0]["role"] == "user"
    assert entries[0]["metadata"] == {"scope": "team", "tags": ["runbook"]}
    assert "distance" not in entries[0]
    assert entries[1]["id"] == "rec-2"
    assert entries[1]["text"] == "world"
    assert isinstance(entries[0]["created_at"], datetime)


def test_context_get_by_external_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    entry = ctx.get(external_id="source-1")

    assert dummy.get_calls == [(None, "source-1")]
    assert entry is not None
    assert entry["id"] == "rec-1"
    assert entry["external_id"] == "source-1"


def test_context_related_forwards_arguments():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    related = ctx.related(
        "doc-1#chunk-1",
        relation="cites",
        limit=5,
        include_expired=True,
        include_retired=True,
    )

    assert dummy.related_calls == [("doc-1#chunk-1", "cites", 5, True, True)]
    assert related[0]["relationships"] == [
        {"target_id": "doc-1#chunk-1", "relation": "cites", "weight": None}
    ]


def test_context_get_by_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    entry = ctx.get(id="rec-1")

    assert dummy.get_calls == [("rec-1", None)]
    assert entry is not None
    assert entry["id"] == "rec-1"


def test_context_get_missing_returns_none():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    assert ctx.get(external_id="missing") is None


def test_context_get_requires_exactly_one_identifier():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="exactly one"):
        ctx.get()
    with pytest.raises(ValueError, match="exactly one"):
        ctx.get(id="rec-1", external_id="source-1")


def test_context_delete_by_external_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    assert ctx.delete(external_id="source-1") is True
    assert dummy.delete_calls == [(None, "source-1")]


def test_context_delete_by_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    assert ctx.delete(id="rec-1") is True
    assert dummy.delete_calls == [("rec-1", None)]


def test_context_delete_missing_returns_false():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    assert ctx.delete(external_id="missing") is False


def test_context_forget_aliases_delete():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    assert ctx.forget(external_id="source-1") is True
    assert dummy.delete_calls == [(None, "source-1")]


def test_context_delete_requires_exactly_one_identifier():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="exactly one"):
        ctx.delete()
    with pytest.raises(ValueError, match="exactly one"):
        ctx.delete(id="rec-1", external_id="source-1")


def test_context_list_default_args():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.list()

    assert dummy.list_calls == [(None, None, None)]


def test_context_list_forwards_filters():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.list(filters={"role": "user", "tags": {"contains": "runbook"}})

    filters_json = dummy.list_calls[0][2]
    assert filters_json is not None
    assert json.loads(filters_json) == {
        "role": "user",
        "tags": {"contains": "runbook"},
    }


def test_context_list_passes_lifecycle_flags():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.list(include_expired=True, include_retired=True)

    assert dummy.list_lifecycle_calls == [(True, True)]


def test_context_add_with_embedding():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    embedding = [0.1, 0.2, 0.3]
    ctx.add("user", "hello", embedding=embedding)

    (
        role,
        content,
        data_type,
        passed_embedding,
        bot_id,
        session_id,
        external_id,
        metadata_json,
    ) = _only_add_call(dummy)
    assert role == "user"
    assert content == "hello"
    assert data_type is None
    assert passed_embedding == [0.1, 0.2, 0.3]
    assert bot_id is None
    assert session_id is None
    assert external_id is None
    assert metadata_json is None


def test_context_add_without_embedding():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("assistant", "world")

    (
        role,
        content,
        data_type,
        passed_embedding,
        bot_id,
        session_id,
        external_id,
        metadata_json,
    ) = _only_add_call(dummy)
    assert role == "assistant"
    assert content == "world"
    assert passed_embedding is None
    assert bot_id is None
    assert session_id is None
    assert external_id is None
    assert metadata_json is None


def test_context_add_with_content_type_and_embedding():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    embedding = [0.5, 0.6]
    ctx.add("system", "prompt", content_type="text/markdown", embedding=embedding)

    (
        role,
        content,
        data_type,
        passed_embedding,
        bot_id,
        session_id,
        external_id,
        metadata_json,
    ) = _only_add_call(dummy)
    assert role == "system"
    assert data_type == "text/markdown"
    assert passed_embedding == [0.5, 0.6]
    assert bot_id is None
    assert session_id is None
    assert external_id is None
    assert metadata_json is None


def test_context_add_with_bot_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("user", "hello", bot_id="support_bot")

    (
        role,
        content,
        data_type,
        passed_embedding,
        bot_id,
        session_id,
        external_id,
        metadata_json,
    ) = _only_add_call(dummy)
    assert role == "user"
    assert content == "hello"
    assert bot_id == "support_bot"
    assert session_id is None
    assert external_id is None
    assert metadata_json is None


def test_context_add_with_session_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("user", "hello", session_id="user_123")

    (
        role,
        content,
        data_type,
        passed_embedding,
        bot_id,
        session_id,
        external_id,
        metadata_json,
    ) = _only_add_call(dummy)
    assert role == "user"
    assert content == "hello"
    assert bot_id is None
    assert session_id == "user_123"
    assert external_id is None
    assert metadata_json is None


def test_context_add_with_agent_and_session_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("user", "hello", bot_id="sales_bot", session_id="conv_456")

    (
        role,
        content,
        data_type,
        passed_embedding,
        bot_id,
        session_id,
        external_id,
        metadata_json,
    ) = _only_add_call(dummy)
    assert role == "user"
    assert bot_id == "sales_bot"
    assert session_id == "conv_456"
    assert external_id is None
    assert metadata_json is None


def test_context_add_with_all_options():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    embedding = [0.1, 0.2]
    ctx.add(
        "user",
        "hello",
        embedding=embedding,
        bot_id="bot",
        session_id="sess",
        external_id="doc-1#chunk-1",
        metadata={
            "tenant": "example-org",
            "scope": "team",
            "tags": ["runbook", "ownership"],
            "confidence": 0.92,
        },
    )

    (
        role,
        content,
        data_type,
        passed_embedding,
        bot_id,
        session_id,
        external_id,
        metadata_json,
    ) = _only_add_call(dummy)
    assert role == "user"
    assert passed_embedding == [0.1, 0.2]
    assert bot_id == "bot"
    assert session_id == "sess"
    assert external_id == "doc-1#chunk-1"
    assert metadata_json is not None
    assert json.loads(metadata_json) == {
        "tenant": "example-org",
        "scope": "team",
        "tags": ["runbook", "ownership"],
        "confidence": 0.92,
    }


def test_context_add_forwards_relationships():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add(
        "user",
        "hello",
        relationships=[
            {"target_id": "doc-1#chunk-1", "relation": "cites", "weight": 0.75},
            {"target_id": "service-a", "relation": "mentions"},
        ],
    )

    relationships_json = dummy.relationship_add_calls[0]
    assert relationships_json is not None
    assert json.loads(relationships_json) == [
        {"relation": "cites", "target_id": "doc-1#chunk-1", "weight": 0.75},
        {"relation": "mentions", "target_id": "service-a"},
    ]


def test_context_add_forwards_state_metadata():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add(
        "assistant",
        "step complete",
        state_metadata={
            "step": 3,
            "active_plan_id": "plan-1",
            "tokens_used": 128,
            "custom": "retrieval",
        },
    )

    assert dummy.state_metadata_add_calls == [
        {
            "step": 3,
            "active_plan_id": "plan-1",
            "tokens_used": 128,
            "custom": "retrieval",
        }
    ]


def test_context_add_rejects_non_json_metadata():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(TypeError, match="metadata must be JSON-serializable"):
        ctx.add("user", "hello", metadata={"bad": object()})


def test_context_add_with_lifecycle_fields():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    expires_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    retired_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    ctx.add(
        "user",
        "hello",
        expires_at=expires_at,
        retention_policy="ttl:30d",
        lifecycle_status="active",
        retired_at=retired_at,
        retired_reason="manual cleanup",
        supersedes_id="old-id",
        superseded_by_id="new-id",
    )

    assert dummy.lifecycle_add_calls == [
        {
            "expires_at": "2026-07-01T00:00:00Z",
            "retention_policy": "ttl:30d",
            "lifecycle_status": "active",
            "retired_at": "2026-08-01T00:00:00Z",
            "retired_reason": "manual cleanup",
            "supersedes_id": "old-id",
            "superseded_by_id": "new-id",
        }
    ]


def test_context_add_rejects_naive_lifecycle_datetime():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="timezone"):
        ctx.add("user", "hello", expires_at=datetime(2026, 7, 1))


def test_context_upsert_requires_external_id_and_supported_key():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="external_id"):
        ctx.upsert("user", "hello")
    with pytest.raises(ValueError, match="Only key='external_id'"):
        ctx.upsert("user", "hello", external_id="source-1", key="id")
    assert dummy.upsert_calls == []


def test_context_upsert_returns_operation_metadata_and_record():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    result = ctx.upsert(
        "user",
        "new value",
        embedding=[0.1, 0.2],
        external_id="source-1",
        metadata={"revision": 2},
        relationships=[{"target_id": "doc-1", "relation": "updates"}],
        expires_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
    )

    assert dummy.upsert_calls == [
        {
            "role": "user",
            "content": "new value",
            "data_type": None,
            "embedding": [0.1, 0.2],
            "bot_id": None,
            "session_id": None,
            "external_id": "source-1",
            "metadata_json": '{"revision":2}',
            "expires_at": "2026-07-01T00:00:00Z",
            "retention_policy": None,
            "lifecycle_status": None,
            "retired_at": None,
            "retired_reason": None,
            "relationships_json": '[{"relation":"updates","target_id":"doc-1"}]',
            "key": "external_id",
        }
    ]
    assert result["inserted"] is False
    assert result["replaced_id"] == "old-id"
    assert result["version"] == 7
    assert result["record"]["id"] == "new-id"
    assert result["record"]["text"] == "new value"
    assert result["record"]["metadata"] == {"revision": 2}
    assert result["record"]["supersedes_id"] == "old-id"
    assert isinstance(result["record"]["created_at"], datetime)


def test_context_update_requires_identifier_and_patch():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="exactly one"):
        ctx.update(metadata={"revision": 2})
    with pytest.raises(ValueError, match="exactly one"):
        ctx.update(id="rec-1", external_id="source-1", metadata={"revision": 2})
    with pytest.raises(ValueError, match="at least one patch field"):
        ctx.update(external_id="source-1")
    assert dummy.update_calls == []


def test_context_update_returns_operation_metadata_and_record():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    result = ctx.update(
        external_id="source-1",
        bot_id="bot",
        session_id="sess",
        metadata={"revision": 2},
        relationships=[{"target_id": "doc-1", "relation": "updates"}],
        expires_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        lifecycle_status="active",
    )

    assert dummy.update_calls == [
        {
            "id": None,
            "external_id": "source-1",
            "bot_id": "bot",
            "session_id": "sess",
            "metadata_json": '{"revision":2}',
            "relationships_json": '[{"relation":"updates","target_id":"doc-1"}]',
            "expires_at": "2026-07-01T00:00:00Z",
            "retention_policy": None,
            "lifecycle_status": "active",
            "retired_at": None,
            "retired_reason": None,
            "embedding": None,
        }
    ]
    assert result["updated"] is True
    assert result["replaced_id"] == "old-id"
    assert result["version"] == 8
    assert result["record"]["id"] == "new-id"
    assert result["record"]["text"] == "stable content"
    assert result["record"]["metadata"] == {"revision": 2}
    assert result["record"]["relationships"] == [
        {"target_id": "doc-1", "relation": "updates"}
    ]
    assert result["record"]["supersedes_id"] == "old-id"


def test_context_update_forwards_embedding():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.update(external_id="source-1", embedding=[0.1, 0.2, 0.3])

    assert dummy.update_calls[0]["embedding"] == [0.1, 0.2, 0.3]


def test_context_update_accepts_embedding_only_patch():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    # An embedding is sufficient on its own; no "at least one patch field" error.
    result = ctx.update(id="rec-1", embedding=[0.1, 0.2])

    assert dummy.update_calls[0]["embedding"] == [0.1, 0.2]
    assert result["updated"] is True


def test_context_update_missing_record_returns_not_updated():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    result = ctx.update(external_id="missing", metadata={"revision": 2})

    assert result == {
        "updated": False,
        "replaced_id": None,
        "version": 7,
        "record": None,
    }


def test_context_add_many_normalizes_records():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add_many(
        [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "world",
                "content_type": "text/markdown",
                "embedding": [0.1, 0.2],
                "bot_id": "bot",
                "session_id": "sess",
                "external_id": "doc-1#chunk-2",
            },
        ]
    )

    assert dummy.add_many_calls == [
        [
            {
                "role": "user",
                "content": "hello",
                "data_type": None,
                "embedding": None,
                "bot_id": None,
                "session_id": None,
                "tenant": None,
                "source": None,
                "external_id": None,
                "run_id": None,
                "created_at": None,
                "state_metadata": None,
                "metadata_json": None,
                "relationships_json": None,
                "expires_at": None,
                "retention_policy": None,
                "lifecycle_status": None,
                "retired_at": None,
                "retired_reason": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "payload_uri": None,
                "payload_size": None,
                "payload_checksum": None,
            },
            {
                "role": "assistant",
                "content": "world",
                "data_type": "text/markdown",
                "embedding": [0.1, 0.2],
                "bot_id": "bot",
                "session_id": "sess",
                "tenant": None,
                "source": None,
                "external_id": "doc-1#chunk-2",
                "run_id": None,
                "created_at": None,
                "state_metadata": None,
                "metadata_json": None,
                "relationships_json": None,
                "expires_at": None,
                "retention_policy": None,
                "lifecycle_status": None,
                "retired_at": None,
                "retired_reason": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "payload_uri": None,
                "payload_size": None,
                "payload_checksum": None,
            },
        ]
    ]


def test_context_add_many_accepts_data_type_alias():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add_many([{"role": "system", "content": "prompt", "data_type": "text/plain"}])

    assert dummy.add_many_calls[0][0]["data_type"] == "text/plain"


def test_context_add_many_forwards_metadata():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add_many(
        [
            {
                "role": "user",
                "content": "hello",
                "metadata": {"scope": "team", "tags": ["runbook"]},
            }
        ]
    )

    metadata_json = dummy.add_many_calls[0][0]["metadata_json"]
    assert metadata_json is not None
    assert json.loads(metadata_json) == {"scope": "team", "tags": ["runbook"]}


def test_context_add_many_forwards_relationships():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add_many(
        [
            {
                "role": "user",
                "content": "hello",
                "relationships": [{"target_id": "doc-1#chunk-1", "relation": "cites"}],
            }
        ]
    )

    relationships_json = dummy.add_many_calls[0][0]["relationships_json"]
    assert relationships_json is not None
    assert json.loads(relationships_json) == [
        {"relation": "cites", "target_id": "doc-1#chunk-1"}
    ]


def test_context_add_many_passes_lifecycle_fields():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add_many(
        [
            {
                "role": "user",
                "content": "hello",
                "expires_at": datetime(2026, 7, 1, tzinfo=timezone.utc),
                "retention_policy": "ttl:30d",
                "lifecycle_status": "superseded",
                "superseded_by_id": "new-id",
            }
        ]
    )

    record = dummy.add_many_calls[0][0]
    assert record["expires_at"] == "2026-07-01T00:00:00Z"
    assert record["retention_policy"] == "ttl:30d"
    assert record["lifecycle_status"] == "superseded"
    assert record["superseded_by_id"] == "new-id"


def test_context_add_many_rejects_invalid_records():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    with pytest.raises(TypeError, match="records\\[0\\]"):
        ctx.add_many(["not-a-record"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="missing required key 'role'"):
        ctx.add_many([{"content": "hello"}])
    with pytest.raises(ValueError, match="missing required key 'content'"):
        ctx.add_many([{"role": "user"}])
    with pytest.raises(ValueError, match="both content_type and data_type"):
        ctx.add_many(
            [
                {
                    "role": "user",
                    "content": "hello",
                    "content_type": "text/plain",
                    "data_type": "text/markdown",
                }
            ]
        )


def test_normalize_record_with_agent_and_session_id():
    result = _normalize_record(
        {
            "id": "rec-1",
            "external_id": "source-1",
            "created_at": "2024-01-01T00:00:00Z",
            "content_type": "text/plain",
            "text_payload": "hello",
            "binary_payload": None,
            "embedding": None,
            "run_id": "run-1",
            "bot_id": "support_bot",
            "session_id": "user_88",
            "role": "user",
            "state_metadata": None,
        }
    )
    assert result["bot_id"] == "support_bot"
    assert result["session_id"] == "user_88"
    assert result["external_id"] == "source-1"


def test_normalize_record_with_lifecycle_fields():
    result = _normalize_record(
        {
            "id": "rec-1",
            "external_id": None,
            "created_at": "2024-01-01T00:00:00Z",
            "content_type": "text/plain",
            "text_payload": "hello",
            "binary_payload": None,
            "embedding": None,
            "run_id": "run-1",
            "role": "user",
            "state_metadata": None,
            "metadata": None,
            "expires_at": "2026-07-01T00:00:00Z",
            "retention_policy": "ttl:30d",
            "lifecycle_status": "superseded",
            "retired_at": "2026-08-01T00:00:00Z",
            "retired_reason": "manual cleanup",
            "supersedes_id": "old-id",
            "superseded_by_id": "new-id",
        }
    )

    assert isinstance(result["expires_at"], datetime)
    assert isinstance(result["retired_at"], datetime)
    assert result["lifecycle_status"] == "superseded"
    assert result["retention_policy"] == "ttl:30d"
    assert result["supersedes_id"] == "old-id"
    assert result["superseded_by_id"] == "new-id"
