from datetime import datetime
from typing import Any

import pytest
from lance_context.api import Context, _coerce_vector, _normalize_record, _normalize_search_hit


class DummyInner:
    def __init__(self) -> None:
        self.search_calls: list[tuple[list[float], int | None]] = []
        self.list_calls: list[tuple[int | None, int | None]] = []
        self.add_calls: list[tuple[str, Any, str | None, list[float] | None, str | None, str | None]] = []

    def add(
        self,
        role: str,
        content: Any,
        data_type: str | None,
        embedding: list[float] | None,
        bot_id: str | None,
        session_id: str | None,
    ):
        self.add_calls.append((role, content, data_type, embedding, bot_id, session_id))

    def search(self, vector: list[float], limit: int | None):
        self.search_calls.append((vector, limit))
        return [
            {
                "id": "rec-1",
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
            }
        ]

    def list(self, limit: int | None, offset: int | None):
        self.list_calls.append((limit, offset))
        return [
            {
                "id": "rec-1",
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
            },
            {
                "id": "rec-2",
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
            },
        ]


def test_coerce_vector_from_list():
    assert _coerce_vector([1, 2.5]) == [1.0, 2.5]


def test_coerce_vector_rejects_invalid():
    with pytest.raises(TypeError):
        _coerce_vector("invalid")


def test_normalize_search_hit_converts_timestamp():
    result = _normalize_search_hit(
        {
            "id": "rec-2",
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

    assert dummy.search_calls == [([0.5, 0.4], 3)]
    assert hits[0]["id"] == "rec-1"
    assert hits[0]["text"] == "hello"
    assert hits[0]["binary"] is None
    assert isinstance(hits[0]["created_at"], datetime)


def test_normalize_record_without_distance():
    result = _normalize_record(
        {
            "id": "rec-1",
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
    assert isinstance(result["created_at"], datetime)


def test_context_list_returns_entries():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    entries = ctx.list(limit=10, offset=5)

    assert dummy.list_calls == [(10, 5)]
    assert len(entries) == 2
    assert entries[0]["id"] == "rec-1"
    assert entries[0]["text"] == "hello"
    assert entries[0]["role"] == "user"
    assert "distance" not in entries[0]
    assert entries[1]["id"] == "rec-2"
    assert entries[1]["text"] == "world"
    assert isinstance(entries[0]["created_at"], datetime)


def test_context_list_default_args():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.list()

    assert dummy.list_calls == [(None, None)]


def test_context_add_with_embedding():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    embedding = [0.1, 0.2, 0.3]
    ctx.add("user", "hello", embedding=embedding)

    assert len(dummy.add_calls) == 1
    role, content, data_type, passed_embedding, bot_id, session_id = dummy.add_calls[0]
    assert role == "user"
    assert content == "hello"
    assert data_type is None
    assert passed_embedding == [0.1, 0.2, 0.3]
    assert bot_id is None
    assert session_id is None


def test_context_add_without_embedding():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("assistant", "world")

    assert len(dummy.add_calls) == 1
    role, content, data_type, passed_embedding, bot_id, session_id = dummy.add_calls[0]
    assert role == "assistant"
    assert content == "world"
    assert passed_embedding is None
    assert bot_id is None
    assert session_id is None


def test_context_add_with_content_type_and_embedding():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    embedding = [0.5, 0.6]
    ctx.add("system", "prompt", content_type="text/markdown", embedding=embedding)

    assert len(dummy.add_calls) == 1
    role, content, data_type, passed_embedding, bot_id, session_id = dummy.add_calls[0]
    assert role == "system"
    assert data_type == "text/markdown"
    assert passed_embedding == [0.5, 0.6]
    assert bot_id is None
    assert session_id is None


def test_context_add_with_bot_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("user", "hello", bot_id="support_bot")

    assert len(dummy.add_calls) == 1
    role, content, data_type, passed_embedding, bot_id, session_id = dummy.add_calls[0]
    assert role == "user"
    assert content == "hello"
    assert bot_id == "support_bot"
    assert session_id is None


def test_context_add_with_session_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("user", "hello", session_id="user_123")

    assert len(dummy.add_calls) == 1
    role, content, data_type, passed_embedding, bot_id, session_id = dummy.add_calls[0]
    assert role == "user"
    assert content == "hello"
    assert bot_id is None
    assert session_id == "user_123"


def test_context_add_with_agent_and_session_id():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    ctx.add("user", "hello", bot_id="sales_bot", session_id="conv_456")

    assert len(dummy.add_calls) == 1
    role, content, data_type, passed_embedding, bot_id, session_id = dummy.add_calls[0]
    assert role == "user"
    assert bot_id == "sales_bot"
    assert session_id == "conv_456"


def test_context_add_with_all_options():
    ctx = Context.__new__(Context)
    dummy = DummyInner()
    ctx._inner = dummy  # type: ignore[attr-defined]

    embedding = [0.1, 0.2]
    ctx.add("user", "hello", embedding=embedding, bot_id="bot", session_id="sess")

    assert len(dummy.add_calls) == 1
    role, content, data_type, passed_embedding, bot_id, session_id = dummy.add_calls[0]
    assert role == "user"
    assert passed_embedding == [0.1, 0.2]
    assert bot_id == "bot"
    assert session_id == "sess"


def test_normalize_record_with_agent_and_session_id():
    result = _normalize_record(
        {
            "id": "rec-1",
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
