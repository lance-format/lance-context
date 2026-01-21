from datetime import datetime

import pytest
from lance_context.api import Context, _coerce_vector, _normalize_record, _normalize_search_hit


class DummyInner:
    def __init__(self) -> None:
        self.search_calls: list[tuple[list[float], int | None]] = []
        self.list_calls: list[tuple[int | None, int | None]] = []

    def search(self, vector: list[float], limit: int | None):
        self.search_calls.append((vector, limit))
        return [
            {
                "id": "rec-1",
                "run_id": "run-1",
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
