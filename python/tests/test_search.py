from datetime import datetime

import pytest

from lance_context.api import Context, _coerce_vector, _normalize_search_hit


class DummyInner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int | None]] = []

    def search(self, vector: list[float], limit: int | None):
        self.calls.append((vector, limit))
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

    assert dummy.calls == [([0.5, 0.4], 3)]
    assert hits[0]["id"] == "rec-1"
    assert hits[0]["text"] == "hello"
    assert hits[0]["binary"] is None
    assert isinstance(hits[0]["created_at"], datetime)
