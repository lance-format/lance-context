"""Tests for raw interaction-log ingestion helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def test_ingest_records_maps_raw_rows_and_preserves_provenance(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "context.lance"))
    created_at = "2026-06-23T20:15:00Z"

    result = ctx.ingest_records(
        [
            {
                "event_id": "chat-1#turn-1",
                "speaker": "user",
                "message": "hello",
                "conversation": "chat-1",
                "trace_id": "run-1",
                "timestamp": created_at,
            }
        ],
        field_map={
            "external_id": "event_id",
            "role": "speaker",
            "content": "message",
            "session_id": "conversation",
            "run_id": "trace_id",
            "created_at": "timestamp",
        },
        defaults={"tenant": "acme"},
    )

    assert result == {"processed": 1, "inserted": 1, "updated": 0, "batches": 1}
    [record] = ctx.list()
    assert record["external_id"] == "chat-1#turn-1"
    assert record["role"] == "user"
    assert record["text"] == "hello"
    assert record["source"] == "raw"
    assert record["tenant"] == "acme"
    assert record["session_id"] == "chat-1"
    assert record["run_id"] == "run-1"
    assert record["created_at"] == datetime(2026, 6, 23, 20, 15, tzinfo=timezone.utc)
    assert record["metadata"]["raw_record"]["event_id"] == "chat-1#turn-1"


def test_ingest_jsonl_streams_batches_and_upserts_idempotently(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "context.lance"))
    path = tmp_path / "chat.jsonl"
    rows = [
        {"id": "turn-1", "role": "user", "text": "first"},
        {"id": "turn-2", "role": "assistant", "text": "second"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    first = ctx.ingest_jsonl(
        path,
        field_map={"external_id": "id", "content": "text"},
        mode="upsert",
        batch_size=1,
    )
    second = ctx.ingest_jsonl(
        path,
        field_map={"external_id": "id", "content": "text"},
        mode="upsert",
        batch_size=1,
    )

    assert first == {"processed": 2, "inserted": 2, "updated": 0, "batches": 2}
    assert second == {"processed": 2, "inserted": 0, "updated": 2, "batches": 2}
    records = ctx.list()
    assert len(records) == 2
    assert {record["external_id"] for record in records} == {"turn-1", "turn-2"}
    first_turn = ctx.get(external_id="turn-1")
    assert first_turn is not None
    assert first_turn["text"] == "first"


def test_ingest_messages_loads_openai_style_chat(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "context.lance"))

    result = ctx.ingest_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "status?"},
            {"role": "assistant", "tool_calls": [{"name": "lookup", "arguments": {}}]},
        ],
        session_id="session-1",
        tenant="acme",
        external_id_prefix="chat-99",
    )

    assert result == {"processed": 3, "inserted": 3, "updated": 0, "batches": 1}
    records = ctx.list()
    assert [record["role"] for record in records] == ["system", "user", "assistant"]
    assert [record["external_id"] for record in records] == [
        "chat-99#message-1",
        "chat-99#message-2",
        "chat-99#message-3",
    ]
    assert {record["source"] for record in records} == {"raw"}
    assert {record["session_id"] for record in records} == {"session-1"}
    assert records[2]["content_type"] == "application/json"
    assert records[2]["metadata"]["raw_message"]["tool_calls"][0]["name"] == "lookup"


def test_upsert_ingestion_requires_external_id(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "context.lance"))

    try:
        ctx.ingest_records([{"content": "missing id"}], mode="upsert")
    except ValueError as exc:
        assert "requires external_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected upsert ingestion to require external_id")
