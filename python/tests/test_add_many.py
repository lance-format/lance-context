"""Tests for batch append support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def test_add_many_appends_records_in_one_call(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add_many(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "second",
                "content_type": "text/markdown",
                "bot_id": "bot",
                "session_id": "session",
                "external_id": "doc-1#chunk-2",
            },
            {
                "role": "tool",
                "content": b"\x01\x02",
                "data_type": "application/octet-stream",
            },
        ]
    )

    assert ctx.entries() == 3
    records = ctx.list()
    assert [record["role"] for record in records] == ["user", "assistant", "tool"]
    assert [record["text"] for record in records[:2]] == ["first", "second"]
    assert records[1]["content_type"] == "text/markdown"
    assert records[1]["bot_id"] == "bot"
    assert records[1]["session_id"] == "session"
    assert records[1]["external_id"] == "doc-1#chunk-2"
    assert records[2]["binary"] == b"\x01\x02"


def test_add_many_empty_batch_is_noop(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add_many([])

    assert ctx.entries() == 0
    assert ctx.list() == []


def test_add_many_validates_records_before_write(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    with pytest.raises(ValueError, match="missing required key 'content'"):
        ctx.add_many([{"role": "user"}])

    assert ctx.entries() == 0
    assert ctx.list() == []
