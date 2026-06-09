"""Tests for logical deletion / forgetting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def _embedding(value: float) -> list[float]:
    vector = [0.0] * 1536
    vector[0] = value
    return vector


def test_delete_by_external_id_hides_record_from_default_reads(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add(
        "user",
        "stale memory",
        embedding=_embedding(0.0),
        external_id="doc-123#chunk-4",
    )
    ctx.add(
        "user",
        "fresh memory",
        embedding=_embedding(1.0),
        external_id="doc-456#chunk-1",
    )

    stale = ctx.get(external_id="doc-123#chunk-4")
    assert stale is not None
    stale_id = stale["id"]

    assert ctx.delete(external_id="doc-123#chunk-4") is True
    assert ctx.delete(external_id="doc-123#chunk-4") is False
    assert ctx.get(external_id="doc-123#chunk-4") is None
    assert ctx.get(id=stale_id) is None

    entries = ctx.list()
    assert [entry["external_id"] for entry in entries] == ["doc-456#chunk-1"]

    hits = ctx.search(_embedding(0.0), limit=10)
    assert [hit["external_id"] for hit in hits] == ["doc-456#chunk-1"]


def test_delete_by_id_hides_record_from_default_reads(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add(
        "user",
        "stale memory",
        embedding=_embedding(0.0),
        external_id="doc-123#chunk-4",
    )
    ctx.add(
        "user",
        "fresh memory",
        embedding=_embedding(1.0),
        external_id="doc-456#chunk-1",
    )

    stale = ctx.get(external_id="doc-123#chunk-4")
    assert stale is not None
    stale_id = stale["id"]

    assert ctx.delete(id=stale_id) is True
    assert ctx.delete(id=stale_id) is False
    assert ctx.get(id=stale_id) is None
    assert ctx.get(external_id="doc-123#chunk-4") is None

    entries = ctx.list()
    assert [entry["external_id"] for entry in entries] == ["doc-456#chunk-1"]

    hits = ctx.search(_embedding(0.0), limit=10)
    assert [hit["external_id"] for hit in hits] == ["doc-456#chunk-1"]


def test_external_id_can_be_reused_after_delete(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add("user", "stale memory", external_id="doc-123#chunk-4")
    assert ctx.forget(external_id="doc-123#chunk-4") is True

    ctx.add("user", "replacement memory", external_id="doc-123#chunk-4")

    entry = ctx.get(external_id="doc-123#chunk-4")
    assert entry is not None
    assert entry["text"] == "replacement memory"
    assert [item["text"] for item in ctx.list()] == ["replacement memory"]
