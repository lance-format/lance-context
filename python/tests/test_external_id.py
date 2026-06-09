"""Tests for caller-supplied external record identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def test_external_id_roundtrip_and_lookup(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add("user", "stable memory", external_id="doc-123#chunk-1")

    entries = ctx.list()
    assert entries[0]["external_id"] == "doc-123#chunk-1"

    by_external_id = ctx.get(external_id="doc-123#chunk-1")
    assert by_external_id is not None
    assert by_external_id["text"] == "stable memory"
    assert by_external_id["external_id"] == "doc-123#chunk-1"

    by_id = ctx.get(id=by_external_id["id"])
    assert by_id is not None
    assert by_id["external_id"] == "doc-123#chunk-1"


def test_external_id_missing_lookup_returns_none(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add("user", "stable memory", external_id="doc-123#chunk-1")

    assert ctx.get(external_id="missing") is None


def test_duplicate_external_id_is_rejected(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add("user", "first", external_id="doc-123#chunk-1")

    with pytest.raises(RuntimeError, match="external_id.*already exists"):
        ctx.add("user", "duplicate", external_id="doc-123#chunk-1")
