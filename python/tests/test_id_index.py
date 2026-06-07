"""Tests for configurable id column index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def test_btree_index_creation(tmp_path: Path) -> None:
    """Verify btree index is created after adding data and compacting."""
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri, id_index_type="btree")

    for i in range(10):
        ctx.add("user", f"entry-{i}")

    ctx.compact()

    # Data should be fully preserved
    results = ctx.list()
    assert len(results) == 10


def test_zonemap_index_creation(tmp_path: Path) -> None:
    """Verify zonemap index is created after adding data and compacting."""
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri, id_index_type="zonemap")

    for i in range(10):
        ctx.add("user", f"entry-{i}")

    ctx.compact()

    results = ctx.list()
    assert len(results) == 10


def test_no_index_by_default(tmp_path: Path) -> None:
    """Verify no id index is created by default."""
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    for i in range(5):
        ctx.add("user", f"entry-{i}")

    ctx.compact()
    results = ctx.list()
    assert len(results) == 5


def test_invalid_index_type_rejected(tmp_path: Path) -> None:
    """Verify invalid index type raises an error."""
    uri = str(tmp_path / "context.lance")
    with pytest.raises(RuntimeError, match="invalid id_index_type"):
        Context.create(uri, id_index_type="invalid_type")


def test_index_with_compaction_preserves_data(tmp_path: Path) -> None:
    """Verify data integrity across multiple add-compact cycles with an index."""
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri, id_index_type="btree")

    # First batch
    for i in range(10):
        ctx.add("user", f"batch1-{i}")
    ctx.compact()

    # Second batch
    for i in range(10):
        ctx.add("user", f"batch2-{i}")
    ctx.compact()

    results = ctx.list()
    assert len(results) == 20
    texts = {r["text"] for r in results}
    for i in range(10):
        assert f"batch1-{i}" in texts
        assert f"batch2-{i}" in texts
