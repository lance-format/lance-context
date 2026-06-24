"""Tests for batch insert-or-replace (upsert_many)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def test_upsert_many_inserts_new_records(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    results = ctx.upsert_many(
        [
            {"role": "user", "content": "a", "external_id": "ext-a"},
            {"role": "user", "content": "b", "external_id": "ext-b"},
        ]
    )

    assert len(results) == 2
    assert all(r["inserted"] for r in results)
    assert all(r["replaced_id"] is None for r in results)
    assert len(ctx.list()) == 2


def test_upsert_many_replaces_and_is_idempotent(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.upsert_many(
        [
            {"role": "user", "content": "a-old", "external_id": "ext-a"},
            {"role": "user", "content": "b-old", "external_id": "ext-b"},
        ]
    )
    results = ctx.upsert_many(
        [
            {"role": "user", "content": "a-new", "external_id": "ext-a"},
            {"role": "user", "content": "b-new", "external_id": "ext-b"},
        ]
    )

    assert all(not r["inserted"] for r in results)
    assert all(r["replaced_id"] is not None for r in results)

    # Only the successors remain visible, one per external_id.
    visible = ctx.list()
    assert len(visible) == 2
    assert ctx.get(external_id="ext-a")["text"] == "a-new"
    assert ctx.get(external_id="ext-b")["text"] == "b-new"


def test_upsert_many_mixed_insert_and_replace(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.upsert_many([{"role": "user", "content": "a", "external_id": "ext-a"}])
    results = ctx.upsert_many(
        [
            {"role": "user", "content": "a2", "external_id": "ext-a"},
            {"role": "user", "content": "c", "external_id": "ext-c"},
        ]
    )

    assert not results[0]["inserted"]
    assert results[0]["replaced_id"] is not None
    assert results[1]["inserted"]
    assert len(ctx.list()) == 2


def test_upsert_many_rejects_within_batch_duplicate_external_id(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    with pytest.raises(RuntimeError, match="duplicate external_id"):
        ctx.upsert_many(
            [
                {"role": "user", "content": "a", "external_id": "dup"},
                {"role": "user", "content": "b", "external_id": "dup"},
            ]
        )

    # All-or-nothing: nothing was written.
    assert ctx.list() == []


def test_upsert_many_requires_external_id(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    with pytest.raises(ValueError, match="external_id"):
        ctx.upsert_many([{"role": "user", "content": "a"}])


def test_upsert_many_empty_batch_returns_empty(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    assert ctx.upsert_many([]) == []
