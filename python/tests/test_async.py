"""Tests for AsyncContext wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context import AsyncContext


@pytest.mark.asyncio
async def test_create_and_add(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    await ctx.add("user", "hello")
    await ctx.add("assistant", "hi there")

    assert ctx.entries() == 2
    assert ctx.uri() == uri


@pytest.mark.asyncio
async def test_list(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    for i in range(5):
        await ctx.add("user", f"msg-{i}")

    results = await ctx.list()
    assert len(results) == 5
    texts = {r["text"] for r in results}
    for i in range(5):
        assert f"msg-{i}" in texts


@pytest.mark.asyncio
async def test_list_with_limit_and_offset(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    for i in range(10):
        await ctx.add("user", f"msg-{i}")

    page = await ctx.list(limit=3, offset=2)
    assert len(page) == 3


@pytest.mark.asyncio
async def test_compact(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    for i in range(10):
        await ctx.add("user", f"entry-{i}")

    metrics = await ctx.compact()
    assert isinstance(metrics, dict)
    assert "fragments_removed" in metrics


@pytest.mark.asyncio
async def test_compaction_stats(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)
    await ctx.add("user", "hello")

    stats = await ctx.compaction_stats()
    assert isinstance(stats, dict)
    assert "total_fragments" in stats


@pytest.mark.asyncio
async def test_snapshot_and_checkout(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    await ctx.add("user", "v1")
    v1 = ctx.version()

    await ctx.add("user", "v2")
    assert ctx.entries() == 2

    await ctx.checkout(v1)


@pytest.mark.asyncio
async def test_fork(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    await ctx.add("user", "main-msg")
    forked = ctx.fork("experiment")

    assert forked.branch() == "experiment"
    assert isinstance(forked, AsyncContext)


@pytest.mark.asyncio
async def test_search(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    dim = 1536
    emb_a = [1.0] + [0.0] * (dim - 1)
    emb_b = [0.0] + [1.0] + [0.0] * (dim - 2)

    await ctx.add("user", "hello", embedding=emb_a)
    await ctx.add("user", "world", embedding=emb_b)

    results = await ctx.search(emb_a, limit=1)
    assert len(results) == 1
    assert results[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_retrieve(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    dim = 1536
    near = [0.0] * dim
    far = [0.0] * dim
    far[0] = 1.0

    await ctx.add("assistant", "general rollout guidance", embedding=near)
    await ctx.add("assistant", "POLICY-123 blocks service-a", embedding=far)

    results = await ctx.retrieve(text="POLICY-123 service-a", vector=near, limit=1)
    assert len(results) == 1
    assert results[0]["text"] == "POLICY-123 blocks service-a"
    assert results[0]["matched_channels"] == ["vector", "text"]


@pytest.mark.asyncio
async def test_metadata_filters(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)

    dim = 1536
    near = [0.0] * dim
    far = [0.0] * dim
    far[0] = 10.0

    await ctx.add(
        "assistant",
        "global nearest",
        embedding=near,
        metadata={"scope": "personal"},
    )
    await ctx.add(
        "assistant",
        "scoped farther",
        embedding=far,
        session_id="incident-1",
        external_id="runbook-1",
        metadata={"scope": "team", "tags": ["runbook"]},
    )

    entries = await ctx.list(filters={"scope": "team"})
    assert len(entries) == 1
    assert entries[0]["external_id"] == "runbook-1"
    assert entries[0]["metadata"] == {"scope": "team", "tags": ["runbook"]}

    results = await ctx.search(
        near,
        limit=1,
        filters={"session_id": "incident-1", "tags": {"contains": "runbook"}},
    )
    assert len(results) == 1
    assert results[0]["text"] == "scoped farther"


@pytest.mark.asyncio
async def test_repr(tmp_path: Path) -> None:
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri)
    r = repr(ctx)
    assert r.startswith("AsyncContext(")
    assert uri in r


@pytest.mark.asyncio
async def test_create_with_options(tmp_path: Path) -> None:
    """AsyncContext.create forwards kwargs to Context.create."""
    uri = str(tmp_path / "ctx.lance")
    ctx = await AsyncContext.create(uri, id_index_type="btree")

    await ctx.add("user", "indexed")
    results = await ctx.list()
    assert len(results) == 1
