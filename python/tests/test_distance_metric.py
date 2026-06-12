"""Tests for the configurable vector-search distance metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context

DIM = 1536


def _vec(x0: float, x1: float = 0.0) -> list[float]:
    vector = [0.0] * DIM
    vector[0] = x0
    vector[1] = x1
    return vector


# Query points along axis 0. ``aligned`` shares the query's direction (cosine
# distance 0, largest dot product) but is far away in L2; ``near`` is closest in
# L2 but off-axis, so cosine ranks it lower.
QUERY = _vec(1.0)
ALIGNED = _vec(10.0)
NEAR = _vec(1.0, 1.0)


def _make(uri: str, **kwargs: str) -> Context:
    ctx = Context.create(uri, **kwargs)
    ctx.add("user", "aligned", embedding=ALIGNED, external_id="aligned")
    ctx.add("user", "near", embedding=NEAR, external_id="near")
    return ctx


def test_default_metric_is_l2(tmp_path: Path) -> None:
    ctx = _make(str(tmp_path / "l2.lance"))
    hits = ctx.search(QUERY, limit=2)
    assert [h["external_id"] for h in hits][0] == "near"


def test_cosine_metric_changes_ranking(tmp_path: Path) -> None:
    ctx = _make(str(tmp_path / "cosine.lance"), distance_metric="cosine")
    hits = ctx.search(QUERY, limit=2)
    assert [h["external_id"] for h in hits][0] == "aligned"


def test_metric_persists_when_reopened_without_option(tmp_path: Path) -> None:
    # Create with cosine, then reopen WITHOUT specifying the metric: it must be
    # recovered from the dataset so ranking still uses cosine.
    uri = str(tmp_path / "persist.lance")
    _make(uri, distance_metric="cosine")
    reopened = Context(uri)
    hits = reopened.search(QUERY, limit=2)
    assert [h["external_id"] for h in hits][0] == "aligned"


def test_metric_mismatch_on_reopen_rejected(tmp_path: Path) -> None:
    uri = str(tmp_path / "mismatch.lance")
    _make(uri, distance_metric="cosine")
    with pytest.raises(RuntimeError, match="distance metric"):
        Context.create(uri, distance_metric="dot")


def test_dot_metric_ranks_by_inner_product(tmp_path: Path) -> None:
    ctx = _make(str(tmp_path / "dot.lance"), distance_metric="dot")
    hits = ctx.search(QUERY, limit=2)
    assert [h["external_id"] for h in hits][0] == "aligned"


def test_invalid_metric_rejected(tmp_path: Path) -> None:
    uri = str(tmp_path / "bad.lance")
    with pytest.raises(RuntimeError, match="invalid distance metric"):
        Context.create(uri, distance_metric="manhattan")
