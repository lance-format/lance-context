"""Tests for the retrieval-quality evaluation harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context

DIM = 8


def _emb(*lead: float) -> list[float]:
    vec = [0.0] * DIM
    for i, value in enumerate(lead):
        vec[i] = value
    return vec


def _make_context(tmp_path: Path) -> Context:
    ctx = Context.create(str(tmp_path / "context.lance"), embedding_dim=DIM)
    ctx.add_many(
        [
            {
                "role": "user",
                "content": "alpha",
                "external_id": "doc-a",
                "embedding": _emb(1.0),
            },
            {
                "role": "user",
                "content": "beta",
                "external_id": "doc-b",
                "embedding": _emb(0.5),
            },
            {
                "role": "user",
                "content": "gamma",
                "external_id": "doc-c",
                "embedding": _emb(0.0, 1.0),
            },
        ]
    )
    return ctx


def test_evaluate_vector_mode(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    report = ctx.evaluate(
        [
            {
                "query_id": "q1",
                "vector": _emb(1.0),
                "relevant": [{"external_id": "doc-a"}],
            }
        ],
        k=2,
        mode="vector",
    )

    assert report["mode"] == "vector"
    assert report["k"] == 2
    assert report["num_queries"] == 1
    assert report["per_query"][0]["retrieved"][0] == "doc-a"
    agg = report["aggregate"]
    assert agg["recall"] == pytest.approx(1.0)
    assert agg["precision"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(1.0)
    assert agg["hit_rate"] == pytest.approx(1.0)


def test_evaluate_graded_ndcg(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    report = ctx.evaluate(
        [
            {
                "query_id": "q1",
                "vector": _emb(1.0),
                "relevant": [
                    {"external_id": "doc-a", "grade": 1.0},
                    {"external_id": "doc-b", "grade": 3.0},
                ],
            }
        ],
        k=2,
        mode="vector",
    )
    # retrieved order is [doc-a(grade1), doc-b(grade3)]; ideal is [b, a].
    assert 0.0 < report["aggregate"]["ndcg"] < 1.0


def test_evaluate_config_ab_delta(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    queries = [
        {
            "query_id": "q1",
            "vector": _emb(1.0),
            "relevant": [{"external_id": "doc-b"}],  # ranks 2nd
        }
    ]
    at_1 = ctx.evaluate(queries, k=1)
    at_2 = ctx.evaluate(queries, k=2)
    assert at_1["aggregate"]["recall"] == pytest.approx(0.0)
    assert at_2["aggregate"]["recall"] == pytest.approx(1.0)


def test_evaluate_versions_same_version_zero_delta(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    version = ctx.version()
    ab = ctx.evaluate_versions(
        [
            {
                "query_id": "q1",
                "vector": _emb(1.0),
                "relevant": [{"external_id": "doc-a"}],
            }
        ],
        version,
        version,
        k=1,
    )
    assert ab["deltas"]["recall"] == pytest.approx(0.0)
    assert ab["baseline"]["version"] == version
    assert ctx.version() == version


def test_evaluate_invalid_mode_raises(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path)
    with pytest.raises(RuntimeError, match="invalid eval mode"):
        ctx.evaluate(
            [{"query_id": "q1", "vector": _emb(1.0), "relevant": []}],
            mode="bogus",
        )
