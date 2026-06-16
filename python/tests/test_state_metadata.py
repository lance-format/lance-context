"""Tests for writing structured state metadata from Python."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def test_add_roundtrips_state_metadata(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add(
        "assistant",
        "plan step complete",
        state_metadata={
            "step": 3,
            "active_plan_id": "plan-1",
            "tokens_used": 128,
            "custom": "retrieval",
        },
    )

    records = ctx.list()
    assert records[0]["state_metadata"] == {
        "step": 3,
        "active_plan_id": "plan-1",
        "tokens_used": 128,
        "custom": "retrieval",
    }


def test_add_many_roundtrips_partial_state_metadata(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    ctx.add_many(
        [
            {
                "role": "user",
                "content": "first",
                "state_metadata": {"step": 1, "tokens_used": 10},
            },
            {"role": "assistant", "content": "second"},
        ]
    )

    records = ctx.list()
    assert records[0]["state_metadata"] == {
        "step": 1,
        "active_plan_id": None,
        "tokens_used": 10,
        "custom": None,
    }
    assert records[1]["state_metadata"] is None
