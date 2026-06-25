"""Tests for curate + export to trainable datasets (export_training)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_export_sft_groups_by_session(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "hello", session_id="s1")
    ctx.add("assistant", "hi there", session_id="s1")

    out = tmp_path / "sft.jsonl"
    manifest = ctx.export_training(str(out), task="sft", group_by="session_id")

    rows = _read_jsonl(out)
    assert len(rows) == 1
    assert [m["content"] for m in rows[0]["messages"]] == ["hello", "hi there"]
    assert manifest["task"] == "sft"
    assert manifest["counts"]["examples"] == 1
    # sibling manifest file written
    assert (tmp_path / "sft.jsonl.manifest.json").exists()


def test_export_sft_rejection_sampling_min_reward(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("assistant", "good", session_id="a", metadata={"reward": 0.9})
    ctx.add("assistant", "bad", session_id="b", metadata={"reward": 0.1})

    out = tmp_path / "sft.jsonl"
    ctx.export_training(str(out), task="sft", min_reward=0.5)

    rows = _read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "good"


def test_export_preference_paired(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "q", session_id="s1")
    ctx.add("assistant", "great", session_id="s1", metadata={"reward": 0.9})
    ctx.add("assistant", "poor", session_id="s1", metadata={"reward": 0.1})

    out = tmp_path / "pref.jsonl"
    ctx.export_training(str(out), task="preference", preference_form="paired")

    rows = _read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["form"] == "paired"
    assert rows[0]["chosen"][0]["content"] == "great"
    assert rows[0]["rejected"][0]["content"] == "poor"


def test_export_preference_unpaired_kto(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "q", session_id="s1")
    ctx.add("assistant", "yes", session_id="s1", metadata={"label": "chosen"})
    ctx.add("assistant", "no", session_id="s1", metadata={"label": "rejected"})

    out = tmp_path / "pref.jsonl"
    ctx.export_training(str(out), task="preference", preference_form="unpaired")

    rows = _read_jsonl(out)
    assert len(rows) == 2
    labels = {r["completion"][0]["content"]: r["label"] for r in rows}
    assert labels == {"yes": True, "no": False}


def test_export_rollout_groups_with_rewards(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "solve", session_id="s1")
    ctx.add(
        "assistant",
        "a1",
        session_id="s1",
        metadata={"reward": 1.0, "reward_source": "verifier", "group_id": "g1"},
    )
    ctx.add(
        "assistant",
        "a2",
        session_id="s1",
        metadata={"reward": 0.0, "reward_source": "verifier", "group_id": "g1"},
    )

    out = tmp_path / "rollout.jsonl"
    ctx.export_training(str(out), task="rollout")

    rows = _read_jsonl(out)
    assert len(rows) == 1
    assert rows[0]["group_id"] == "g1"
    assert len(rows[0]["responses"]) == 2
    assert rows[0]["responses"][0]["reward_source"] == "verifier"


def test_export_dedup_collapses_duplicates(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"), embedding_dim=4)
    ctx.add("user", "one", session_id="a", embedding=[1.0, 0.0, 0.0, 0.0])
    ctx.add("user", "two", session_id="b", embedding=[1.0, 0.0, 0.0, 0.0])

    out = tmp_path / "sft.jsonl"
    manifest = ctx.export_training(str(out), task="sft", dedup_threshold=0.01)

    assert manifest["counts"]["after_dedup"] == 1
    assert len(_read_jsonl(out)) == 1


def test_export_invalid_task_raises(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "hi")
    with pytest.raises(RuntimeError, match="invalid export task"):
        ctx.export_training(str(tmp_path / "x.jsonl"), task="bogus")


def test_export_reproducible(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "a", session_id="s1")
    ctx.add("assistant", "b", session_id="s1")

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    ctx.export_training(str(first), task="sft")
    ctx.export_training(str(second), task="sft")
    assert first.read_text() == second.read_text()
