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


def test_export_train_eval_split(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    for s in range(20):
        ctx.add("user", f"q{s}", session_id=f"s{s}")
        ctx.add("assistant", f"a{s}", session_id=f"s{s}")

    base = tmp_path / "cut.jsonl"
    manifest = ctx.export_training(
        str(base),
        task="sft",
        group_by="session_id",
        split={"eval_fraction": 0.3, "by": "session_id", "seed": 42},
    )

    train = tmp_path / "cut.train.jsonl"
    eval_ = tmp_path / "cut.eval.jsonl"
    assert train.exists() and eval_.exists()
    assert manifest["split"]["side"] == "train"
    assert manifest["split"]["seed"] == 42

    def sessions(path: Path) -> set[str]:
        return {
            json.loads(line)["provenance"]["session_id"]
            for line in path.read_text().splitlines()
            if line.strip()
        }

    train_sessions = sessions(train)
    eval_sessions = sessions(eval_)
    assert train_sessions and eval_sessions
    assert train_sessions.isdisjoint(eval_sessions)  # group-disjoint
    assert len(train_sessions) + len(eval_sessions) == 20


def test_export_split_is_deterministic(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    for s in range(10):
        ctx.add("user", f"q{s}", session_id=f"s{s}")

    split = {"eval_fraction": 0.5, "by": "session_id", "seed": 7}
    ctx.export_training(str(tmp_path / "a.jsonl"), task="sft", split=split)
    ctx.export_training(str(tmp_path / "b.jsonl"), task="sft", split=split)

    assert (tmp_path / "a.eval.jsonl").read_text() == (
        tmp_path / "b.eval.jsonl"
    ).read_text()
    assert (tmp_path / "a.train.jsonl").read_text() == (
        tmp_path / "b.train.jsonl"
    ).read_text()


def test_export_emits_stats_report(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "hello there friend", session_id="s1", source="memory")
    ctx.add("assistant", "hi", session_id="s1", source="memory")

    out = tmp_path / "sft.jsonl"
    ctx.export_training(str(out), task="sft", emit_stats=True)

    stats_path = tmp_path / "sft.jsonl.stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["examples"] == 1
    assert stats["by_role"]["user"] == 1
    assert stats["by_role"]["assistant"] == 1
    assert stats["by_source"]["memory"] == 2
    assert stats["tokens"]["source"] == "length_proxy"
    assert stats["tokens"]["max"] == 3.0  # "hello there friend"


def test_export_no_stats_without_flag(tmp_path: Path) -> None:
    ctx = Context.create(str(tmp_path / "ctx.lance"))
    ctx.add("user", "hi", session_id="s1")
    out = tmp_path / "sft.jsonl"
    ctx.export_training(str(out), task="sft")
    assert not (tmp_path / "sft.jsonl.stats.json").exists()
