"""Tests for the rollout store Python bindings (local/embedded path).

The embedded `RolloutStore.open` path exercises the exact same binding surface
(JSON marshalling of records/DTOs, base64 blob handling, projection) that the
remote path uses, since both dispatch through the unified Rust `RolloutStore`
enum. Remote-specific wiring (HTTP) is covered by the Rust client tests.
"""

from __future__ import annotations

import tempfile

import pytest
from lance_context import RolloutStore


@pytest.fixture()
def store_uri():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_add_dict_list_get(store_uri):
    store = RolloutStore.open(store_uri)
    assert store.version() >= 0

    resp = store.add(
        {
            "id": "row-0",
            "rollout_id": "traj-1",
            "problem_id": "p-1",
            "role": "assistant",
            "content": "the answer is 42",
            "reward": 1.0,
            "policy_version": "ckpt-7",
        }
    )
    assert resp["count"] == 1
    assert resp["ids"] == ["row-0"]

    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["id"] == "row-0"
    assert rows[0]["rollout_id"] == "traj-1"
    assert rows[0]["reward"] == 1.0
    assert rows[0]["policy_version"] == "ckpt-7"

    fetched = store.get("row-0")
    assert fetched is not None
    assert fetched["id"] == "row-0"
    assert store.get("missing") is None


def test_add_many_and_add_one(store_uri):
    store = RolloutStore.open(store_uri)
    store.add(
        [
            {"id": f"row-{i}", "rollout_id": "traj-1", "reward": float(i)}
            for i in range(3)
        ]
    )
    store.add_one(id="row-3", rollout_id="traj-1", reward=3.0)

    rows = store.list()
    assert {r["id"] for r in rows} == {"row-0", "row-1", "row-2", "row-3"}


def test_binary_payload_roundtrip(store_uri):
    store = RolloutStore.open(store_uri)
    blob = b"\x00\x01\x02trace-bytes"
    store.add(
        {
            "id": "art-0",
            "rollout_id": "traj-1",
            "role": "artifact",
            "content_type": "application/octet-stream",
            "binary_payload": blob,
            "payload_size": len(blob),
        }
    )

    # list/get project the blob column out (cheap metadata scans).
    row = store.get("art-0")
    assert row is not None
    assert row.get("binary_payload") is None
    assert row["payload_size"] == len(blob)

    # get_blob materializes the bytes on demand.
    assert store.get_blob("art-0") == blob
    assert store.get_blob("missing") is None


def test_empty_add_rejected(store_uri):
    store = RolloutStore.open(store_uri)
    with pytest.raises(ValueError):
        store.add([])


def test_reopen_sees_prior_rows(store_uri):
    store = RolloutStore.open(store_uri)
    store.add({"id": "row-0", "rollout_id": "traj-1"})
    del store

    reopened = RolloutStore.open(store_uri)
    rows = reopened.list()
    assert len(rows) == 1
    assert rows[0]["id"] == "row-0"
