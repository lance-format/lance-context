"""Remote (HTTP) rollout store tests.

Spins up a real `lance-context-server` subprocess and drives it through the
async `AsyncRolloutStore` wrapper — the path RL generation workers and the
learner use in a deployed setup. Skips gracefully if the server binary has not
been built locally; in CI (``CI`` env var set) a missing binary is a hard
failure rather than a silent skip.
"""

from __future__ import annotations

import asyncio
import time

from lance_context import AsyncRolloutStore


async def _eventually(fn, predicate, timeout: float = 15.0):
    """Poll `fn` until `predicate` holds, or fail after `timeout`.

    `add` is durable on return but not visible until the server's sweeper seals
    the memtable, so a read immediately after a write legitimately returns
    nothing. Polling asserts the row *arrives* without pinning the test to the
    flush interval.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = await fn()
        if predicate(last):
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"condition not met within {timeout}s; last value: {last!r}")


def test_remote_roundtrip(server_without_sweeps):
    async def run():
        store = await AsyncRolloutStore.connect_or_create(
            server_without_sweeps,
            "rl-run-1",
        )

        resp = await store.add(
            [
                {
                    "id": "row-0",
                    "rollout_id": "traj-1",
                    "problem_id": "p-1",
                    "role": "assistant",
                    "content": "the answer is 42",
                    "reward": 1.0,
                    "policy_version": "ckpt-7",
                },
                {
                    "id": "row-1",
                    "rollout_id": "traj-1",
                    "role": "artifact",
                    "content_type": "application/octet-stream",
                    "binary_payload": b"\x00\x01\x02trace",
                    "payload_size": 8,
                },
            ],
            flush=True,
        )
        assert resp["count"] == 2

        rows = await store.list()
        assert {r["id"] for r in rows} == {"row-0", "row-1"}

        one = await store.get("row-0")
        assert one is not None
        assert one["reward"] == 1.0
        # blob projected out of get, materialized on demand
        assert one.get("binary_payload") is None

        blob = await store.get_blob("row-1")
        assert blob == b"\x00\x01\x02trace"
        assert await store.get("missing") is None

    asyncio.run(run())


def test_remote_add_one_and_connect(server_without_sweeps):
    async def run():
        store = await AsyncRolloutStore.connect_or_create(
            server_without_sweeps,
            "rl-run-2",
        )
        await store.add_one(id="only", rollout_id="traj-9", reward=0.5, flush=True)

        # A separately connected client sees the flushed row immediately.
        reader = await AsyncRolloutStore.connect(server_without_sweeps, "rl-run-2")
        rows = await reader.list()
        assert [r["id"] for r in rows] == ["only"]

    asyncio.run(run())


def test_remote_filtered_list(server):
    async def run():
        store = await AsyncRolloutStore.connect_or_create(server, "rl-filtered")
        await store.add(
            [
                {
                    "id": "row-7",
                    "rollout_id": "traj-7",
                    "dataset": "dataset-7",
                    "role": "assistant",
                    "content_type": "application/json",
                    "policy_version": "ckpt-7",
                    "include_in_training": True,
                },
                {
                    "id": "row-8",
                    "rollout_id": "traj-8",
                    "dataset": "dataset-8",
                    "role": "assistant",
                    "content_type": "text/plain",
                    "policy_version": "ckpt-8",
                    "include_in_training": True,
                },
            ]
        )

        rows = await _eventually(
            lambda: store.list(
                filters={"policy_version": "ckpt-7", "include_in_training": True}
            ),
            lambda r: len(r) == 1,
        )
        assert [row["id"] for row in rows] == ["row-7"]

        rows = await store.list(filters={"dataset": "dataset-7"})
        assert [row["id"] for row in rows] == ["row-7"]

        rows = await store.list(filters={"content_type": "application/json"})
        assert [row["id"] for row in rows] == ["row-7"]

    asyncio.run(run())


def test_remote_get_trajectory_orders_rows(server):
    async def run():
        store = await AsyncRolloutStore.connect_or_create(server, "rl-trajectory")
        await store.add(
            [
                {"id": "row-a", "rollout_id": "target", "sequence_order": 2},
                {"id": "other", "rollout_id": "other", "sequence_order": 0},
                {"id": "row-b", "rollout_id": "target", "sequence_order": 0},
            ]
        )

        rows = await _eventually(
            lambda: store.get_trajectory("target"), lambda r: len(r) == 2
        )
        assert [row["id"] for row in rows] == ["row-b", "row-a"]

    asyncio.run(run())
