"""Remote (HTTP) rollout store tests.

Spins up a real `lance-context-server` subprocess and drives it through the
async `AsyncRolloutStore` wrapper — the path RL generation workers and the
learner use in a deployed setup. Skips gracefully if the server binary has not
been built (e.g. a pure-Python CI job).
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from lance_context import AsyncRolloutStore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVER_BIN = _REPO_ROOT / "target" / "debug" / "lance-context-server"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    url = f"{base_url}/api/v1/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.2)
    raise RuntimeError(f"server did not become healthy at {url}")


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


@pytest.fixture()
def server():
    if not _SERVER_BIN.exists():
        pytest.skip(f"server binary not built at {_SERVER_BIN}")
    port = _free_port()
    with tempfile.TemporaryDirectory() as data_dir:
        # Rows are durable on `add` but only become visible when the server's
        # sweeper seals the memtable. The 30s production default would make
        # every write-then-assert below hang; 1s keeps the tests honest about
        # the async-visibility contract without waiting on it.
        env = {**os.environ, "ROLLOUT_FLUSH_INTERVAL_SECS": "1"}
        proc = subprocess.Popen(
            [
                str(_SERVER_BIN),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--data-dir",
                data_dir,
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            _wait_for_health(base_url)
            yield base_url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_remote_roundtrip(server):
    async def run():
        store = await AsyncRolloutStore.connect_or_create(server, "rl-run-1")

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
            ]
        )
        assert resp["count"] == 2

        rows = await _eventually(store.list, lambda r: len(r) == 2)
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


def test_remote_add_one_and_connect(server):
    async def run():
        store = await AsyncRolloutStore.connect_or_create(server, "rl-run-2")
        await store.add_one(id="only", rollout_id="traj-9", reward=0.5)

        # A second connection sees the first's flushed write (durable, no
        # read affinity).
        reader = await AsyncRolloutStore.connect(server, "rl-run-2")
        rows = await _eventually(reader.list, lambda r: len(r) == 1)
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
                    "role": "assistant",
                    "policy_version": "ckpt-7",
                    "include_in_training": True,
                },
                {
                    "id": "row-8",
                    "rollout_id": "traj-8",
                    "role": "assistant",
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
