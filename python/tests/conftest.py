from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_SERVER_BIN = Path(__file__).resolve().parents[2] / "target/debug/lance-context-server"


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


@pytest.fixture()
def server() -> Iterator[str]:
    """Run a real HTTP server with an isolated dataset directory."""
    if not _SERVER_BIN.exists():
        # Locally a missing binary is a fair reason to skip. In CI it is not:
        # these suites cover the remote/HTTP client path, so a silent skip would
        # hide a regression. Fail loudly instead.
        msg = f"server binary not built at {_SERVER_BIN}"
        if os.environ.get("CI"):
            pytest.fail(f"{msg} (run `cargo build -p lance-context-server`)")
        pytest.skip(msg)
    port = _free_port()
    with tempfile.TemporaryDirectory() as data_dir:
        # Rows are durable on `add` but only become visible when the server's
        # sweeper seals the memtable. The 30s production default would make
        # every write-then-assert below hang; 1s keeps the tests honest about
        # the async-visibility contract without waiting on it.
        env = {**os.environ, "ROLLOUT_FLUSH_INTERVAL_SECS": "1"}
        with tempfile.TemporaryFile() as log:
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
                stdout=log,
                stderr=log,
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
                    proc.wait(timeout=10)
                log.seek(0)
                print(log.read().decode(errors="replace"))
                if os.name == "posix":
                    # Popen.terminate sends SIGTERM on POSIX. A clean exit
                    # proves the server reached its graceful-shutdown path.
                    assert proc.returncode == 0, (
                        "server did not handle SIGTERM gracefully "
                        f"(exit code {proc.returncode})"
                    )
