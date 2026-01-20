from __future__ import annotations

import socket
import subprocess
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lance_context.api import Context  # noqa: E402

lance = pytest.importorskip("lance")

_S3_ACCESS_KEY = "test"
_S3_SECRET_KEY = "test"
_S3_REGION = "us-east-1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _s3_storage_options(endpoint: str) -> dict[str, str]:
    return {
        "aws_access_key_id": _S3_ACCESS_KEY,
        "aws_secret_access_key": _S3_SECRET_KEY,
        "aws_region": _S3_REGION,
        "aws_endpoint_url": endpoint,
        "aws_allow_http": "true",
    }


def _wait_for_moto_ready(client: Any, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client.list_buckets()
            return
        except Exception as exc:  # pragma: no cover - best effort
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError("moto server did not become ready") from last_error


@pytest.fixture(scope="module")
def moto_endpoint() -> str:
    pytest.importorskip("moto.server")
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config  # type: ignore[import-not-found]

    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "moto.server",
        "s3",
        "-H",
        "127.0.0.1",
        "-p",
        str(port),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    endpoint = f"http://127.0.0.1:{port}"

    session = boto3.session.Session(
        aws_access_key_id=_S3_ACCESS_KEY,
        aws_secret_access_key=_S3_SECRET_KEY,
        region_name=_S3_REGION,
    )
    client = session.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )

    try:
        _wait_for_moto_ready(client)
        yield endpoint
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture
def s3_client(moto_endpoint: str):
    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config  # type: ignore[import-not-found]

    session = boto3.session.Session(
        aws_access_key_id=_S3_ACCESS_KEY,
        aws_secret_access_key=_S3_SECRET_KEY,
        region_name=_S3_REGION,
    )
    return session.client(
        "s3",
        endpoint_url=moto_endpoint,
        config=Config(signature_version="s3v4"),
    )


def _read_rows(
    uri: str,
    version: int | None = None,
    storage_options: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    kwargs: dict[str, Any] = {}
    if version is not None:
        kwargs["version"] = version
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    dataset = lance.dataset(uri, **kwargs)
    table = dataset.to_table()
    return table.to_pylist()


def _image_bytes(image: Any, *, format: str | None = None) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=format or getattr(image, "format", None) or "PNG")
    return buffer.getvalue()


def test_text_round_trip(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    ctx.add("user", "hello world")

    rows = _read_rows(str(uri))
    assert len(rows) == 1

    record = rows[0]
    assert record["role"] == "user"
    assert record["text_payload"] == "hello world"
    assert record["binary_payload"] is None
    assert record["content_type"] == "text/plain"


def test_image_round_trip(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))

    image = Image.new("RGB", (4, 4), color="magenta")
    ctx.add("assistant", image)

    rows = _read_rows(str(uri))
    assert len(rows) == 1

    record = rows[0]
    assert record["role"] == "assistant"
    assert record["text_payload"] is None
    assert record["content_type"] == "image/png"
    assert record["binary_payload"] == _image_bytes(image)


def test_time_travel_checkout(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))

    ctx.add("system", "first-entry")
    version_first = ctx.version()

    ctx.add("system", "second-entry")
    version_second = ctx.version()
    assert version_second >= version_first

    ctx.checkout(version_first)

    rows_versioned = _read_rows(str(uri), version=ctx.version())
    assert len(rows_versioned) == 1
    assert rows_versioned[0]["text_payload"] == "first-entry"

    latest_rows = _read_rows(str(uri))
    assert [row["text_payload"] for row in latest_rows] == [
        "first-entry",
        "second-entry",
    ]


def test_s3_round_trip_remote_store(moto_endpoint: str, s3_client) -> None:
    bucket = f"context-{uuid.uuid4().hex}"
    s3_client.create_bucket(Bucket=bucket)
    key = f"contexts/{uuid.uuid4().hex}/context.lance"
    uri = f"s3://{bucket}/{key}"

    ctx = Context.create(
        uri,
        aws_access_key_id=_S3_ACCESS_KEY,
        aws_secret_access_key=_S3_SECRET_KEY,
        region=_S3_REGION,
        endpoint_url=moto_endpoint,
        allow_http=True,
    )

    ctx.add("user", "remote-hello")
    ctx.add("assistant", "remote-response")
    ctx.checkout(ctx.version())

    rows = _read_rows(uri, storage_options=_s3_storage_options(moto_endpoint))
    assert [row["text_payload"] for row in rows] == ["remote-hello", "remote-response"]
    assert ctx.entries() == 2
