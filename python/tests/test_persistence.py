from __future__ import annotations

import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
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


def _embedding(pivot: float) -> list[float]:
    values = [0.0] * 1536
    values[0] = pivot
    return values


def _embedding_with_dim(dim: int, pivot: float) -> list[float]:
    values = [0.0] * dim
    values[0] = pivot
    return values


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
    # moto >= 5 dropped the positional service argument; a single mock_server
    # now serves every service. Older moto (<5) accepted an `s3` positional
    # arg which is silently ignored here.
    cmd = [
        sys.executable,
        "-m",
        "moto.server",
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

    rows = ctx.list()
    assert len(rows) == 1

    record = rows[0]
    assert record["role"] == "user"
    assert record["text"] == "hello world"
    assert record["binary"] is None
    assert record["content_type"] == "text/plain"


def test_metadata_and_filters_round_trip(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    ctx.add(
        "assistant",
        "The runbook owner is the platform team.",
        bot_id="support-bot",
        session_id="incident-1",
        metadata={
            "tenant": "example-org",
            "scope": "team",
            "source_uri": "docs://runbooks/service-a",
            "tags": ["runbook", "ownership"],
            "confidence": 0.92,
        },
    )
    ctx.add(
        "user",
        "What is the owner?",
        bot_id="support-bot",
        session_id="incident-2",
        metadata={"tenant": "example-org", "scope": "personal"},
    )

    scoped = ctx.list(
        filters={
            "bot_id": "support-bot",
            "session_id": "incident-1",
            "role": "assistant",
            "content_type": "text/plain",
            "scope": "team",
            "tags": {"contains": "runbook"},
        }
    )

    assert len(scoped) == 1
    assert scoped[0]["text"] == "The runbook owner is the platform team."
    assert scoped[0]["metadata"] == {
        "tenant": "example-org",
        "scope": "team",
        "source_uri": "docs://runbooks/service-a",
        "tags": ["runbook", "ownership"],
        "confidence": 0.92,
    }

    created_at = scoped[0]["created_at"]
    assert isinstance(created_at, datetime)
    timestamp_scoped = ctx.list(
        filters={
            "created_at": {
                "gte": created_at.isoformat(),
                "lte": created_at.isoformat(),
            }
        }
    )
    assert [record["id"] for record in timestamp_scoped] == [scoped[0]["id"]]


def test_relationships_round_trip_search_and_related(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    relationships = [
        {"target_id": "doc-1#chunk-1", "relation": "cites", "weight": 0.75},
        {"target_id": "service-a", "relation": "mentions"},
    ]
    ctx.add(
        "assistant",
        "The service runbook points at the rollout checklist.",
        embedding=_embedding(0.0),
        relationships=relationships,
    )
    ctx.add("user", "unrelated", embedding=_embedding(1.0))

    records = ctx.list()
    related_record = next(record for record in records if record["role"] == "assistant")
    assert related_record["relationships"] == [
        {"target_id": "doc-1#chunk-1", "relation": "cites", "weight": 0.75},
        {"target_id": "service-a", "relation": "mentions", "weight": None},
    ]

    default_hits = ctx.search(_embedding(0.0), limit=1)
    assert default_hits[0]["relationships"] == []

    hits = ctx.search(_embedding(0.0), limit=1, include_relationships=True)
    assert hits[0]["relationships"] == related_record["relationships"]

    related = ctx.related("doc-1#chunk-1", relation="cites")
    assert len(related) == 1
    assert related[0]["text"] == "The service runbook points at the rollout checklist."


def test_search_applies_filters_before_limit(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    near = [0.0] * 1536
    far = [0.0] * 1536
    far[0] = 10.0

    ctx.add(
        "assistant",
        "global nearest",
        embedding=near,
        bot_id="support-bot",
        session_id="other",
        metadata={"scope": "personal"},
    )
    ctx.add(
        "assistant",
        "scoped farther",
        embedding=far,
        bot_id="support-bot",
        session_id="incident-1",
        metadata={"scope": "team", "tags": ["runbook"]},
    )

    hits = ctx.search(
        near,
        limit=1,
        filters={"session_id": "incident-1", "tags": {"contains": "runbook"}},
    )

    assert len(hits) == 1
    assert hits[0]["text"] == "scoped farther"
    assert hits[0]["metadata"] == {"scope": "team", "tags": ["runbook"]}


def test_retrieve_fuses_text_vector_and_filters(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    near = [0.0] * 1536
    far = [0.0] * 1536
    far[0] = 1.0

    ctx.add(
        "assistant",
        "general rollout risk guidance",
        embedding=near,
        metadata={"scope": "team", "tags": ["runbook"]},
    )
    ctx.add(
        "assistant",
        "POLICY-123 blocks service-a rollouts",
        embedding=far,
        metadata={"scope": "team", "tags": ["policy"]},
    )
    ctx.add(
        "assistant",
        "POLICY-123 personal note for service-a",
        embedding=far,
        metadata={"scope": "personal", "tags": ["policy"]},
    )

    hits = ctx.retrieve(
        text="POLICY-123 service-a",
        vector=near,
        limit=2,
        filters={"scope": "team"},
    )

    assert [hit["text"] for hit in hits] == [
        "POLICY-123 blocks service-a rollouts",
        "general rollout risk guidance",
    ]
    assert hits[0]["matched_channels"] == ["vector", "text"]
    assert hits[0]["score"] > hits[1]["score"]
    assert hits[0]["vector_distance"] is not None
    assert hits[0]["text_score"] == 1.0
    assert hits[1]["matched_channels"] == ["vector"]


def test_retrieve_supports_text_only(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))

    ctx.add("assistant", "The rollout owner is service-a.")
    ctx.add("assistant", "The unrelated deployment note mentions service-b.")

    hits = ctx.retrieve(text="service-a rollout", limit=1)

    assert len(hits) == 1
    assert hits[0]["text"] == "The rollout owner is service-a."
    assert hits[0]["matched_channels"] == ["text"]
    assert hits[0]["vector_distance"] is None
    assert hits[0]["text_score"] == 1.0


def test_custom_embedding_dimension_round_trips(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri), embedding_dim=3)
    near = _embedding_with_dim(3, 0.0)
    far = _embedding_with_dim(3, 1.0)

    ctx.add("assistant", "small vector near", embedding=near)
    ctx.add("assistant", "small vector far", embedding=far)

    hits = ctx.search(far, limit=1)
    assert hits[0]["text"] == "small vector far"

    reopened = Context.create(str(uri))
    hits = reopened.search(far, limit=1)
    assert hits[0]["text"] == "small vector far"

    with pytest.raises(RuntimeError, match="embedding dimension 3"):
        reopened.search(_embedding(1.0), limit=1)


def test_lifecycle_fields_round_trip_and_default_filtering(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    now = datetime.now(timezone.utc)

    ctx.add(
        "user",
        "active memory",
        expires_at=now + timedelta(days=1),
        retention_policy="long-term",
    )
    ctx.add("assistant", "expired trace", expires_at=now - timedelta(days=1))
    ctx.add(
        "system",
        "superseded fact",
        lifecycle_status="superseded",
        retired_at=now,
        retired_reason="replaced by newer fact",
        superseded_by_id="active-id",
    )
    ctx.add(
        "system",
        "failed approach",
        lifecycle_status="contradicted",
        retired_reason="negative knowledge",
    )

    visible = ctx.list()
    assert [record["text"] for record in visible] == [
        "active memory",
        "failed approach",
    ]
    assert visible[0]["retention_policy"] == "long-term"
    assert visible[0]["lifecycle_status"] == "active"

    all_records = ctx.list(include_expired=True, include_retired=True)
    assert [record["text"] for record in all_records] == [
        "active memory",
        "expired trace",
        "superseded fact",
        "failed approach",
    ]

    expired = all_records[1]
    assert expired["expires_at"] is not None
    assert expired["expires_at"] < datetime.now(timezone.utc)

    superseded = all_records[2]
    assert superseded["lifecycle_status"] == "superseded"
    assert superseded["retired_reason"] == "replaced by newer fact"
    assert superseded["superseded_by_id"] == "active-id"


def test_search_applies_lifecycle_filter_before_limit(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    now = datetime.now(timezone.utc)

    ctx.add("user", "active memory", embedding=_embedding(1.0))
    ctx.add(
        "assistant",
        "expired but closer",
        embedding=_embedding(0.0),
        expires_at=now - timedelta(minutes=1),
    )

    hits = ctx.search(_embedding(0.0), limit=1)
    assert [hit["text"] for hit in hits] == ["active memory"]

    hits_with_expired = ctx.search(_embedding(0.0), limit=1, include_expired=True)
    assert [hit["text"] for hit in hits_with_expired] == ["expired but closer"]


def test_supersedes_pointer_hides_old_record_by_default(tmp_path: Path) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))

    ctx.add("user", "old value", embedding=_embedding(0.0))
    old = ctx.list()[0]

    ctx.add(
        "user",
        "new value",
        embedding=_embedding(1.0),
        supersedes_id=old["id"],
    )

    assert [record["text"] for record in ctx.list()] == ["new value"]
    assert [record["text"] for record in ctx.search(_embedding(0.0), limit=10)] == [
        "new value"
    ]

    history = ctx.list(include_retired=True)
    assert [record["text"] for record in history] == ["old value", "new value"]
    assert history[1]["supersedes_id"] == old["id"]


def test_upsert_by_external_id_inserts_then_replaces_visible_record(
    tmp_path: Path,
) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    external_id = "doc-123#chunk-1"

    inserted = ctx.upsert(
        "user",
        "old value",
        embedding=_embedding(0.0),
        external_id=external_id,
    )
    assert inserted["inserted"] is True
    assert inserted["replaced_id"] is None
    old_id = inserted["record"]["id"]

    replaced = ctx.upsert(
        "user",
        "new value",
        embedding=_embedding(1.0),
        external_id=external_id,
        metadata={"revision": 2},
    )
    assert replaced["inserted"] is False
    assert replaced["replaced_id"] == old_id
    assert replaced["record"]["external_id"] == external_id
    assert replaced["record"]["supersedes_id"] == old_id

    assert ctx.get(external_id=external_id)["text"] == "new value"  # type: ignore[index]
    assert [record["text"] for record in ctx.list()] == ["new value"]
    assert [record["text"] for record in ctx.search(_embedding(0.0), limit=10)] == [
        "new value"
    ]

    history = ctx.list(include_retired=True)
    assert [record["text"] for record in history] == ["old value", "new value"]


def test_update_by_external_id_patches_mutable_fields_and_preserves_payload(
    tmp_path: Path,
) -> None:
    uri = tmp_path / "context.lance"
    ctx = Context.create(str(uri))
    external_id = "doc-123#chunk-1"

    ctx.add(
        "user",
        "stable content",
        embedding=_embedding(0.0),
        external_id=external_id,
        metadata={"revision": 1},
    )
    original = ctx.get(external_id=external_id)
    assert original is not None

    updated = ctx.update(
        external_id=external_id,
        metadata={"revision": 2, "confidence": 0.9},
        relationships=[{"target_id": "doc-123", "relation": "derived_from"}],
    )

    assert updated["updated"] is True
    assert updated["replaced_id"] == original["id"]
    assert updated["record"]["id"] != original["id"]
    assert updated["record"]["external_id"] == external_id
    assert updated["record"]["text"] == "stable content"
    assert updated["record"]["metadata"] == {"revision": 2, "confidence": 0.9}
    assert updated["record"]["relationships"] == [
        {"target_id": "doc-123", "relation": "derived_from", "weight": None}
    ]
    assert updated["record"]["supersedes_id"] == original["id"]

    visible = ctx.get(external_id=external_id)
    assert visible is not None
    assert visible["id"] == updated["record"]["id"]
    assert [record["text"] for record in ctx.list()] == ["stable content"]

    history = ctx.list(include_retired=True)
    assert {record["id"] for record in history} == {
        original["id"],
        updated["record"]["id"],
    }


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
    if version_second == version_first:
        pytest.xfail("MemWAL-backed writes do not advance base-table manifest versions")

    ctx.checkout(version_first)

    rows_versioned = _read_rows(str(uri), version=ctx.version())
    assert len(rows_versioned) == 1
    assert rows_versioned[0]["text_payload"] == "first-entry"

    latest_rows = _read_rows(str(uri))
    assert [row["text_payload"] for row in latest_rows] == [
        "first-entry",
        "second-entry",
    ]


_S3_MEMWAL_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "Upstream: lance 7.0.0 mem_wal_writer builds its object store with "
        "ObjectStore::from_uri(base_uri), dropping the dataset's "
        "storage_options (lance/src/dataset/mem_wal/api.rs:607). The WAL "
        "writer therefore ignores aws_endpoint_url and targets real AWS, so "
        "`add` fails with 'bucket not found' against any custom-endpoint S3 "
        "(moto, MinIO, Ceph, R2). `create` works because it goes through the "
        "options-aware path. Needs from_uri_and_params upstream; there is no "
        "local workaround since ShardWriterConfig cannot carry the options. "
        "Remove this marker once lance is bumped past the fix."
    ),
)


@_S3_MEMWAL_XFAIL
def test_s3_round_trip_with_storage_options(moto_endpoint: str, s3_client) -> None:
    """Canonical path: generic storage_options dict (aligns with lance/lance-graph)."""
    bucket = f"context-{uuid.uuid4().hex}"
    s3_client.create_bucket(Bucket=bucket)
    key = f"contexts/{uuid.uuid4().hex}/context.lance"
    uri = f"s3://{bucket}/{key}"

    ctx = Context.create(uri, storage_options=_s3_storage_options(moto_endpoint))

    ctx.add("user", "remote-hello")
    ctx.add("assistant", "remote-response")
    ctx.checkout(ctx.version())

    rows = _read_rows(uri, storage_options=_s3_storage_options(moto_endpoint))
    assert [row["text_payload"] for row in rows] == ["remote-hello", "remote-response"]
    assert ctx.entries() == 2


@_S3_MEMWAL_XFAIL
def test_s3_deprecated_aws_kwargs_still_work(moto_endpoint: str, s3_client) -> None:
    """AWS kwargs keep working (back-compat) and emit a DeprecationWarning."""
    bucket = f"context-{uuid.uuid4().hex}"
    s3_client.create_bucket(Bucket=bucket)
    key = f"contexts/{uuid.uuid4().hex}/context.lance"
    uri = f"s3://{bucket}/{key}"

    with pytest.warns(DeprecationWarning, match="storage_options"):
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

    rows = _read_rows(uri, storage_options=_s3_storage_options(moto_endpoint))
    assert [row["text_payload"] for row in rows] == ["remote-hello", "remote-response"]
    assert ctx.entries() == 2
