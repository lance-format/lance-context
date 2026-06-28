"""End-to-end tests for external media references (issue #115).

Large media lives as an object in the configured object store and is referenced
from a record by a typed ``payload_uri`` (plus optional ``payload_size`` /
``payload_checksum``). ``get``/``list`` return the reference without fetching
bytes; ``fetch_payload`` resolves them on demand via the context's storage path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lance_context.api import Context


def test_external_payload_reference_roundtrips(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    # Offload the media bytes to an object, then reference it from a record.
    object_uri = str(tmp_path / "media-001.png")
    payload = b"\x89PNG\r\n\x1a\n external media bytes"
    written = ctx.put_payload(object_uri, payload)
    assert written == len(payload)

    ctx.add(
        "user",
        "diagram of the incident timeline",  # inline caption, not the media
        content_type="image/png",
        external_id="img-1",
        payload_uri=object_uri,
        payload_size=len(payload),
        payload_checksum="sha256:abc",
    )

    # The reference round-trips and no bytes are inlined.
    record = ctx.get(external_id="img-1")
    assert record is not None
    assert record["payload_uri"] == object_uri
    assert record["payload_size"] == len(payload)
    assert record["payload_checksum"] == "sha256:abc"
    assert record["binary"] is None

    # list returns the reference without materializing the bytes.
    listed = ctx.list()
    assert len(listed) == 1
    assert listed[0]["payload_uri"] == object_uri
    assert listed[0]["binary"] is None

    # Opt-in fetch resolves the bytes through the context's storage path.
    assert ctx.fetch_payload(record["id"]) == payload


def test_fetch_payload_missing_record_and_missing_reference(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    # Unknown id resolves to None rather than raising.
    assert ctx.fetch_payload("does-not-exist") is None

    # A record without an external reference is an error to fetch.
    ctx.add("user", "inline text only", external_id="inline-1")
    record = ctx.get(external_id="inline-1")
    assert record is not None
    with pytest.raises(Exception, match="no external payload reference"):
        ctx.fetch_payload(record["id"])


def test_update_attaches_payload_reference_later(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    object_uri = str(tmp_path / "audio.wav")
    payload = b"RIFF....WAVEfmt external audio"
    ctx.put_payload(object_uri, payload)

    # Capture the record first, attach the media reference afterwards.
    ctx.add("user", "voice note", external_id="note-1")
    result = ctx.update(
        external_id="note-1",
        payload_uri=object_uri,
        payload_size=len(payload),
    )
    assert result["updated"] is True
    assert result["record"]["payload_uri"] == object_uri
    assert ctx.fetch_payload(result["record"]["id"]) == payload


def test_upsert_many_forwards_payload_reference(tmp_path: Path) -> None:
    uri = str(tmp_path / "context.lance")
    ctx = Context.create(uri)

    object_uri = str(tmp_path / "frame-001.jpg")
    payload = b"\xff\xd8\xff external jpeg bytes"
    ctx.put_payload(object_uri, payload)

    # Bulk insert-or-replace must carry the external reference through, just like
    # add_many / single upsert.
    ctx.upsert_many(
        [
            {
                "role": "assistant",
                "content": "captured frame",
                "content_type": "image/jpeg",
                "external_id": "frame-001",
                "payload_uri": object_uri,
                "payload_size": len(payload),
                "payload_checksum": "sha256:frame",
            }
        ]
    )

    record = ctx.get(external_id="frame-001")
    assert record is not None
    assert record["payload_uri"] == object_uri
    assert record["payload_size"] == len(payload)
    assert record["payload_checksum"] == "sha256:frame"
    assert record["binary"] is None
    assert ctx.fetch_payload(record["id"]) == payload
