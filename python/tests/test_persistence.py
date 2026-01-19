from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

lance = pytest.importorskip("lance")

from lance_context.api import Context


def _read_rows(uri: str, version: int | None = None) -> list[dict[str, object]]:
    dataset = lance.dataset(uri, version=version) if version is not None else lance.dataset(uri)
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
    assert [row["text_payload"] for row in latest_rows] == ["first-entry", "second-entry"]
