from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

lance = pytest.importorskip("lance")
Image = pytest.importorskip("PIL.Image")

from lance_context.api import Context


def _read_rows(uri: str, version: int | None = None) -> list[dict[str, object]]:
    dataset = lance.dataset(uri, version=version) if version is not None else lance.dataset(uri)
    table = dataset.to_table()
    return table.to_pylist()


def _image_bytes(image: Image.Image, *, format: str | None = None) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=format or image.format or "PNG")
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
