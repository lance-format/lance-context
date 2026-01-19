from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "python" / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

lance = pytest.importorskip("lance")

from lance_context.api import Context


def _read_rows(uri: str, version: int | None = None) -> list[dict[str, object]]:
    dataset = lance.dataset(uri, version=version) if version is not None else lance.dataset(uri)
    table = dataset.to_table()
    return table.to_pylist()


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
