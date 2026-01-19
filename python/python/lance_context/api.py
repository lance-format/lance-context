from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any

from ._internal import Context as _Context  # pyright: ignore[reportMissingImports]
from ._internal import version as _version  # pyright: ignore[reportMissingImports]

__all__ = ["Context", "__version__"]

__version__ = _version()

_ARROW_STREAM_MIME = "application/vnd.apache.arrow.stream"


def _is_module(value: Any, prefix: str) -> bool:
    return type(value).__module__.startswith(prefix)


def _get_pyarrow():
    try:
        import pyarrow as pa  # pyright: ignore[reportMissingImports,reportMissingTypeStubs]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pyarrow is required to serialize pandas/polars dataframes"
        ) from exc
    return pa


def _coerce_arrow_table(value: Any):
    pa = _get_pyarrow()
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if _is_module(value, "polars."):
        table = value.to_arrow()
    elif _is_module(value, "pandas."):
        table = pa.Table.from_pandas(value)
    elif hasattr(value, "to_arrow"):
        table = value.to_arrow()
    else:
        return None

    if isinstance(table, pa.RecordBatch):
        return pa.Table.from_batches([table])
    if not isinstance(table, pa.Table):
        raise TypeError("to_arrow() did not return a pyarrow Table or RecordBatch")
    return table


def _serialize_dataframe(value: Any):
    table = _coerce_arrow_table(value)
    if table is None:
        return None
    pa = _get_pyarrow()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes(), _ARROW_STREAM_MIME


def _serialize_image(value: Any):
    if not _is_module(value, "PIL."):
        return None
    try:
        from PIL import Image  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Pillow is required to serialize images") from exc
    if not isinstance(value, Image.Image):
        return None

    image_format = value.format or "PNG"
    mime = None
    if hasattr(value, "get_format_mimetype"):
        mime = value.get_format_mimetype()
    if not mime:
        mime = Image.MIME.get(image_format.upper())
    if not mime:
        mime = "application/octet-stream"

    buffer = BytesIO()
    value.save(buffer, format=image_format)
    return buffer.getvalue(), mime


def _normalize_content(value: Any, content_type: str | None):
    serialized = _serialize_dataframe(value)
    if serialized is not None:
        payload, inferred = serialized
        return payload, content_type or inferred
    serialized = _serialize_image(value)
    if serialized is not None:
        payload, inferred = serialized
        return payload, content_type or inferred
    return value, content_type


def _coerce_vector(query: Any) -> list[float]:
    if hasattr(query, "tolist"):
        query = query.tolist()
    elif hasattr(query, "__array__"):
        query = query.__array__().tolist()
    if isinstance(query, (list, tuple)):
        return [float(item) for item in query]
    raise TypeError("search query must be a sequence of floats")


def _normalize_search_hit(raw: dict[str, Any]) -> dict[str, Any]:
    created_at = raw.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return {
        "id": raw.get("id"),
        "run_id": raw.get("run_id"),
        "role": raw.get("role"),
        "content_type": raw.get("content_type"),
        "text": raw.get("text_payload"),
        "binary": raw.get("binary_payload"),
        "embedding": raw.get("embedding"),
        "distance": raw.get("distance"),
        "created_at": created_at,
        "state_metadata": raw.get("state_metadata"),
    }


class Context:
    def __init__(self, uri: str) -> None:
        self._inner = _Context.create(uri)

    @classmethod
    def create(cls, uri: str) -> Context:
        return cls(uri)

    def uri(self) -> str:
        return self._inner.uri()

    def branch(self) -> str:
        return self._inner.branch()

    def entries(self) -> int:
        return self._inner.entries()

    def version(self) -> int:
        return self._inner.version()

    def add(
        self,
        role: str,
        content: Any,
        content_type: str | None = None,
        data_type: str | None = None,
    ) -> None:
        if content_type is not None and data_type is not None:
            raise ValueError("Specify only one of content_type or data_type")
        if content_type is None:
            content_type = data_type
        payload, resolved_type = _normalize_content(content, content_type)
        self._inner.add(role, payload, resolved_type)

    def snapshot(self, label: str | None = None) -> str:
        return self._inner.snapshot(label)

    def fork(self, branch_name: str) -> Context:
        inner = self._inner.fork(branch_name)
        return self._from_inner(inner)

    def checkout(self, version_id: int | str) -> None:
        self._inner.checkout(int(version_id))

    def search(self, query: Any, limit: int | None = None) -> list[dict[str, Any]]:
        vector = _coerce_vector(query)
        results = self._inner.search(vector, limit)
        return [_normalize_search_hit(item) for item in results]

    def __repr__(self) -> str:
        return (
            f"Context(uri={self._inner.uri()!r}, "
            f"branch={self._inner.branch()!r}, "
            f"entries={self._inner.entries()})"
        )

    @classmethod
    def _from_inner(cls, inner: _Context) -> Context:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj
