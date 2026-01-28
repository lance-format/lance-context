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


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw record dict from the Rust layer."""
    created_at = raw.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return {
        "id": raw.get("id"),
        "run_id": raw.get("run_id"),
        "bot_id": raw.get("bot_id"),
        "session_id": raw.get("session_id"),
        "role": raw.get("role"),
        "content_type": raw.get("content_type"),
        "text": raw.get("text_payload"),
        "binary": raw.get("binary_payload"),
        "embedding": raw.get("embedding"),
        "created_at": created_at,
        "state_metadata": raw.get("state_metadata"),
    }


def _normalize_search_hit(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a search hit - adds distance to the base record."""
    result = _normalize_record(raw)
    result["distance"] = raw.get("distance")
    return result


class Context:
    def __init__(
        self,
        uri: str,
        *,
        storage_options: dict[str, Any] | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        allow_http: bool = False,
        # Compaction configuration
        enable_background_compaction: bool = False,
        compaction_interval_secs: int = 300,
        compaction_min_fragments: int = 5,
        compaction_target_rows: int = 1_000_000,
        quiet_hours: list[tuple[int, int]] | None = None,
    ) -> None:
        options = dict(storage_options or {})
        if aws_access_key_id is not None:
            options["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key is not None:
            options["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token is not None:
            options["aws_session_token"] = aws_session_token
        if region is not None:
            options["aws_region"] = region
        if endpoint_url is not None:
            options["aws_endpoint_url"] = endpoint_url
        if allow_http:
            options["aws_allow_http"] = True

        # Build compaction config
        compaction_config = {
            "enabled": enable_background_compaction,
            "check_interval_secs": compaction_interval_secs,
            "min_fragments": compaction_min_fragments,
            "target_rows_per_fragment": compaction_target_rows,
            "quiet_hours": quiet_hours or [],
        }

        if options or compaction_config["enabled"]:
            self._inner = _Context.create(
                uri,
                storage_options=options or None,
                compaction_config=compaction_config,
            )
        else:
            self._inner = _Context.create(uri)

    @classmethod
    def create(
        cls,
        uri: str,
        *,
        storage_options: dict[str, Any] | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        allow_http: bool = False,
        # Compaction configuration
        enable_background_compaction: bool = False,
        compaction_interval_secs: int = 300,
        compaction_min_fragments: int = 5,
        compaction_target_rows: int = 1_000_000,
        quiet_hours: list[tuple[int, int]] | None = None,
    ) -> Context:
        return cls(
            uri,
            storage_options=storage_options,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region=region,
            endpoint_url=endpoint_url,
            allow_http=allow_http,
            enable_background_compaction=enable_background_compaction,
            compaction_interval_secs=compaction_interval_secs,
            compaction_min_fragments=compaction_min_fragments,
            compaction_target_rows=compaction_target_rows,
            quiet_hours=quiet_hours,
        )

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
        embedding: list[float] | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if content_type is not None and data_type is not None:
            raise ValueError("Specify only one of content_type or data_type")
        if content_type is None:
            content_type = data_type
        payload, resolved_type = _normalize_content(content, content_type)
        self._inner.add(role, payload, resolved_type, embedding, bot_id, session_id)

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

    def list(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        """Return stored entries.

        Args:
            limit: Maximum number of entries to return. If None, returns all.
            offset: Number of entries to skip before returning results.

        Returns:
            List of entry dicts with keys: id, run_id, role, content_type,
            text, binary, embedding, created_at, state_metadata.
        """
        results = self._inner.list(limit, offset)
        return [_normalize_record(item) for item in results]

    def compact(
        self,
        *,
        target_rows_per_fragment: int | None = None,
        materialize_deletions: bool = True,
    ) -> dict[str, int]:
        """Manually trigger compaction.

        Compaction merges small fragments into larger ones, improving
        read performance and reducing storage overhead.

        Args:
            target_rows_per_fragment: Target rows per fragment (default: 1M)
            materialize_deletions: Remove deleted rows during compaction

        Returns:
            Metrics dict with:
                - fragments_removed: Number of old fragments removed
                - fragments_added: Number of new fragments created
                - files_removed: Number of data files removed
                - files_added: Number of data files created

        Example:
            >>> ctx = Context.create("context.lance")
            >>> for i in range(100):
            ...     ctx.add("user", f"message {i}")
            >>> metrics = ctx.compact()
            >>> print(f"Reduced fragments by {metrics['fragments_removed']}")
        """
        return self._inner.compact(target_rows_per_fragment, materialize_deletions)

    def compaction_stats(self) -> dict[str, Any]:
        """Get current compaction statistics.

        Returns:
            Stats dict with:
                - total_fragments: Current fragment count
                - is_compacting: Whether compaction is running
                - last_compaction: ISO timestamp of last compaction
                - last_error: Error message from last failed compaction
                - total_compactions: Total successful compactions

        Example:
            >>> stats = ctx.compaction_stats()
            >>> if stats['total_fragments'] > 50:
            ...     ctx.compact()
        """
        return self._inner.compaction_stats()

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
