from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime
from io import BytesIO
from typing import Any

from ._internal import Context as _Context  # pyright: ignore[reportMissingImports]
from ._internal import (  # pyright: ignore[reportMissingImports]
    ContextNamespace as _ContextNamespace,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    RemoteContext as _RemoteContext,
)
from ._internal import version as _version  # pyright: ignore[reportMissingImports]
from .embeddings import EmbeddingProvider, _build_provider

__all__ = [
    "AsyncContext",
    "Context",
    "ContextNamespace",
    "EmbeddingProvider",
    "RemoteContext",
    "__version__",
]

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


def _coerce_timestamp(value: datetime | str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{field_name} must include timezone information")
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    raise TypeError(f"{field_name} must be a datetime, RFC3339 string, or None")


def _normalize_timestamp(value: Any) -> Any:
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw record dict from the Rust layer."""
    return {
        "id": raw.get("id"),
        "external_id": raw.get("external_id"),
        "run_id": raw.get("run_id"),
        "bot_id": raw.get("bot_id"),
        "session_id": raw.get("session_id"),
        "tenant": raw.get("tenant"),
        "source": raw.get("source"),
        "role": raw.get("role"),
        "content_type": raw.get("content_type"),
        "text": raw.get("text_payload"),
        "binary": raw.get("binary_payload"),
        "embedding": raw.get("embedding"),
        "created_at": _normalize_timestamp(raw.get("created_at")),
        "state_metadata": raw.get("state_metadata"),
        "metadata": raw.get("metadata"),
        "relationships": raw.get("relationships") or [],
        "expires_at": _normalize_timestamp(raw.get("expires_at")),
        "retention_policy": raw.get("retention_policy"),
        "lifecycle_status": raw.get("lifecycle_status"),
        "retired_at": _normalize_timestamp(raw.get("retired_at")),
        "retired_reason": raw.get("retired_reason"),
        "supersedes_id": raw.get("supersedes_id"),
        "superseded_by_id": raw.get("superseded_by_id"),
    }


def _normalize_search_hit(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a search hit - adds distance to the base record."""
    result = _normalize_record(raw)
    result["distance"] = raw.get("distance")
    return result


def _normalize_retrieve_hit(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a retrieve hit with hybrid ranking diagnostics."""
    result = _normalize_record(raw)
    result["score"] = raw.get("score")
    result["vector_distance"] = raw.get("vector_distance")
    result["text_score"] = raw.get("text_score")
    result["matched_channels"] = list(raw.get("matched_channels") or [])
    return result


def _normalize_state_metadata(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("state_metadata must be a mapping")
    return {
        "step": value.get("step"),
        "active_plan_id": value.get("active_plan_id"),
        "tokens_used": value.get("tokens_used"),
        "custom": value.get("custom"),
    }


_AWS_KWARG_MAP: dict[str, str] = {
    "aws_access_key_id": "aws_access_key_id",
    "aws_secret_access_key": "aws_secret_access_key",
    "aws_session_token": "aws_session_token",
    "region": "aws_region",
    "endpoint_url": "aws_endpoint_url",
}


def _json_dumps(value: Any | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON-serializable") from exc


def _merge_storage_options(
    storage_options: dict[str, Any] | None,
    *,
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
    aws_session_token: str | None,
    region: str | None,
    endpoint_url: str | None,
    allow_http: bool,
) -> dict[str, Any]:
    """Merge deprecated AWS-specific kwargs into a generic storage_options dict.

    Emits a single DeprecationWarning when any AWS kwarg is used so callers
    can migrate to the generic `storage_options` path (which works for S3,
    GCS, Azure, and any other lance/object_store backend).
    """
    options: dict[str, Any] = dict(storage_options or {})

    aws_kwargs = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
        "region": region,
        "endpoint_url": endpoint_url,
    }
    used = [name for name, value in aws_kwargs.items() if value is not None]
    if allow_http:
        used.append("allow_http")

    if used:
        warnings.warn(
            "The AWS-specific kwargs "
            f"({', '.join(sorted(used))}) are deprecated and will be removed in a "
            "future release. Pass credentials via the generic "
            "`storage_options` dict instead (e.g. "
            "storage_options={'aws_access_key_id': ..., "
            "'aws_secret_access_key': ...} for S3, or "
            "storage_options={'google_service_account_key': ...} for GCS).",
            DeprecationWarning,
            stacklevel=3,
        )

    for kwarg_name, option_key in _AWS_KWARG_MAP.items():
        value = aws_kwargs[kwarg_name]
        if value is not None:
            options.setdefault(option_key, value)
    if allow_http:
        options.setdefault("aws_allow_http", True)

    return options


class Context:
    """Multimodal, versioned context store backed by Lance.

    Storage backends are configured via the generic ``storage_options`` dict,
    aligned with the conventions used by ``lance`` and ``lance-graph``. Any
    keys understood by the underlying ``object_store`` crate are accepted.

    Examples:
        Local filesystem::

            Context.create("/tmp/context.lance")

        Amazon S3 (or S3-compatible endpoints like MinIO / moto)::

            Context.create(
                "s3://bucket/prefix/context.lance",
                storage_options={
                    "aws_access_key_id": "...",
                    "aws_secret_access_key": "...",
                    "aws_region": "us-east-1",
                    "aws_endpoint_url": "http://localhost:9000",  # optional
                    "aws_allow_http": "true",                      # optional
                },
            )

        Google Cloud Storage::

            Context.create(
                "gs://bucket/prefix/context.lance",
                storage_options={
                    # Any one of these is enough; pick whatever fits your
                    # deployment (inline JSON, file path, or ADC).
                    "google_service_account_key": service_account_json,
                    # "google_service_account_path": "/path/to/sa.json",
                    # "google_application_credentials": "/path/to/adc.json",
                },
            )

        Azure Blob Storage::

            Context.create(
                "az://container/prefix/context.lance",
                storage_options={
                    "azure_storage_account_name": "...",
                    "azure_storage_account_key": "...",
                },
            )
    """

    def __init__(
        self,
        uri: str,
        *,
        storage_options: dict[str, Any] | None = None,
        # --- Deprecated AWS-specific shortcuts (kept for backwards compat). ---
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        allow_http: bool = False,
        # --- Compaction configuration. ---
        enable_background_compaction: bool = False,
        compaction_interval_secs: int = 300,
        compaction_min_fragments: int = 5,
        compaction_target_rows: int = 1_000_000,
        quiet_hours: list[tuple[int, int]] | None = None,
        id_index_type: str | None = None,
        embedding_dim: int | None = None,
        distance_metric: str | None = None,
        # --- Embedding provider. ---
        embedding_provider: EmbeddingProvider | dict[str, Any] | None = None,
    ) -> None:
        options = _merge_storage_options(
            storage_options,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region=region,
            endpoint_url=endpoint_url,
            allow_http=allow_http,
        )

        compaction_config = {
            "enabled": enable_background_compaction,
            "check_interval_secs": compaction_interval_secs,
            "min_fragments": compaction_min_fragments,
            "target_rows_per_fragment": compaction_target_rows,
            "quiet_hours": quiet_hours or [],
        }

        if (
            options
            or compaction_config["enabled"]
            or id_index_type
            or embedding_dim is not None
            or distance_metric
        ):
            self._inner = _Context.create(
                uri,
                storage_options=options or None,
                compaction_config=compaction_config,
                id_index_type=id_index_type,
                embedding_dim=embedding_dim,
                distance_metric=distance_metric,
            )
        else:
            self._inner = _Context.create(uri)

        if isinstance(embedding_provider, dict):
            self._embedding_provider: EmbeddingProvider | None = _build_provider(
                embedding_provider
            )
        else:
            self._embedding_provider = embedding_provider
        self._default_fields: dict[str, str] = {}

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
        enable_background_compaction: bool = False,
        compaction_interval_secs: int = 300,
        compaction_min_fragments: int = 5,
        compaction_target_rows: int = 1_000_000,
        quiet_hours: list[tuple[int, int]] | None = None,
        id_index_type: str | None = None,
        embedding_dim: int | None = None,
        distance_metric: str | None = None,
        embedding_provider: EmbeddingProvider | dict[str, Any] | None = None,
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
            id_index_type=id_index_type,
            embedding_dim=embedding_dim,
            distance_metric=distance_metric,
            embedding_provider=embedding_provider,
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
        tenant: str | None = None,
        source: str | None = None,
        external_id: str | None = None,
        state_metadata: Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
        supersedes_id: str | None = None,
        superseded_by_id: str | None = None,
    ) -> None:
        if content_type is not None and data_type is not None:
            raise ValueError("Specify only one of content_type or data_type")
        if content_type is None:
            content_type = data_type
        payload, resolved_type = _normalize_content(content, content_type)
        provider = getattr(self, "_embedding_provider", None)
        if embedding is None and provider is not None and isinstance(payload, str):
            embedding = provider.embed_texts([payload])[0]
        defaults = getattr(self, "_default_fields", {})
        if bot_id is None:
            bot_id = defaults.get("bot_id")
        if session_id is None:
            session_id = defaults.get("session_id")
        if tenant is None:
            tenant = defaults.get("tenant")
        if source is None:
            source = defaults.get("source")
        self._inner.add(
            role,
            payload,
            resolved_type,
            embedding,
            bot_id,
            session_id,
            tenant,
            source,
            external_id,
            _normalize_state_metadata(state_metadata),
            _json_dumps(metadata, "metadata"),
            _coerce_timestamp(expires_at, field_name="expires_at"),
            retention_policy,
            lifecycle_status,
            _coerce_timestamp(retired_at, field_name="retired_at"),
            retired_reason,
            supersedes_id,
            superseded_by_id,
            _json_dumps(relationships, "relationships"),
        )

    def upsert(
        self,
        role: str,
        content: Any,
        content_type: str | None = None,
        data_type: str | None = None,
        embedding: list[float] | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        tenant: str | None = None,
        source: str | None = None,
        external_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
        *,
        key: str = "external_id",
    ) -> dict[str, Any]:
        """Insert a record, or replace the visible record with the same external_id."""
        if content_type is not None and data_type is not None:
            raise ValueError("Specify only one of content_type or data_type")
        if key != "external_id":
            raise ValueError("Only key='external_id' is currently supported")
        if not external_id:
            raise ValueError("upsert requires external_id")
        if content_type is None:
            content_type = data_type

        payload, resolved_type = _normalize_content(content, content_type)
        provider = getattr(self, "_embedding_provider", None)
        if embedding is None and provider is not None and isinstance(payload, str):
            embedding = provider.embed_texts([payload])[0]
        defaults = getattr(self, "_default_fields", {})
        if bot_id is None:
            bot_id = defaults.get("bot_id")
        if session_id is None:
            session_id = defaults.get("session_id")
        if tenant is None:
            tenant = defaults.get("tenant")
        if source is None:
            source = defaults.get("source")
        result = self._inner.upsert(
            role,
            payload,
            resolved_type,
            embedding,
            bot_id,
            session_id,
            tenant,
            source,
            external_id,
            _json_dumps(metadata, "metadata"),
            _coerce_timestamp(expires_at, field_name="expires_at"),
            retention_policy,
            lifecycle_status,
            _coerce_timestamp(retired_at, field_name="retired_at"),
            retired_reason,
            _json_dumps(relationships, "relationships"),
            key,
        )
        return {
            "inserted": bool(result["inserted"]),
            "replaced_id": result.get("replaced_id"),
            "version": result["version"],
            "record": _normalize_record(result["record"]),
        }

    def update(
        self,
        *,
        id: str | None = None,
        external_id: str | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        tenant: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        """Patch mutable fields on a visible record by id or external_id.

        Pass ``embedding`` to attach or replace a record's vector after it was
        appended without one (deferred / enrich-later ingestion). The updated
        record participates in vector search once the embedding is set.
        """
        if (id is None) == (external_id is None):
            raise ValueError("Specify exactly one of id or external_id")
        if (
            bot_id is None
            and session_id is None
            and tenant is None
            and source is None
            and metadata is None
            and relationships is None
            and expires_at is None
            and retention_policy is None
            and lifecycle_status is None
            and retired_at is None
            and retired_reason is None
            and embedding is None
        ):
            raise ValueError("update requires at least one patch field")

        result = self._inner.update(
            id,
            external_id,
            bot_id,
            session_id,
            tenant,
            source,
            _json_dumps(metadata, "metadata"),
            _json_dumps(relationships, "relationships"),
            _coerce_timestamp(expires_at, field_name="expires_at"),
            retention_policy,
            lifecycle_status,
            _coerce_timestamp(retired_at, field_name="retired_at"),
            retired_reason,
            embedding,
        )
        record = result.get("record")
        return {
            "updated": bool(result["updated"]),
            "replaced_id": result.get("replaced_id"),
            "version": result["version"],
            "record": _normalize_record(record) if record is not None else None,
        }

    def add_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Append multiple records in one storage operation.

        Each record accepts the same fields as :meth:`add`: ``role``,
        ``content``, optional ``content_type``/``data_type``, ``embedding``,
        ``bot_id``, ``session_id``, ``tenant``, ``source``, ``external_id``,
        ``state_metadata``,
        ``metadata``, ``relationships``, and lifecycle fields such as
        ``expires_at`` and ``lifecycle_status``.
        """
        normalized: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"records[{index}] must be a mapping")
            if "role" not in record:
                raise ValueError(f"records[{index}] is missing required key 'role'")
            if "content" not in record:
                raise ValueError(f"records[{index}] is missing required key 'content'")

            content_type = record.get("content_type")
            data_type = record.get("data_type")
            if content_type is not None and data_type is not None:
                raise ValueError(
                    f"records[{index}] specifies both content_type and data_type"
                )
            if content_type is None:
                content_type = data_type

            payload, resolved_type = _normalize_content(record["content"], content_type)
            normalized.append(
                {
                    "role": record["role"],
                    "content": payload,
                    "data_type": resolved_type,
                    "embedding": record.get("embedding"),
                    "bot_id": record.get(
                        "bot_id", getattr(self, "_default_fields", {}).get("bot_id")
                    ),
                    "session_id": record.get(
                        "session_id",
                        getattr(self, "_default_fields", {}).get("session_id"),
                    ),
                    "tenant": record.get(
                        "tenant", getattr(self, "_default_fields", {}).get("tenant")
                    ),
                    "source": record.get(
                        "source", getattr(self, "_default_fields", {}).get("source")
                    ),
                    "external_id": record.get("external_id"),
                    "state_metadata": _normalize_state_metadata(
                        record.get("state_metadata")
                    ),
                    "metadata_json": _json_dumps(record.get("metadata"), "metadata"),
                    "relationships_json": _json_dumps(
                        record.get("relationships"), "relationships"
                    ),
                    "expires_at": _coerce_timestamp(
                        record.get("expires_at"),
                        field_name=f"records[{index}].expires_at",
                    ),
                    "retention_policy": record.get("retention_policy"),
                    "lifecycle_status": record.get("lifecycle_status"),
                    "retired_at": _coerce_timestamp(
                        record.get("retired_at"),
                        field_name=f"records[{index}].retired_at",
                    ),
                    "retired_reason": record.get("retired_reason"),
                    "supersedes_id": record.get("supersedes_id"),
                    "superseded_by_id": record.get("superseded_by_id"),
                }
            )

        self._auto_embed_batch(normalized)
        self._inner.add_many(normalized)

    def _auto_embed_batch(self, records: list[dict[str, Any]]) -> None:
        """Embed text records without an embedding in one provider call."""
        provider = getattr(self, "_embedding_provider", None)
        if provider is None:
            return
        indices = [
            i
            for i, r in enumerate(records)
            if r.get("embedding") is None and isinstance(r.get("content"), str)
        ]
        if not indices:
            return
        texts = [records[i]["content"] for i in indices]
        vectors = provider.embed_texts(texts)
        for i, vec in zip(indices, vectors):
            records[i]["embedding"] = vec

    def snapshot(self, label: str | None = None) -> str:
        return self._inner.snapshot(label)

    def fork(self, branch_name: str) -> Context:
        inner = self._inner.fork(branch_name)
        return self._from_inner(inner, self._embedding_provider, self._default_fields)

    def checkout(self, version_id: int | str) -> None:
        self._inner.checkout(int(version_id))

    def search(
        self,
        query: Any,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        *,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
    ) -> list[dict[str, Any]]:
        provider = getattr(self, "_embedding_provider", None)
        if isinstance(query, str) and provider is not None:
            query = provider.embed_texts([query])[0]
        vector = _coerce_vector(query)
        results = self._inner.search(
            vector,
            limit,
            _json_dumps(filters, "filters"),
            include_expired,
            include_retired,
            include_relationships,
        )
        return [_normalize_search_hit(item) for item in results]

    def retrieve(
        self,
        *,
        text: str | None = None,
        vector: Any | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
        fusion: str = "rrf",
    ) -> list[dict[str, Any]]:
        if text is None and vector is None:
            raise ValueError("retrieve requires text or vector")
        if fusion != "rrf":
            raise ValueError("retrieve fusion currently supports only 'rrf'")

        coerced_vector = _coerce_vector(vector) if vector is not None else None
        results = self._inner.retrieve(
            text,
            coerced_vector,
            limit,
            _json_dumps(filters, "filters"),
            include_expired,
            include_retired,
            include_relationships,
            fusion,
        )
        return [_normalize_retrieve_hit(item) for item in results]

    def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
        *,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> list[dict[str, Any]]:
        """Return stored entries.

        Args:
            limit: Maximum number of entries to return. If None, returns all.
            offset: Number of entries to skip before returning results.
            filters: Optional equality filters for built-in fields
                (bot_id, session_id, tenant, source, role, content_type),
                created_at range filters, or metadata fields.
            include_expired: Include records whose ``expires_at`` is in the past.
            include_retired: Include retired/superseded/revoked records.

        Returns:
            List of entry dicts with keys: id, run_id, role, content_type,
            text, binary, embedding, created_at, metadata, state_metadata, and
            lifecycle metadata.
        """
        results = self._inner.list(
            limit,
            offset,
            _json_dumps(filters, "filters"),
            include_expired,
            include_retired,
        )
        return [_normalize_record(item) for item in results]

    def get(
        self, *, id: str | None = None, external_id: str | None = None
    ) -> dict[str, Any] | None:
        """Return one entry by internal id or caller-supplied external id."""
        if (id is None) == (external_id is None):
            raise ValueError("Specify exactly one of id or external_id")
        result = self._inner.get(id, external_id)
        if result is None:
            return None
        return _normalize_record(result)

    def related(
        self,
        target_id: str,
        relation: str | None = None,
        limit: int | None = None,
        *,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> list[dict[str, Any]]:
        """Return records with relationships that point at ``target_id``."""
        results = self._inner.related(
            target_id,
            relation,
            limit,
            include_expired,
            include_retired,
        )
        return [_normalize_record(item) for item in results]

    def migrate_relationships(self) -> bool:
        """Add the relationships column to an older dataset if it is missing."""
        return bool(self._inner.migrate_relationships())

    def delete(self, *, id: str | None = None, external_id: str | None = None) -> bool:
        """Logically forget one entry by internal id or caller-supplied external id.

        Returns True when an entry was found and tombstoned, False when the
        identifier is already absent. This is a versioned logical delete:
        default reads hide the entry, but older dataset versions and underlying
        storage files may still contain the original payload until retention or
        physical cleanup policies remove them.
        """
        if (id is None) == (external_id is None):
            raise ValueError("Specify exactly one of id or external_id")
        return bool(self._inner.delete(id, external_id))

    def forget(self, *, id: str | None = None, external_id: str | None = None) -> bool:
        """Alias for :meth:`delete`."""
        return self.delete(id=id, external_id=external_id)

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
    def _from_inner(
        cls,
        inner: _Context,
        embedding_provider: EmbeddingProvider | None = None,
        default_fields: Mapping[str, str] | None = None,
    ) -> Context:
        obj = cls.__new__(cls)
        obj._inner = inner
        obj._embedding_provider = embedding_provider
        obj._default_fields = dict(default_fields or {})
        return obj


class ContextNamespace:
    """Resolver for physically isolated context partitions under one namespace root."""

    def __init__(
        self,
        root_uri: str,
        fields: Iterable[str],
        *,
        storage_options: dict[str, Any] | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
        allow_http: bool = False,
        enable_background_compaction: bool = False,
        compaction_interval_secs: int = 300,
        compaction_min_fragments: int = 5,
        compaction_target_rows: int = 1_000_000,
        quiet_hours: list[tuple[int, int]] | None = None,
        id_index_type: str | None = None,
        embedding_dim: int | None = None,
        distance_metric: str | None = None,
        embedding_provider: EmbeddingProvider | dict[str, Any] | None = None,
    ) -> None:
        options = _merge_storage_options(
            storage_options,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region=region,
            endpoint_url=endpoint_url,
            allow_http=allow_http,
        )

        compaction_config = {
            "enabled": enable_background_compaction,
            "check_interval_secs": compaction_interval_secs,
            "min_fragments": compaction_min_fragments,
            "target_rows_per_fragment": compaction_target_rows,
            "quiet_hours": quiet_hours or [],
        }

        field_list = list(fields)
        if (
            options
            or compaction_config["enabled"]
            or id_index_type
            or embedding_dim is not None
            or distance_metric
        ):
            self._inner = _ContextNamespace.create(
                root_uri,
                field_list,
                storage_options=options or None,
                compaction_config=compaction_config,
                id_index_type=id_index_type,
                embedding_dim=embedding_dim,
                distance_metric=distance_metric,
            )
        else:
            self._inner = _ContextNamespace.create(root_uri, field_list)

        if isinstance(embedding_provider, dict):
            self._embedding_provider: EmbeddingProvider | None = _build_provider(
                embedding_provider
            )
        else:
            self._embedding_provider = embedding_provider

    @classmethod
    def create(
        cls,
        root_uri: str,
        fields: Iterable[str],
        **kwargs: Any,
    ) -> ContextNamespace:
        return cls(root_uri, fields, **kwargs)

    def root_uri(self) -> str:
        return self._inner.root_uri()

    def manifest_uri(self) -> str:
        return self._inner.manifest_uri()

    def partition_uri(self, **selector: str) -> str:
        return self._inner.partition_uri(selector)

    def context(self, **selector: str) -> Context:
        inner = self._inner.context(selector)
        return Context._from_inner(inner, self._embedding_provider, selector)

    def partitions(self) -> list[dict[str, Any]]:
        return list(self._inner.partitions())


class AsyncContext:
    """Async wrapper around :class:`Context`.

    Every I/O method is dispatched to a thread-pool executor via
    :func:`asyncio.get_running_loop().run_in_executor`. The underlying Rust
    code releases the GIL during I/O, so the executor thread is only occupied
    briefly for the Python ↔ Rust boundary crossing.

    Usage::

        ctx = await AsyncContext.create("/tmp/context.lance")
        await ctx.add("user", "hello")
        results = await ctx.list()
    """

    def __init__(self, sync_ctx: Context) -> None:
        self._sync = sync_ctx

    @classmethod
    async def create(
        cls,
        uri: str,
        **kwargs: Any,
    ) -> AsyncContext:
        loop = asyncio.get_running_loop()
        sync_ctx = await loop.run_in_executor(
            None, lambda: Context.create(uri, **kwargs)
        )
        return cls(sync_ctx)

    def uri(self) -> str:
        return self._sync.uri()

    def branch(self) -> str:
        return self._sync.branch()

    def entries(self) -> int:
        return self._sync.entries()

    def version(self) -> int:
        return self._sync.version()

    async def add(
        self,
        role: str,
        content: Any,
        content_type: str | None = None,
        data_type: str | None = None,
        embedding: list[float] | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        tenant: str | None = None,
        source: str | None = None,
        external_id: str | None = None,
        state_metadata: Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
        supersedes_id: str | None = None,
        superseded_by_id: str | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._sync.add(
                role,
                content,
                content_type=content_type,
                data_type=data_type,
                embedding=embedding,
                bot_id=bot_id,
                session_id=session_id,
                tenant=tenant,
                source=source,
                external_id=external_id,
                state_metadata=state_metadata,
                metadata=metadata,
                relationships=relationships,
                expires_at=expires_at,
                retention_policy=retention_policy,
                lifecycle_status=lifecycle_status,
                retired_at=retired_at,
                retired_reason=retired_reason,
                supersedes_id=supersedes_id,
                superseded_by_id=superseded_by_id,
            ),
        )

    async def upsert(
        self,
        role: str,
        content: Any,
        content_type: str | None = None,
        data_type: str | None = None,
        embedding: list[float] | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        tenant: str | None = None,
        source: str | None = None,
        external_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
        *,
        key: str = "external_id",
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.upsert(
                role,
                content,
                content_type=content_type,
                data_type=data_type,
                embedding=embedding,
                bot_id=bot_id,
                session_id=session_id,
                tenant=tenant,
                source=source,
                external_id=external_id,
                metadata=metadata,
                relationships=relationships,
                expires_at=expires_at,
                retention_policy=retention_policy,
                lifecycle_status=lifecycle_status,
                retired_at=retired_at,
                retired_reason=retired_reason,
                key=key,
            ),
        )

    async def update(
        self,
        *,
        id: str | None = None,
        external_id: str | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        tenant: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.update(
                id=id,
                external_id=external_id,
                bot_id=bot_id,
                session_id=session_id,
                tenant=tenant,
                source=source,
                metadata=metadata,
                relationships=relationships,
                expires_at=expires_at,
                retention_policy=retention_policy,
                lifecycle_status=lifecycle_status,
                retired_at=retired_at,
                retired_reason=retired_reason,
            ),
        )

    def snapshot(self, label: str | None = None) -> str:
        return self._sync.snapshot(label)

    def fork(self, branch_name: str) -> AsyncContext:
        sync_fork = self._sync.fork(branch_name)
        return AsyncContext(sync_fork)

    async def checkout(self, version_id: int | str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._sync.checkout(version_id))

    async def search(
        self,
        query: Any,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        *,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.search(
                query,
                limit,
                filters,
                include_expired=include_expired,
                include_retired=include_retired,
                include_relationships=include_relationships,
            ),
        )

    async def retrieve(
        self,
        *,
        text: str | None = None,
        vector: Any | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
        fusion: str = "rrf",
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.retrieve(
                text=text,
                vector=vector,
                limit=limit,
                filters=filters,
                include_expired=include_expired,
                include_retired=include_retired,
                include_relationships=include_relationships,
                fusion=fusion,
            ),
        )

    async def related(
        self,
        target_id: str,
        relation: str | None = None,
        limit: int | None = None,
        *,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.related(
                target_id,
                relation,
                limit,
                include_expired=include_expired,
                include_retired=include_retired,
            ),
        )

    async def migrate_relationships(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.migrate_relationships)

    async def get(
        self, *, id: str | None = None, external_id: str | None = None
    ) -> dict[str, Any] | None:
        """Asynchronously retrieve a single context record by id or external_id."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._sync.get(id=id, external_id=external_id)
        )

    async def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
        *,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.list(
                limit,
                offset,
                filters,
                include_expired=include_expired,
                include_retired=include_retired,
            ),
        )

    async def compact(
        self,
        *,
        target_rows_per_fragment: int | None = None,
        materialize_deletions: bool = True,
    ) -> dict[str, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.compact(
                target_rows_per_fragment=target_rows_per_fragment,
                materialize_deletions=materialize_deletions,
            ),
        )

    async def compaction_stats(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.compaction_stats)

    def __repr__(self) -> str:
        return (
            f"AsyncContext(uri={self._sync.uri()!r}, "
            f"branch={self._sync.branch()!r}, "
            f"entries={self._sync.entries()})"
        )


class RemoteContext:
    """Async wrapper for a remote lance-context server over HTTP."""

    def __init__(self, sync_ctx: _RemoteContext) -> None:
        self._sync = sync_ctx

    @classmethod
    async def connect(cls, base_url: str, name: str) -> "RemoteContext":
        """Connect to a remote context server."""
        loop = asyncio.get_running_loop()
        sync_ctx = await loop.run_in_executor(
            None, lambda: _RemoteContext.connect(base_url, name)
        )
        return cls(sync_ctx)

    def version(self) -> int:
        """Return the current version (cached, sync)."""
        return self._sync.version()

    async def add(
        self,
        *,
        role: str = "user",
        content: Any = None,
        content_type: str | None = None,
        embedding: Iterable[float] | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        external_id: str | None = None,
        state_metadata: Mapping[str, Any] | None = None,
        metadata: Any = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        supersedes_id: str | None = None,
        relationships: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Add a record to the remote context."""
        payload, data_type = _normalize_content(content, content_type)
        emb = _coerce_vector(embedding) if embedding is not None else None
        sm = _normalize_state_metadata(state_metadata)
        meta_json = _json_dumps(metadata, "metadata")
        rel_list = list(relationships) if relationships else None
        rel_json = _json_dumps(rel_list, "relationships")
        exp = _coerce_timestamp(expires_at, field_name="expires_at")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.add(
                role=role,
                content=payload,
                data_type=data_type,
                embedding=emb,
                bot_id=bot_id,
                session_id=session_id,
                external_id=external_id,
                state_metadata=sm,
                metadata_json=meta_json,
                expires_at=exp,
                retention_policy=retention_policy,
                supersedes_id=supersedes_id,
                relationships_json=rel_json,
            ),
        )

    async def upsert(
        self,
        *,
        role: str = "user",
        content: Any = None,
        content_type: str | None = None,
        embedding: Iterable[float] | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        external_id: str | None = None,
        metadata: Any = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        supersedes_id: str | None = None,
        relationships: Iterable[Mapping[str, Any]] | None = None,
        key: str = "external_id",
    ) -> dict[str, Any]:
        """Upsert a record by external_id."""
        payload, data_type = _normalize_content(content, content_type)
        emb = _coerce_vector(embedding) if embedding is not None else None
        meta_json = _json_dumps(metadata, "metadata")
        rel_list = list(relationships) if relationships else None
        rel_json = _json_dumps(rel_list, "relationships")
        exp = _coerce_timestamp(expires_at, field_name="expires_at")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._sync.upsert(
                role=role,
                content=payload,
                data_type=data_type,
                embedding=emb,
                bot_id=bot_id,
                session_id=session_id,
                external_id=external_id,
                metadata_json=meta_json,
                expires_at=exp,
                retention_policy=retention_policy,
                supersedes_id=supersedes_id,
                relationships_json=rel_json,
                key=key,
            ),
        )
        result["record"] = _normalize_record(result["record"])
        return result

    async def update(
        self,
        *,
        id: str | None = None,
        external_id: str | None = None,
        bot_id: str | None = None,
        session_id: str | None = None,
        metadata: Any = None,
        relationships: Iterable[Mapping[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
        embedding: Iterable[float] | None = None,
    ) -> dict[str, Any]:
        """Update a record by id or external_id."""
        meta_json = _json_dumps(metadata, "metadata")
        rel_list = list(relationships) if relationships else None
        rel_json = _json_dumps(rel_list, "relationships")
        exp = _coerce_timestamp(expires_at, field_name="expires_at")
        ret = _coerce_timestamp(retired_at, field_name="retired_at")
        emb = _coerce_vector(embedding) if embedding is not None else None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._sync.update(
                id=id,
                external_id=external_id,
                bot_id=bot_id,
                session_id=session_id,
                metadata_json=meta_json,
                relationships_json=rel_json,
                expires_at=exp,
                retention_policy=retention_policy,
                lifecycle_status=lifecycle_status,
                retired_at=ret,
                retired_reason=retired_reason,
                embedding=emb,
            ),
        )
        if result.get("record") is not None:
            result["record"] = _normalize_record(result["record"])
        return result

    async def get(
        self,
        *,
        id: str | None = None,
        external_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get a record by id or external_id."""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._sync.get(id=id, external_id=external_id),
        )
        return _normalize_record(result) if result is not None else None

    async def delete(
        self,
        *,
        id: str | None = None,
        external_id: str | None = None,
    ) -> bool:
        """Delete a record by id or external_id."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.delete(id=id, external_id=external_id),
        )

    async def list(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        filters: Any = None,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> list[dict[str, Any]]:
        """List records with optional filters."""
        filters_json = _json_dumps(filters, "filters")
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._sync.list(
                limit=limit,
                offset=offset,
                filters_json=filters_json,
                include_expired=include_expired,
                include_retired=include_retired,
            ),
        )
        return [_normalize_record(r) for r in results]

    async def related(
        self,
        target_id: str,
        *,
        relation: str | None = None,
        limit: int | None = None,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> list[dict[str, Any]]:
        """Get records related to a target."""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._sync.related(
                target_id=target_id,
                relation=relation,
                limit=limit,
                include_expired=include_expired,
                include_retired=include_retired,
            ),
        )
        return [_normalize_record(r) for r in results]

    async def search(
        self,
        query: Iterable[float],
        *,
        limit: int | None = None,
        filters: Any = None,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
    ) -> list[dict[str, Any]]:
        """Vector search over the context."""
        q = _coerce_vector(query)
        filters_json = _json_dumps(filters, "filters")
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._sync.search(
                query=q,
                limit=limit,
                filters_json=filters_json,
                include_expired=include_expired,
                include_retired=include_retired,
                include_relationships=include_relationships,
            ),
        )
        return [_normalize_search_hit(h) for h in results]

    async def retrieve(
        self,
        *,
        text: str | None = None,
        vector: Iterable[float] | None = None,
        limit: int | None = None,
        filters: Any = None,
        include_expired: bool = False,
        include_retired: bool = False,
        include_relationships: bool = False,
        fusion: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid retrieval (text + vector)."""
        v = _coerce_vector(vector) if vector is not None else None
        filters_json = _json_dumps(filters, "filters")
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._sync.retrieve(
                text=text,
                vector=v,
                limit=limit,
                filters_json=filters_json,
                include_expired=include_expired,
                include_retired=include_retired,
                include_relationships=include_relationships,
                fusion=fusion,
            ),
        )
        return [_normalize_retrieve_hit(h) for h in results]

    async def checkout(self, version: int) -> None:
        """Checkout a specific version."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._sync.checkout(version),
        )

    async def compact(
        self,
        *,
        target_rows_per_fragment: int | None = None,
        materialize_deletions: bool | None = None,
    ) -> dict[str, Any]:
        """Run compaction on the remote context."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.compact(
                target_rows_per_fragment=target_rows_per_fragment,
                materialize_deletions=materialize_deletions,
            ),
        )

    async def compaction_stats(self) -> dict[str, Any]:
        """Get compaction statistics."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync.compaction_stats)

    def __repr__(self) -> str:
        return f"RemoteContext(version={self._sync.version()})"
