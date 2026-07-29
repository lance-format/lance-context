from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike

from ._internal import (  # pyright: ignore[reportMissingImports]
    AlreadyExistsError as _AlreadyExistsError,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    CompactionInProgressError as _CompactionInProgressError,
)
from ._internal import Context as _Context  # pyright: ignore[reportMissingImports]
from ._internal import (  # pyright: ignore[reportMissingImports]
    ContextNamespace as _ContextNamespace,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    ContextStoreError as _ContextStoreError,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    DatagenItemId as _DatagenItemId,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    DatagenStore as _DatagenStore,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    DatagenStreamWriter as _DatagenStreamWriter,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    GenericStore as _GenericStore,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    InternalError as _InternalError,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    InvalidRequestError as _InvalidRequestError,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    NotFoundError as _NotFoundError,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    RemoteContext as _RemoteContext,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    RolloutStore as _RolloutStore,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    datagen_event_id as _datagen_event_id,
)
from ._internal import (  # pyright: ignore[reportMissingImports]
    generate_id as _generate_id,
)
from ._internal import version as _version  # pyright: ignore[reportMissingImports]
from .embeddings import EmbeddingProvider, _build_provider, supports_media

__all__ = [
    "AlreadyExistsError",
    "AsyncContext",
    "AsyncRolloutStore",
    "CompactionInProgressError",
    "Context",
    "ContextNamespace",
    "ContextStoreError",
    "GenericStore",
    "DatagenItemId",
    "DatagenStore",
    "DatagenStreamWriter",
    "EmbeddingProvider",
    "InternalError",
    "InvalidRequestError",
    "NotFoundError",
    "RemoteContext",
    "RolloutStore",
    "__version__",
    "datagen_event_id",
    "generate_id",
]

__version__ = _version()

# The store error hierarchy, re-exported from the native module. `ContextStoreError` is
# rooted at `RuntimeError` — what this package raised before these classes existed — so
# `except RuntimeError` still catches every one of them. `InternalError` and
# `CompactionInProgressError` are the retryable pair; the rest are verdicts on the
# request itself, which an identical retry can only earn again.
ContextStoreError = _ContextStoreError
NotFoundError = _NotFoundError
AlreadyExistsError = _AlreadyExistsError
InvalidRequestError = _InvalidRequestError
InternalError = _InternalError
CompactionInProgressError = _CompactionInProgressError


def generate_id() -> str:
    """Generate a new time-ordered id (UUIDv7) as a string.

    Ids sort roughly in creation order, making them a good default primary key
    for append-heavy tables such as the rollout store.
    """
    return _generate_id()


def datagen_event_id(item_id: str, checkpoint_id: str, ordinal: int) -> str:
    """Compute the deterministic event id for a datagen checkpoint field write.

    The id is a pure function of `item_id`, `checkpoint_id`, and `ordinal`, so a
    retried append produces the same id and dedupes against the log.
    """
    return _datagen_event_id(item_id, checkpoint_id, ordinal)


_ARROW_STREAM_MIME = "application/vnd.apache.arrow.stream"
_DEFAULT_INGEST_BATCH_SIZE = 1000
_MISSING = object()
_INGEST_DIRECT_FIELDS = {
    "role",
    "content",
    "content_type",
    "data_type",
    "embedding",
    "bot_id",
    "session_id",
    "tenant",
    "source",
    "external_id",
    "run_id",
    "created_at",
    "state_metadata",
    "relationships",
    "expires_at",
    "retention_policy",
    "lifecycle_status",
    "retired_at",
    "retired_reason",
    "supersedes_id",
    "superseded_by_id",
    "payload_uri",
    "payload_size",
    "payload_checksum",
}


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
        "payload_uri": raw.get("payload_uri"),
        "payload_size": raw.get("payload_size"),
        "payload_checksum": raw.get("payload_checksum"),
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


def _normalize_field_map(field_map: Mapping[str, str] | None) -> dict[str, str]:
    if field_map is None:
        return {}
    if not isinstance(field_map, Mapping):
        raise TypeError("field_map must be a mapping")

    normalized: dict[str, str] = {}
    for key, value in field_map.items():
        if not isinstance(key, str):
            raise TypeError("field_map keys must be strings")
        if not isinstance(value, str):
            raise TypeError("field_map values must be strings")
        normalized[key] = value
    return normalized


def _mapped_value(
    raw: Mapping[str, Any], field_map: Mapping[str, str], field: str
) -> Any:
    source_field = field_map.get(field, field)
    if source_field in raw:
        return raw[source_field]
    return _MISSING


def _jsonable_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    normalized = dict(value)
    _json_dumps(normalized, name)
    return normalized


def _normalize_ingest_record(
    raw: Mapping[str, Any],
    *,
    field_map: Mapping[str, str],
    defaults: Mapping[str, Any],
    source: str,
    preserve_raw: bool,
    raw_metadata_key: str,
    index: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {}

    for field in _INGEST_DIRECT_FIELDS:
        value = _mapped_value(raw, field_map, field)
        if value is not _MISSING:
            record[field] = value

    for field, value in defaults.items():
        if field != "metadata":
            record.setdefault(field, value)

    record.setdefault("source", source)
    record.setdefault("role", "source")

    if "content" not in record:
        raise ValueError(f"records[{index}] is missing required ingest field 'content'")

    metadata: dict[str, Any] = {}
    mapped_metadata = _mapped_value(raw, field_map, "metadata")
    if mapped_metadata is not _MISSING:
        if not isinstance(mapped_metadata, Mapping):
            raise TypeError(f"records[{index}].metadata must be a mapping")
        metadata.update(
            _jsonable_mapping(mapped_metadata, name=f"records[{index}].metadata")
        )

    default_metadata = defaults.get("metadata")
    if default_metadata is not None:
        if not isinstance(default_metadata, Mapping):
            raise TypeError("defaults['metadata'] must be a mapping")
        merged = _jsonable_mapping(default_metadata, name="defaults['metadata']")
        merged.update(metadata)
        metadata = merged

    if preserve_raw:
        metadata.setdefault(
            raw_metadata_key,
            _jsonable_mapping(raw, name=f"records[{index}].raw"),
        )

    if metadata:
        record["metadata"] = metadata

    return record


def _validate_ingest_defaults(defaults: Mapping[str, Any] | None) -> dict[str, Any]:
    if defaults is None:
        return {}
    if not isinstance(defaults, Mapping):
        raise TypeError("defaults must be a mapping")

    normalized = dict(defaults)
    invalid = set(normalized) - (_INGEST_DIRECT_FIELDS | {"metadata"})
    if invalid:
        joined = ", ".join(sorted(invalid))
        raise ValueError(f"defaults contains unsupported record field(s): {joined}")
    return normalized


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
        run_id: str | None = None,
        created_at: datetime | str | None = None,
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
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
    ) -> None:
        if content_type is not None and data_type is not None:
            raise ValueError("Specify only one of content_type or data_type")
        if content_type is None:
            content_type = data_type
        payload, resolved_type = _normalize_content(content, content_type)
        provider = getattr(self, "_embedding_provider", None)
        if embedding is None and provider is not None:
            if isinstance(payload, str):
                embedding = provider.embed_texts([payload])[0]
            elif isinstance(payload, (bytes, bytearray)) and supports_media(provider):
                embedding = provider.embed_media([(bytes(payload), resolved_type)])[0]
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
            run_id,
            _coerce_timestamp(created_at, field_name="created_at"),
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
            payload_uri,
            payload_size,
            payload_checksum,
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
        run_id: str | None = None,
        created_at: datetime | str | None = None,
        state_metadata: Mapping[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        expires_at: datetime | str | None = None,
        retention_policy: str | None = None,
        lifecycle_status: str | None = None,
        retired_at: datetime | str | None = None,
        retired_reason: str | None = None,
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
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
        if embedding is None and provider is not None:
            if isinstance(payload, str):
                embedding = provider.embed_texts([payload])[0]
            elif isinstance(payload, (bytes, bytearray)) and supports_media(provider):
                embedding = provider.embed_media([(bytes(payload), resolved_type)])[0]
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
            run_id,
            _coerce_timestamp(created_at, field_name="created_at"),
            _normalize_state_metadata(state_metadata),
            _json_dumps(metadata, "metadata"),
            _coerce_timestamp(expires_at, field_name="expires_at"),
            retention_policy,
            lifecycle_status,
            _coerce_timestamp(retired_at, field_name="retired_at"),
            retired_reason,
            _json_dumps(relationships, "relationships"),
            payload_uri,
            payload_size,
            payload_checksum,
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
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
    ) -> dict[str, Any]:
        """Patch mutable fields on a visible record by id or external_id.

        Pass ``embedding`` to attach or replace a record's vector after it was
        appended without one (deferred / enrich-later ingestion). The updated
        record participates in vector search once the embedding is set.

        Pass ``payload_uri`` (with optional ``payload_size`` / ``payload_checksum``)
        to attach an external media reference after the record was created.
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
            and payload_uri is None
            and payload_size is None
            and payload_checksum is None
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
            payload_uri,
            payload_size,
            payload_checksum,
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
        ``run_id``, ``created_at``, ``state_metadata``,
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
                    "run_id": record.get("run_id"),
                    "created_at": _coerce_timestamp(
                        record.get("created_at"),
                        field_name=f"records[{index}].created_at",
                    ),
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
                    "payload_uri": record.get("payload_uri"),
                    "payload_size": record.get("payload_size"),
                    "payload_checksum": record.get("payload_checksum"),
                }
            )

        self._auto_embed_batch(normalized)
        self._inner.add_many(normalized)

    def upsert_many(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        key: str = "external_id",
    ) -> list[dict[str, Any]]:
        """Insert-or-replace multiple records by ``external_id`` in one call.

        Each record accepts the same fields as :meth:`add` and must include a
        non-empty ``external_id``. For each record, the currently-visible record
        with the same ``external_id`` is replaced (superseded); otherwise the
        record is inserted. Duplicate ``id``/``external_id`` values within the
        batch are rejected, and the whole batch is validated before anything is
        written.

        Returns one result mapping per input record, in order, each with
        ``inserted``, ``replaced_id``, ``version`` and the resulting ``record``.
        """
        if key != "external_id":
            raise ValueError("Only key='external_id' is currently supported")

        normalized: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"records[{index}] must be a mapping")
            if "role" not in record:
                raise ValueError(f"records[{index}] is missing required key 'role'")
            if "content" not in record:
                raise ValueError(f"records[{index}] is missing required key 'content'")
            if not record.get("external_id"):
                raise ValueError(
                    f"records[{index}] is missing required key 'external_id'"
                )

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
                    "run_id": record.get("run_id"),
                    "created_at": _coerce_timestamp(
                        record.get("created_at"),
                        field_name=f"records[{index}].created_at",
                    ),
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
                    "payload_uri": record.get("payload_uri"),
                    "payload_size": record.get("payload_size"),
                    "payload_checksum": record.get("payload_checksum"),
                }
            )

        self._auto_embed_batch(normalized)
        results = self._inner.upsert_many(normalized, key)
        return [
            {
                "inserted": bool(result["inserted"]),
                "replaced_id": result.get("replaced_id"),
                "version": result["version"],
                "record": _normalize_record(result["record"]),
            }
            for result in results
        ]

    def ingest_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        field_map: Mapping[str, str] | None = None,
        defaults: Mapping[str, Any] | None = None,
        source: str = "raw",
        batch_size: int = _DEFAULT_INGEST_BATCH_SIZE,
        mode: str = "append",
        preserve_raw: bool = True,
        raw_metadata_key: str = "raw_record",
    ) -> dict[str, int]:
        """Ingest raw row-like records into faithful ``ContextRecord`` entries.

        ``field_map`` maps context-record fields to raw input fields, for
        example ``{"content": "message", "external_id": "event_id"}``.
        ``mode="append"`` batches through :meth:`add_many`; ``mode="upsert"``
        uses each row's ``external_id`` for idempotent re-ingestion.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if mode not in {"append", "upsert"}:
            raise ValueError("mode must be 'append' or 'upsert'")
        if not source:
            raise ValueError("source must be non-empty")
        if not raw_metadata_key:
            raise ValueError("raw_metadata_key must be non-empty")

        normalized_field_map = _normalize_field_map(field_map)
        normalized_defaults = _validate_ingest_defaults(defaults)
        processed = 0
        inserted = 0
        updated = 0
        batches = 0
        batch: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal inserted, updated, batches, batch
            if not batch:
                return
            if mode == "append":
                self.add_many(batch)
                inserted += len(batch)
            else:
                for record in batch:
                    external_id = record.get("external_id")
                    if not external_id:
                        raise ValueError(
                            "mode='upsert' requires external_id on every record"
                        )
                    result = self.upsert(
                        record["role"],
                        record["content"],
                        content_type=record.get("content_type"),
                        data_type=record.get("data_type"),
                        embedding=record.get("embedding"),
                        bot_id=record.get("bot_id"),
                        session_id=record.get("session_id"),
                        tenant=record.get("tenant"),
                        source=record.get("source"),
                        external_id=external_id,
                        run_id=record.get("run_id"),
                        created_at=record.get("created_at"),
                        state_metadata=record.get("state_metadata"),
                        metadata=record.get("metadata"),
                        relationships=record.get("relationships"),
                        expires_at=record.get("expires_at"),
                        retention_policy=record.get("retention_policy"),
                        lifecycle_status=record.get("lifecycle_status"),
                        retired_at=record.get("retired_at"),
                        retired_reason=record.get("retired_reason"),
                    )
                    if result["inserted"]:
                        inserted += 1
                    else:
                        updated += 1
            batches += 1
            batch = []

        for index, raw in enumerate(records):
            if not isinstance(raw, Mapping):
                raise TypeError(f"records[{index}] must be a mapping")
            batch.append(
                _normalize_ingest_record(
                    raw,
                    field_map=normalized_field_map,
                    defaults=normalized_defaults,
                    source=source,
                    preserve_raw=preserve_raw,
                    raw_metadata_key=raw_metadata_key,
                    index=index,
                )
            )
            processed += 1
            if len(batch) >= batch_size:
                flush()

        flush()
        return {
            "processed": processed,
            "inserted": inserted,
            "updated": updated,
            "batches": batches,
        }

    def ingest_jsonl(
        self,
        path: str | PathLike[str],
        *,
        field_map: Mapping[str, str] | None = None,
        defaults: Mapping[str, Any] | None = None,
        source: str = "raw",
        batch_size: int = _DEFAULT_INGEST_BATCH_SIZE,
        mode: str = "append",
        preserve_raw: bool = True,
        raw_metadata_key: str = "raw_record",
    ) -> dict[str, int]:
        """Stream-ingest newline-delimited JSON objects from ``path``."""

        def iter_rows() -> Iterable[Mapping[str, Any]]:
            with open(path, encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    row = json.loads(stripped)
                    if not isinstance(row, Mapping):
                        raise ValueError(f"JSONL line {line_number} must be an object")
                    yield row

        return self.ingest_records(
            iter_rows(),
            field_map=field_map,
            defaults=defaults,
            source=source,
            batch_size=batch_size,
            mode=mode,
            preserve_raw=preserve_raw,
            raw_metadata_key=raw_metadata_key,
        )

    def ingest_messages(
        self,
        messages: Iterable[Mapping[str, Any]],
        *,
        session_id: str | None = None,
        tenant: str | None = None,
        bot_id: str | None = None,
        external_id_prefix: str | None = None,
        source: str = "raw",
        batch_size: int = _DEFAULT_INGEST_BATCH_SIZE,
        mode: str = "append",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, int]:
        """Ingest OpenAI-style chat messages as raw context records."""
        default_metadata = dict(metadata or {})

        def iter_rows() -> Iterable[Mapping[str, Any]]:
            for index, message in enumerate(messages):
                if not isinstance(message, Mapping):
                    raise TypeError(f"messages[{index}] must be a mapping")
                content = message.get("content")
                content_type = message.get("content_type")
                if content is None:
                    content = {k: v for k, v in message.items() if k != "role"}
                    content_type = content_type or "application/json"
                if not isinstance(content, (bytes, str)):
                    content = json.dumps(content, sort_keys=True, separators=(",", ":"))
                    content_type = content_type or "application/json"

                row_metadata = dict(default_metadata)
                row_metadata["raw_message"] = dict(message)
                row: dict[str, Any] = {
                    "role": message.get("role", "message"),
                    "content": content,
                    "content_type": content_type,
                    "source": source,
                    "metadata": row_metadata,
                }
                if session_id is not None:
                    row["session_id"] = session_id
                if tenant is not None:
                    row["tenant"] = tenant
                if bot_id is not None:
                    row["bot_id"] = bot_id
                if external_id_prefix is not None:
                    row["external_id"] = f"{external_id_prefix}#message-{index + 1}"
                yield row

        return self.ingest_records(
            iter_rows(),
            batch_size=batch_size,
            mode=mode,
            source=source,
            preserve_raw=False,
        )

    def _auto_embed_batch(self, records: list[dict[str, Any]]) -> None:
        """Embed records without an embedding in one provider call per modality."""
        provider = getattr(self, "_embedding_provider", None)
        if provider is None:
            return
        text_indices = [
            i
            for i, r in enumerate(records)
            if r.get("embedding") is None and isinstance(r.get("content"), str)
        ]
        if text_indices:
            vectors = provider.embed_texts(
                [records[i]["content"] for i in text_indices]
            )
            for i, vec in zip(text_indices, vectors):
                records[i]["embedding"] = vec
        if supports_media(provider):
            media_indices = [
                i
                for i, r in enumerate(records)
                if r.get("embedding") is None
                and isinstance(r.get("content"), (bytes, bytearray))
            ]
            if media_indices:
                items = [
                    (
                        bytes(records[i]["content"]),
                        records[i].get("data_type") or "application/octet-stream",
                    )
                    for i in media_indices
                ]
                vectors = provider.embed_media(items)
                for i, vec in zip(media_indices, vectors):
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
        include_binary: bool = True,
        include_embedding: bool = True,
    ) -> list[dict[str, Any]]:
        provider = getattr(self, "_embedding_provider", None)
        if provider is not None:
            if isinstance(query, str):
                query = provider.embed_texts([query])[0]
            elif isinstance(query, (bytes, bytearray)) and supports_media(provider):
                query = provider.embed_media(
                    [(bytes(query), "application/octet-stream")]
                )[0]
        vector = _coerce_vector(query)
        results = self._inner.search(
            vector,
            limit,
            _json_dumps(filters, "filters"),
            include_expired,
            include_retired,
            include_relationships,
            include_binary,
            include_embedding,
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

    def export_training(
        self,
        output_path: str,
        *,
        task: str = "sft",
        format: str = "jsonl",
        group_by: str = "session_id",
        preference_form: str = "paired",
        filters: dict[str, Any] | None = None,
        dedup_threshold: float | None = None,
        decontaminate_against: list[list[float]] | None = None,
        decontaminate_threshold: float | None = None,
        min_reward: float | None = None,
        version: int | None = None,
        include_expired: bool = False,
        include_retired: bool = False,
        split: dict[str, Any] | None = None,
        emit_stats: bool = False,
    ) -> dict[str, Any]:
        """Curate stored records and export a trainable dataset to JSONL.

        Writes one JSON object per line to ``output_path`` (plus a sibling
        ``<output_path>.manifest.json``) and returns the manifest dict.

        ``task`` selects the shape: ``"sft"`` (ordered messages; pass
        ``min_reward`` for rejection-sampling / Best-of-N), ``"preference"``
        (set ``preference_form`` to ``"paired"`` for DPO-style chosen/rejected,
        ``"unpaired"`` for KTO binary labels, or ``"ranked"`` for N-way judge
        rankings), or ``"rollout"`` (RL groups with per-response reward).

        Curation runs before export: lifecycle-correct filtering (drops
        tombstoned/expired/retired/superseded/contradicted), optional
        ``min_reward`` thresholding, semantic ``dedup_threshold`` (cosine), and
        ``decontaminate_against`` a holdout-embedding set. Reward / preference
        signals are read from each record's ``metadata`` (``reward``,
        ``reward_source``, ``group_id``, ``label``, ``rank``). Pass ``version``
        to pin the export to a dataset version for reproducibility.

        Pass         ``split={"eval_fraction": 0.1, "by": "session_id", "seed": 42}``
        to write group-disjoint, reproducible ``<path>.train.jsonl`` and
        ``<path>.eval.jsonl`` outputs (each with its own manifest) instead of a
        single file; the returned manifest is the train side.

        Pass ``emit_stats=True`` to also write a ``<output_path>.stats.json``
        dataset report (example/role/source/tenant counts, token-length
        distribution, per-group counts, curation exclusions by reason, and
        reward distribution).
        """
        if format != "jsonl":
            raise ValueError("export format currently supports only 'jsonl'")
        split_eval_fraction = None
        split_by = None
        split_seed = None
        if split is not None:
            if "eval_fraction" not in split:
                raise ValueError("split requires 'eval_fraction'")
            split_eval_fraction = float(split["eval_fraction"])
            split_by = split.get("by")
            split_seed = int(split["seed"]) if split.get("seed") is not None else None
        manifest_json = self._inner.export_training(
            output_path,
            task,
            group_by,
            preference_form,
            _json_dumps(filters, "filters"),
            dedup_threshold,
            decontaminate_against,
            decontaminate_threshold,
            min_reward,
            int(version) if version is not None else None,
            include_expired,
            include_retired,
            split_eval_fraction,
            split_by,
            split_seed,
            emit_stats,
        )
        return json.loads(manifest_json)

    def evaluate(
        self,
        queries: Iterable[Mapping[str, Any]],
        *,
        query_set_id: str = "eval",
        k: int = 10,
        mode: str = "vector",
        filters: dict[str, Any] | None = None,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> dict[str, Any]:
        """Evaluate retrieval quality against a labeled query set.

        Each query is a mapping with ``query_id``, an optional ``text`` and/or
        ``vector`` channel, and ``relevant`` — a list of ``{"external_id": ...,
        "grade": <float, default 1.0>}`` labels. ``mode`` selects ``"vector"``
        (search) or ``"hybrid"`` (retrieve).

        Returns a report dict with ``aggregate`` metrics (``recall``,
        ``precision``, ``mrr``, ``ndcg``, ``hit_rate``), a ``per_query``
        breakdown, and a manifest (``query_set_id``, ``version``, ``k``,
        ``mode``, ``distance_metric``) for reproducibility.
        """
        query_set = json.dumps(
            {"id": query_set_id, "queries": [dict(query) for query in queries]}
        )
        report_json = self._inner.evaluate(
            query_set,
            k,
            mode,
            _json_dumps(filters, "filters"),
            include_expired,
            include_retired,
        )
        return json.loads(report_json)

    def evaluate_versions(
        self,
        queries: Iterable[Mapping[str, Any]],
        baseline_version: int,
        candidate_version: int,
        *,
        query_set_id: str = "eval",
        k: int = 10,
        mode: str = "vector",
        filters: dict[str, Any] | None = None,
        include_expired: bool = False,
        include_retired: bool = False,
    ) -> dict[str, Any]:
        """A/B the same query set across two dataset versions.

        Runs :meth:`evaluate` at ``baseline_version`` and ``candidate_version``
        (via time-travel checkout) and returns ``{"baseline", "candidate",
        "deltas"}`` where ``deltas`` is ``candidate - baseline`` per metric. The
        context is restored to its current version before returning.
        """
        query_set = json.dumps(
            {"id": query_set_id, "queries": [dict(query) for query in queries]}
        )
        report_json = self._inner.evaluate_versions(
            query_set,
            int(baseline_version),
            int(candidate_version),
            k,
            mode,
            _json_dumps(filters, "filters"),
            include_expired,
            include_retired,
        )
        return json.loads(report_json)

    def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: dict[str, Any] | None = None,
        *,
        include_expired: bool = False,
        include_retired: bool = False,
        include_binary: bool = True,
        include_embedding: bool = True,
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
            include_binary: Load ``binary_payload``. Set ``False`` to avoid
                materializing large media bytes for metadata/listing queries;
                fetch a record's bytes on demand with :meth:`get_blob`.
            include_embedding: Load the embedding vector. Set ``False`` to skip
                it when only metadata is needed.

        Returns:
            List of entry dicts with keys: id, run_id, role, content_type,
            text, binary, embedding, created_at, metadata, state_metadata, and
            lifecycle metadata. Omitted payloads come back as ``None``.
        """
        results = self._inner.list(
            limit,
            offset,
            _json_dumps(filters, "filters"),
            include_expired,
            include_retired,
            include_binary,
            include_embedding,
        )
        return [_normalize_record(item) for item in results]

    def get_blob(self, id: str) -> bytes | None:
        """Fetch a single record's ``binary_payload`` bytes on demand.

        Pairs with ``list(..., include_binary=False)`` / ``search(...,
        include_binary=False)``: query metadata cheaply, then pull the media for
        the records you actually need. Returns ``None`` if the record or its
        binary payload is absent.
        """
        return self._inner.get_blob(id)

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

    def fetch_payload(self, id: str) -> bytes | None:
        """Resolve a record's external ``payload_uri`` to its bytes on demand.

        Reads the referenced object (``gs://``/``s3://``/local) using the
        context's configured ``storage_options``, so large media can live in
        object storage while ``get``/``list``/``search`` return only the
        reference. Returns ``None`` if no record with ``id`` exists; raises if
        the record carries no external payload reference.
        """
        return self._inner.fetch_payload(id)

    def put_payload(self, uri: str, data: bytes) -> int:
        """Offload ``data`` to an object at ``uri`` using the context's
        ``storage_options``; returns the number of bytes written.

        Pair with ``add(..., payload_uri=uri)`` to store large media outside the
        dataset and reference it from a record.
        """
        return int(self._inner.put_payload(uri, data))

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
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
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
                payload_uri=payload_uri,
                payload_size=payload_size,
                payload_checksum=payload_checksum,
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

    async def fetch_payload(self, id: str) -> bytes | None:
        """Asynchronously resolve a record's external ``payload_uri`` to bytes."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._sync.fetch_payload(id))

    async def put_payload(self, uri: str, data: bytes) -> int:
        """Asynchronously offload ``data`` to an object at ``uri``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._sync.put_payload(uri, data)
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
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
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
                payload_uri=payload_uri,
                payload_size=payload_size,
                payload_checksum=payload_checksum,
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
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
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
                payload_uri=payload_uri,
                payload_size=payload_size,
                payload_checksum=payload_checksum,
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
        payload_uri: str | None = None,
        payload_size: int | None = None,
        payload_checksum: str | None = None,
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
                payload_uri=payload_uri,
                payload_size=payload_size,
                payload_checksum=payload_checksum,
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

    async def fetch_payload(self, id: str) -> bytes:
        """Resolve a record's external ``payload_uri`` to its bytes via the
        remote server, which reads from object storage using the context's
        ``storage_options``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync.fetch_payload(id),
        )

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


# ---------------------------------------------------------------------------
# Rollout stores
# ---------------------------------------------------------------------------

# Fields that carry raw artifact bytes; the native layer marshals records as
# JSON with these base64-encoded, so the wrapper encodes on the way in.
_ROLLOUT_BLOB_FIELD = "binary_payload"


def _rollout_record_to_native(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one rollout record dict for the JSON FFI boundary.

    Accepts ``binary_payload`` as raw ``bytes``/``bytearray`` and base64-encodes
    it (the DTO's JSON representation). If ``id`` is missing/empty, a UUID4 is
    generated. Everything else is left untouched.
    """
    out = dict(record)
    # `id` is required by the store; if the caller omits it (or passes an empty
    # value) generate a UUID4 so users never have to mint ids by hand.
    if not out.get("id"):
        import uuid

        out["id"] = str(uuid.uuid4())
    blob = out.get(_ROLLOUT_BLOB_FIELD)
    if isinstance(blob, (bytes, bytearray)):
        import base64

        out[_ROLLOUT_BLOB_FIELD] = base64.b64encode(bytes(blob)).decode("ascii")
    return out


def _rollout_records_to_json(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> str:
    # `records` is either a single mapping or an iterable of mappings. Because a
    # Mapping is itself Iterable, static narrowing of the union is ambiguous, so
    # branch on the concrete runtime type via a local typed as Any.
    raw: Any = records
    if isinstance(raw, Mapping):
        payload = [_rollout_record_to_native(raw)]
    else:
        payload = [_rollout_record_to_native(r) for r in raw]
    if not payload:
        raise ValueError("records must not be empty")
    return json.dumps(payload)


def _rollout_record_from_json(record: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a rollout record dict coming back from the native layer.

    ``binary_payload``, when present, arrives base64-encoded; decode it to raw
    ``bytes`` so callers get bytes back from ``get``/``list`` if it was
    materialized.
    """
    out = dict(record)
    blob = out.get(_ROLLOUT_BLOB_FIELD)
    if isinstance(blob, str):
        import base64

        out[_ROLLOUT_BLOB_FIELD] = base64.b64decode(blob)
    return out


class RolloutStore:
    """Synchronous store for RL rollout trajectories.

    Open an embedded dataset with :meth:`open`, or talk to a remote
    ``lance-context-server`` with :meth:`connect` / :meth:`connect_or_create`.

    Records are plain dicts matching the rollout schema (``id``, ``rollout_id``,
    ``reward``, ``binary_payload`` as raw ``bytes``, ...). Only ``id`` and
    ``rollout_id`` are required; every other field is optional.
    """

    def __init__(self, sync_store: _RolloutStore) -> None:
        self._sync = sync_store

    @classmethod
    def open(
        cls,
        uri: str,
        *,
        storage_options: Mapping[str, str] | None = None,
    ) -> "RolloutStore":
        """Open (or create) an embedded rollout dataset at ``uri``."""
        opts = dict(storage_options) if storage_options else None
        return cls(_RolloutStore.open(uri, opts))

    @classmethod
    def connect(cls, base_url: str, name: str) -> "RolloutStore":
        """Connect to an existing rollout store on a remote server."""
        return cls(_RolloutStore.connect(base_url, name))

    @classmethod
    def connect_or_create(
        cls,
        base_url: str,
        name: str,
        *,
        storage_options: Mapping[str, str] | None = None,
    ) -> "RolloutStore":
        """Connect to a remote rollout store, creating it if absent."""
        opts = dict(storage_options) if storage_options else None
        return cls(_RolloutStore.connect_or_create(base_url, name, opts))

    def version(self) -> int:
        """Return the current base-table version."""
        return self._sync.version()

    def add(
        self,
        records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Append one record (dict) or many (iterable of dicts)."""
        return self._sync.add(_rollout_records_to_json(records))

    def add_one(self, **fields: Any) -> dict[str, Any]:
        """Append a single record given as keyword arguments."""
        return self.add(fields)

    def flush(self) -> None:
        """Make previously added rows visible to subsequent reads.

        ``add`` is durable on return but *not* visible on return: rows land in
        a write-ahead log and only reach a readable generation when the
        memtable is sealed. A deployed server does this on a timer; an embedded
        store has no such timer, so a write-then-read sequence returns nothing
        until you call this.

        No-op for stores connected to a remote server.
        """
        self._sync.flush()

    def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """List rollout rows matching exact field filters.

        Supported filter keys are ``rollout_id``, ``problem_id``,
        ``policy_version``, ``role``, ``include_in_training``, and
        ``artifact_type``. Artifact bytes remain projected out.
        """
        raw = self._sync.list(limit, offset, _json_dumps(filters, "filters"))
        return [_rollout_record_from_json(r) for r in json.loads(raw)]

    def get_trajectory(self, rollout_id: str) -> list[dict[str, Any]]:
        """Return one complete trajectory ordered by sequence and row id."""
        raw = self._sync.get_trajectory(rollout_id)
        return [_rollout_record_from_json(r) for r in json.loads(raw)]

    def get(self, id: str) -> dict[str, Any] | None:
        """Fetch a single rollout row by id, or ``None``."""
        raw = self._sync.get(id)
        if raw is None:
            return None
        return _rollout_record_from_json(json.loads(raw))

    def get_blob(self, id: str) -> bytes | None:
        """Fetch a single artifact row's inline bytes on demand, or ``None``."""
        return self._sync.get_blob(id)

    def checkout(self, version: int) -> None:
        """Checkout a base-table version (base-table time travel only)."""
        self._sync.checkout(version)

    def __repr__(self) -> str:
        return f"RolloutStore(version={self._sync.version()})"


class AsyncRolloutStore:
    """Async wrapper around a rollout store, for use from asyncio code.

    Mirrors :class:`RolloutStore` but runs the blocking calls in an executor so
    they can be awaited without blocking the event loop. Every method matches
    its sync counterpart. This is the interface RL generation workers and the
    learner use when talking to a remote ``lance-context-server`` over HTTP.
    """

    def __init__(self, sync_store: _RolloutStore) -> None:
        self._sync = sync_store

    @classmethod
    async def connect(cls, base_url: str, name: str) -> "AsyncRolloutStore":
        """Connect to an existing remote rollout store."""
        loop = asyncio.get_running_loop()
        sync_store = await loop.run_in_executor(
            None, lambda: _RolloutStore.connect(base_url, name)
        )
        return cls(sync_store)

    @classmethod
    async def connect_or_create(
        cls,
        base_url: str,
        name: str,
        *,
        storage_options: Mapping[str, str] | None = None,
    ) -> "AsyncRolloutStore":
        """Connect to a remote rollout store, creating it if absent."""
        opts = dict(storage_options) if storage_options else None
        loop = asyncio.get_running_loop()
        sync_store = await loop.run_in_executor(
            None, lambda: _RolloutStore.connect_or_create(base_url, name, opts)
        )
        return cls(sync_store)

    def version(self) -> int:
        """Return the current version (cached, sync)."""
        return self._sync.version()

    async def add(
        self,
        records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Append one record (dict) or many (iterable of dicts)."""
        payload = _rollout_records_to_json(records)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._sync.add(payload))

    async def add_one(self, **fields: Any) -> dict[str, Any]:
        """Append a single record given as keyword arguments."""
        return await self.add(fields)

    async def flush(self) -> None:
        """Make previously added rows visible to subsequent reads.

        For an embedded store this seals the local memtable. For a store
        connected to a remote server this is a no-op: the server flushes on its
        own interval (``ROLLOUT_FLUSH_INTERVAL_SECS``), which bounds how long a
        just-written row stays invisible.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync.flush)

    async def list(
        self,
        limit: int | None = None,
        offset: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """List rollout rows matching exact field filters."""
        filters_json = _json_dumps(filters, "filters")
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None, lambda: self._sync.list(limit, offset, filters_json)
        )
        return [_rollout_record_from_json(r) for r in json.loads(raw)]

    async def get_trajectory(self, rollout_id: str) -> list[dict[str, Any]]:
        """Return one complete trajectory ordered by sequence and row id."""
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None, lambda: self._sync.get_trajectory(rollout_id)
        )
        return [_rollout_record_from_json(r) for r in json.loads(raw)]

    async def get(self, id: str) -> dict[str, Any] | None:
        """Fetch a single rollout row by id, or ``None``."""
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, lambda: self._sync.get(id))
        if raw is None:
            return None
        return _rollout_record_from_json(json.loads(raw))

    async def get_blob(self, id: str) -> bytes | None:
        """Fetch a single artifact row's bytes on demand, or ``None``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._sync.get_blob(id))

    async def checkout(self, version: int) -> None:
        """Checkout a base-table version (base-table time travel only)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._sync.checkout(version))

    def __repr__(self) -> str:
        return f"AsyncRolloutStore(version={self._sync.version()})"


class DatagenStore:
    """Synchronous append-only datagen delta-log store.

    Wraps the native store: events are appended as plain dicts and item state is
    recovered by folding an item's events. Open an embedded log with :meth:`open`.

    Event dicts carry the datagen event schema (``event_id``, ``item_id``,
    ``root_item_id``, ``event_type``, ``item_seq``, ...); values use the tagged
    ``{"kind": ..., "value": ...}`` shape. Folded items and failures come back as
    plain dicts.
    """

    def __init__(self, sync_store: _DatagenStore) -> None:
        self._sync = sync_store

    @classmethod
    def open(
        cls,
        uri: str,
        *,
        storage_options: Mapping[str, str] | None = None,
        shard_id: str | None = None,
    ) -> "DatagenStore":
        """Open (or create) an embedded datagen log at ``uri``.

        ``shard_id`` gives this writer a stable identity for multi-writer fencing.
        """
        opts = dict(storage_options) if storage_options else None
        return cls(_DatagenStore.open(uri, opts, shard_id))

    @classmethod
    def connect(cls, base_url: str, name: str) -> "DatagenStore":
        """Connect to an existing datagen store on a remote server."""
        return cls(_DatagenStore.connect(base_url, name))

    @classmethod
    def connect_or_create(
        cls,
        base_url: str,
        name: str,
        *,
        storage_options: Mapping[str, str] | None = None,
    ) -> "DatagenStore":
        """Connect to a remote datagen store, creating it if absent."""
        opts = dict(storage_options) if storage_options else None
        return cls(_DatagenStore.connect_or_create(base_url, name, opts))

    def version(self) -> int:
        """Return the current store version (base dataset version)."""
        return self._sync.version()

    def append_checkpoint(self, events: Iterable[Mapping[str, Any]]) -> int:
        """Append one completed step boundary atomically.

        ``events`` share one item/checkpoint/writer attempt and contain exactly
        one ``STEP_COMPLETED``. Returns the new store version.
        """
        return self._sync.append_checkpoint(list(events))

    def append(self, events: Iterable[Mapping[str, Any]]) -> int:
        """Append raw events as one MemWAL generation. Returns the new store version."""
        return self._sync.append(list(events))

    def fold_item(
        self, item_id: str, *, load_blobs: bool = False
    ) -> dict[str, Any] | None:
        """Fold an item's events into its latest state, or ``None`` if never started.

        ``load_blobs`` materializes blob-field bytes inline; the default leaves them
        lazy, to be resolved through :meth:`get_blob` / :meth:`load_blob`.
        """
        return self._sync.fold_item(item_id, load_blobs)

    def overview(self) -> dict[str, Any]:
        """Aggregate the whole store into a run overview.

        Returns root-item counts by status (``items`` / ``running`` / ``completed`` /
        ``filtered``), ``completed_steps`` per step name, and a failure roll-up: total
        ``failures``, ``failures_by_error_type``, and ``failures_by_run`` — one bucket
        per ``run_id``, each with its own error-type counts and a small sample of
        failing root item ids to drill into.
        """
        return self._sync.overview()

    def root_item_statuses(self, root_item_ids: Sequence[str]) -> dict[str, str]:
        """Classify each root item id by folded lifecycle status.

        Missing ids (never started) are absent from the returned dict.
        """
        return self._sync.root_item_statuses(list(root_item_ids))

    def item_failures(self, item_id: str) -> list[dict[str, Any]]:
        """All failure records for an item (the failure lens), oldest first."""
        return self._sync.item_failures(item_id)

    def get_blob(self, event_id: str) -> bytes | None:
        """Materialize one ``FIELD_*`` event's blob bytes by event id, or ``None``."""
        return self._sync.get_blob(event_id)

    def load_blob(self, folded: Mapping[str, Any], field_name: str) -> bytes | None:
        """Resolve a folded item's blob field by name to its bytes, or ``None``.

        ``folded`` is a dict from :meth:`fold_item`; its ``blob_event_ids`` map points
        each blob field to the event id that carries the bytes. Returns ``None`` when
        ``field_name`` is not a blob field of ``folded`` (never written, or written with
        a non-blob value). Works for embedded and remote stores.
        """
        event_id = folded.get("blob_event_ids", {}).get(field_name)
        if event_id is None:
            return None
        return self._sync.get_blob(event_id)

    def item_tree(self, root_item_id: str) -> dict[str, Any]:
        """Assemble the inspection tree rooted at ``root_item_id``.

        Every projected descendant is folded to its latest state and linked
        parent->child. Returns ``{"roots": [item_id, ...], "nodes": {item_id: {"item":
        folded, "children": [item_id, ...]}}}``. Works for embedded and remote stores.
        """
        return self._sync.item_tree(root_item_id)

    def open_stream(
        self,
        item_id: str,
        *,
        run_id: str,
        writer_epoch: str,
        parent_item_id: str | None = None,
        query_tags: Any = None,
    ) -> "DatagenStreamWriter":
        """Open a fresh stream: persist ITEM_CREATED, return a writer to continue.

        ``run_id``/``writer_epoch`` stamp every event the writer emits; ``query_tags``
        is captured onto ITEM_CREATED. The writer is client-side state — hand its
        emitted events back to :meth:`append`/:meth:`append_checkpoint`. Works for
        embedded and remote stores.
        """
        return DatagenStreamWriter(
            self._sync.open_stream(
                item_id, run_id, writer_epoch, parent_item_id, query_tags
            )
        )

    def resume_stream(
        self, item_id: str, *, run_id: str, writer_epoch: str
    ) -> "DatagenStreamWriter | None":
        """Rebuild a writer to resume an already-started item, or ``None`` if absent.

        Pure — emits nothing; the returned writer continues the item's ``item_seq`` and
        bumps ``attempt``. Works for embedded and remote stores.
        """
        writer = self._sync.resume_stream(item_id, run_id, writer_epoch)
        return DatagenStreamWriter(writer) if writer is not None else None

    def __repr__(self) -> str:
        return f"DatagenStore(version={self._sync.version()})"


class DatagenItemId:
    """A structured datagen item id.

    Root ids come from the executor's source key; sub-item ids extend a parent with one
    fan-out segment (``5/expand:0``). ``str(id)`` is the stored path form. Pure and
    client-side — composing an id does no I/O.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @classmethod
    def from_source_key(cls, key: str) -> "DatagenItemId":
        """A root id for the executor's source key."""
        return cls(_DatagenItemId.from_source_key(key))

    @classmethod
    def parse(cls, path: str) -> "DatagenItemId":
        """Parse a stored path form (``5`` or ``5/expand:0``)."""
        return cls(_DatagenItemId.parse(path))

    def child(self, origin_step: str, branch_idx: int) -> "DatagenItemId":
        """The sub-item id forked at ``origin_step`` branch ``branch_idx``."""
        return DatagenItemId(self._inner.child(origin_step, branch_idx))

    def parent(self) -> "DatagenItemId | None":
        """The parent id, or ``None`` for a root."""
        inner = self._inner.parent()
        return DatagenItemId(inner) if inner is not None else None

    def root(self) -> "DatagenItemId":
        """The root this id descends from (itself when already a root)."""
        return DatagenItemId(self._inner.root())

    @property
    def origin_step(self) -> str | None:
        """The fan-out step this id was forked at, or ``None`` for a root."""
        return self._inner.origin_step

    @property
    def branch_idx(self) -> int | None:
        """The branch index within ``origin_step``, or ``None`` for a root."""
        return self._inner.branch_idx

    @property
    def is_root(self) -> bool:
        """Whether this id has no fan-out segment."""
        return self._inner.is_root

    def __str__(self) -> str:
        return str(self._inner)

    def __repr__(self) -> str:
        return f"DatagenItemId({str(self._inner)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatagenItemId):
            return NotImplemented
        return bool(self._inner == other._inner)

    def __hash__(self) -> int:
        return hash(str(self._inner))


class DatagenStreamWriter:
    """A per-stream write handle over the native writer.

    Owns the bookkeeping columns callers should never touch (``item_seq``, ``attempt``,
    ``checkpoint_id``, ``event_id``), stamping them onto every event it emits. Each
    method returns the event dict(s) to persist — hand them to
    :meth:`DatagenStore.append` or :meth:`DatagenStore.append_checkpoint`. Pure and
    client-side, identical for embedded and remote stores.
    """

    def __init__(self, inner: _DatagenStreamWriter) -> None:
        self._inner = inner

    @property
    def item_id(self) -> str:
        """The item this writer streams to."""
        return self._inner.item_id

    @property
    def attempt(self) -> int:
        """Attempt stamped onto events (0 fresh, ``last_attempt + 1`` on resume)."""
        return self._inner.attempt

    def step_started(self, position: Mapping[str, Any]) -> dict[str, Any]:
        """Emit STEP_STARTED for a driver frame (Sequence/Loop). Returns one event."""
        return self._inner.step_started(dict(position))

    def step_completed(
        self, position: Mapping[str, Any], fields: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Emit a checkpoint boundary: the step's field writes plus its STEP_COMPLETED.

        All share one ``checkpoint_id``; returns the list of event dicts (the atomic
        unit :meth:`DatagenStore.append_checkpoint` persists).
        """
        return self._inner.step_completed(
            dict(position), [dict(field) for field in fields]
        )

    def item_terminal(self, terminal: str) -> dict[str, Any]:
        """Emit TERMINAL; ``terminal`` is ``"completed"`` or ``"filtered"``."""
        return self._inner.item_terminal(terminal)

    def item_failed(
        self,
        position: Mapping[str, Any],
        error_type: str,
        *,
        error_dump: str | None = None,
        traceback: str | None = None,
    ) -> dict[str, Any]:
        """Emit FAILED at a step position (failure lens; does not terminate)."""
        return self._inner.item_failed(
            dict(position), error_type, error_dump, traceback
        )

    def __repr__(self) -> str:
        return (
            f"DatagenStreamWriter(item_id={self._inner.item_id!r}, "
            f"attempt={self._inner.attempt})"
        )


class GenericStore:
    """Synchronous store over a schema you declare.

    Unlike the fixed-schema stores, the columns come from you: pass a ``schema``
    mapping column names to types at creation, then read and write plain dicts.
    An ``id`` column is required — it is the primary key and the key rows are
    deduplicated by.

    Types may be written in full form or shorthand::

        store = GenericStore.open(uri, schema={
            "id":     {"type": "string", "nullable": False},
            "user":   "string",
            "score":  "float32",
            "tags":   {"type": "list", "item": {"type": "string"}},
            "vec":    {"type": "vector", "dim": 768, "metric": "cosine"},
            "video":  {"type": "binary", "blob": True},
        })

    ``blob: True`` does not change where the bytes live — they are stored inline
    like any other column — it excludes the column from bulk reads. :meth:`list`
    never materializes a blob column; fetch one per row with
    ``get(id, columns=["video"])``. That is what keeps a listing cheap on a
    store holding large payloads.
    """

    def __init__(self, sync_store: _GenericStore) -> None:
        self._sync = sync_store

    @classmethod
    def open(
        cls,
        uri: str,
        schema: Mapping[str, Any],
        *,
        storage_options: Mapping[str, str] | None = None,
        seal_on_add: bool = False,
    ) -> "GenericStore":
        """Open (or create) an embedded store at ``uri`` with ``schema``.

        ``seal_on_add=True`` makes rows readable as soon as :meth:`add` returns,
        at the cost of serializing concurrent appends. The default defers the
        seal — durable on return, readable after :meth:`flush`.
        """
        opts = dict(storage_options) if storage_options else None
        return cls(_GenericStore.open(uri, dict(schema), opts, seal_on_add))

    @classmethod
    def open_existing(
        cls,
        uri: str,
        *,
        storage_options: Mapping[str, str] | None = None,
        seal_on_add: bool = False,
    ) -> "GenericStore":
        """Open an existing embedded store, reading its schema from the dataset."""
        opts = dict(storage_options) if storage_options else None
        return cls(_GenericStore.open_existing(uri, opts, seal_on_add))

    @classmethod
    def connect(cls, base_url: str, name: str) -> "GenericStore":
        """Connect to an existing store on a remote server."""
        return cls(_GenericStore.connect(base_url, name))

    @classmethod
    def connect_or_create(
        cls,
        base_url: str,
        name: str,
        schema: Mapping[str, Any],
        *,
        storage_options: Mapping[str, str] | None = None,
        seal_on_add: bool = False,
    ) -> "GenericStore":
        """Connect to a remote store, creating it with ``schema`` if absent."""
        opts = dict(storage_options) if storage_options else None
        return cls(
            _GenericStore.connect_or_create(
                base_url, name, dict(schema), opts, seal_on_add
            )
        )

    def schema(self) -> dict[str, Any]:
        """The store's schema, as declared at creation."""
        return self._sync.schema()

    def version(self) -> int:
        """Base dataset version."""
        return self._sync.version()

    def add(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Append rows.

        Omitted nullable columns are stored as null; a column the schema does
        not declare raises, rather than being silently dropped.
        """
        return self._sync.add(list(rows))

    def list(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read rows, newest write wins per ``id``. Blob columns are excluded.

        ``filter`` is a SQL predicate over your own columns, e.g.
        ``"score > 0.5 AND user = 'u1'"``.
        """
        return self._sync.list(limit, offset, filter)

    def get(
        self, id: str, *, columns: Sequence[str] | None = None
    ) -> dict[str, Any] | None:
        """Fetch one row by ``id``, or ``None``.

        ``columns`` selects what to read; omit it to read everything except blob
        columns. Name a blob column explicitly to fetch its bytes, which come
        back base64-encoded.
        """
        return self._sync.get(id, list(columns) if columns is not None else None)

    def flush(self) -> None:
        """Seal buffered rows so they become readable.

        Unnecessary when the store was opened with ``seal_on_add=True``.
        """
        self._sync.flush()

    def cleanup_wal(self) -> int:
        """Merge pending WAL generations into the base table (embedded only).

        Returns how many were reclaimed. Without this, generations accumulate
        and every read unions all of them.
        """
        return self._sync.cleanup_wal()

    def __repr__(self) -> str:
        return f"GenericStore(version={self._sync.version()})"
