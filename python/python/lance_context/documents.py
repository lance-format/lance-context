"""Embedded JSON documents with automatic field selection and reconstruction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike

from ._internal import (
    DocumentStore as _DocumentStore,  # pyright: ignore[reportMissingImports]
)


class DocumentStore:
    """Store JSON documents without declaring their shredded paths.

    Use one writer per dataset. This embedded adapter commits directly to Lance;
    it does not use a WAL or provide a remote HTTP client. IDs returned by ``add``
    identify records independently of any ``id`` key inside the JSON document.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    @classmethod
    def create(
        cls,
        uri: str | PathLike[str],
        *,
        policy: Mapping[str, Any] | None = None,
        storage_options: Mapping[str, str] | None = None,
    ) -> DocumentStore:
        """Create a new store with automatic shredding enabled.

        ``policy`` overrides resource/selection limits such as ``max_columns``
        and ``min_frequency``. It contains no extraction paths. Reopening reads
        the persisted policy, so callers do not redeclare it.
        """
        return cls(
            _DocumentStore.create(
                str(uri),
                dict(policy) if policy is not None else None,
                dict(storage_options) if storage_options is not None else None,
            )
        )

    @classmethod
    def open(
        cls,
        uri: str | PathLike[str],
        *,
        storage_options: Mapping[str, str] | None = None,
    ) -> DocumentStore:
        """Open an existing store with its persisted layout and policy."""
        return cls(
            _DocumentStore.open(
                str(uri),
                dict(storage_options) if storage_options is not None else None,
            )
        )

    def add(self, documents: Sequence[Any]) -> list[str]:
        """Commit a batch of JSON values and return generated record IDs."""
        if isinstance(documents, (str, bytes, Mapping)):
            raise TypeError("add expects a sequence of JSON documents")
        return self._backend.add(list(documents))

    def add_iter(
        self,
        documents: Iterable[Any],
        *,
        max_batch_bytes: int = 4 * 1024 * 1024,
        max_batch_rows: int = 1024,
    ) -> Iterator[list[str]]:
        """Consume documents incrementally, yielding IDs after each batch commits.

        Each yielded batch is atomic; the entire iterable is not. Earlier batches
        remain committed if a later document or source raises. Iteration advances
        only as batches are requested, with at most one lookahead document.
        The byte target is a conservative estimate, not a process memory cap.
        A single oversized document is accepted as its own batch.
        """
        if isinstance(documents, (str, bytes, Mapping)):
            raise TypeError("add_iter expects an iterable of JSON documents")
        if any(
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
            for limit in (max_batch_bytes, max_batch_rows)
        ):
            raise ValueError("batch byte and row limits must be positive integers")
        batch: list[Any] = []
        size = 0
        for document in documents:
            document_size = _document_size(document)
            if batch and size + document_size > max_batch_bytes:
                yield self.add(batch)
                batch = []
                size = 0
            batch.append(document)
            size += document_size
            if len(batch) >= max_batch_rows or size >= max_batch_bytes:
                yield self.add(batch)
                batch = []
                size = 0
        if batch:
            yield self.add(batch)

    def get(self, record_id: str) -> Any:
        """Reconstruct a document; raise KeyError if the record is missing."""
        return self._backend.get(record_id)

    def get_path(self, record_id: str, path: Sequence[str]) -> Any:
        """Read literal object-key components, using column projection when possible.

        Missing records/paths raise KeyError; explicit JSON null returns None.
        For example, ``["metadata", "source"]`` addresses a nested field, while
        ``["metadata.source"]`` addresses a literal key containing a dot.
        """
        return self._backend.get_path(record_id, _path(path))

    def filter_eq(
        self, path: Sequence[str], value: Any, *, limit: int = 100
    ) -> list[str]:
        """Find IDs by logical equality, including historical residual values."""
        return self._backend.filter_eq(_path(path), value, limit)

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Search JSON string values and return IDs and scores.

        The first search after layout growth backfills historical values and
        builds indexes. Call prepare_text_search ahead of latency-sensitive reads.
        Indexed matches rank first. If fewer than ``limit`` are found, residual
        strings (including type changes, arrays, and fields outside the column
        budget) are scanned with the same tokenizer. Residual-only matches have
        score 0 and no relevance ordering. This fallback may read large residuals;
        it stores no duplicate text.
        """
        return self._backend.search(query, limit)

    def prepare_text_search(self) -> None:
        """Prepare historical text and indexes before serving text searches."""
        self._backend.prepare_text_search()

    def prepare_point_reads(self) -> None:
        """Build/extend the ID index, including on stores from earlier releases.

        Writes maintain it automatically after committing data. An automatic
        maintenance failure is logged and leaves correct scan-based reads;
        this explicit method reports failures and can be retried safely.
        """
        self._backend.prepare_point_reads()

    def layout(self) -> dict[str, Any]:
        """Inspect the automatically selected paths and scalar types."""
        return self._backend.layout()

    def version(self) -> int:
        return self._backend.version()

    def checkout(self, version: int) -> None:
        """Pin a read-only snapshot, including its original shredding layout."""
        self._backend.checkout(version)

    def refresh(self) -> None:
        """Refresh the layout and data to the latest committed version."""
        self._backend.refresh()


def _path(path: Sequence[str]) -> list[str]:
    if isinstance(path, str):
        raise TypeError("path must be a sequence of literal object keys")
    return list(path)


def _document_size(value: Any, depth: int = 0) -> int:
    # Upper estimate for UTF-8/escaping without materializing another JSON string.
    if value is None or isinstance(value, bool):
        return 32
    if isinstance(value, str):
        return 32 + 24 * len(value)
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**64 - 1:
            raise ValueError("JSON integers must fit signed or unsigned 64 bits")
        return 32
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return 32
    if depth >= 127:
        raise ValueError("document exceeds 127 JSON nesting levels")
    if isinstance(value, dict):
        size = 32
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            size += 32 + 24 * len(key) + _document_size(child, depth + 1)
        return size
    if isinstance(value, (list, tuple)):
        return 32 + sum(_document_size(child, depth + 1) for child in value)
    raise TypeError("expected a JSON-compatible value")
