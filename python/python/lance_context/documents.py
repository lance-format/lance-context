"""Embedded JSON documents with automatic field selection and reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike

from ._internal import DocumentStore as _DocumentStore  # pyright: ignore[reportMissingImports]


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
        """Search automatically selected string fields and return IDs and scores.

        The first search after layout growth backfills historical values and
        builds indexes. Call prepare_text_search ahead of latency-sensitive reads.
        Arrays and paths outside the shredding budget are not full-text indexed.
        """
        return self._backend.search(query, limit)

    def prepare_text_search(self) -> None:
        """Prepare historical text and indexes before serving text searches."""
        self._backend.prepare_text_search()

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
