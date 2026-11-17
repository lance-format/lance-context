"""Automatic shredding using the canonical Arrow Variant representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._internal import (
    VariantDocumentStore as _VariantDocumentStore,  # pyright: ignore[reportMissingImports]
)
from .documents import DocumentStore

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike


class VariantDocumentStore(DocumentStore):
    """Store documents in standard Arrow Variant arrays with automatic layouts.

    This is distinct from the custom JSON-residual ``DocumentStore`` baseline.
    Arrow kernels shred and reconstruct values. New selected paths/types trigger
    an atomic rewrite of existing data; unchanged layouts append directly.
    Use one writer per dataset. This adapter does not provide a WAL/HTTP client.
    """

    @classmethod
    def create(
        cls,
        uri: str | PathLike[str],
        *,
        policy: Mapping[str, Any] | None = None,
        storage_options: Mapping[str, str] | None = None,
    ) -> VariantDocumentStore:
        """Create with selection limits; no caller-specified paths are required."""
        return cls(
            _VariantDocumentStore.create(
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
    ) -> VariantDocumentStore:
        """Open using the persisted standard Variant layout and policy."""
        return cls(
            _VariantDocumentStore.open(
                str(uri),
                dict(storage_options) if storage_options is not None else None,
            )
        )

    def prepare_point_reads(self) -> None:
        """Build/extend the ID index; writes attempt this after committing data."""
        self._backend.prepare_point_reads()

    def prepare_text_search(self) -> None:
        """Index standard typed text children directly, without copying text."""
        self._backend.prepare_text_search()

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return indexed hits first, then residual string hits with score zero.

        Residuals use matching OR tokenization but have no relevance ordering.
        Queries with fewer than ``limit`` indexed hits may scan all residuals.
        """
        return self._backend.search(query, limit)
