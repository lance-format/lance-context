from __future__ import annotations

from ._internal import Context  # pyright: ignore[reportMissingImports]
from ._internal import version as _version  # pyright: ignore[reportMissingImports]

__all__ = ["Context", "__version__"]

__version__ = _version()
