from __future__ import annotations

from ._internal import Context
from ._internal import version as _version

__all__ = ["Context", "__version__"]

__version__ = _version()
