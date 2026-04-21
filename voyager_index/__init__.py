"""
Compatibility shim for the renamed package.

`voyager_index` was renamed to `colsearch` in 0.1.7. This shim eagerly walks
the on-disk `colsearch` package tree at import time and aliases every
submodule under the matching `voyager_index.<sub>` name in `sys.modules`,
so that `import voyager_index.X.Y` resolves to the *exact same module
object* as `colsearch.X.Y` (no duplicate enums or classes). A single
`DeprecationWarning` is emitted on first import so callers can migrate.
The shim will be removed in 0.2.0.
"""

from __future__ import annotations

import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import warnings as _warnings
from typing import Any as _Any

import colsearch as _colsearch
from colsearch import __all__ as _all
from colsearch import __version__ as __version__  # noqa: F401

_warnings.warn(
    "The `voyager_index` package has been renamed to `colsearch`. "
    "Update your imports to `from colsearch import ...`. "
    "The `voyager_index` alias is a compatibility shim that will be removed in 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)


def _alias_submodules() -> None:
    """Discover every `colsearch.*` submodule on disk and alias it under
    `voyager_index.*` in `sys.modules`. Modules that fail to import (e.g. due
    to an optional dependency) are silently skipped — they will simply be
    unavailable through the shim until their dependency is installed, at
    which point the user can `import colsearch.<sub>` directly."""

    # Always alias the top-level package itself.
    _sys.modules.setdefault("voyager_index", _sys.modules[__name__])

    # Walk colsearch's submodule tree from disk. Each successfully-imported
    # submodule becomes both `colsearch.X` and `voyager_index.X`.
    for module_info in _pkgutil.walk_packages(_colsearch.__path__, prefix="colsearch."):
        target = module_info.name
        try:
            module = _importlib.import_module(target)
        except Exception:
            # Optional-dep modules (e.g. fastapi/pydantic-only paths) may
            # fail to import in lean environments; that's fine.
            continue
        alias = "voyager_index." + target[len("colsearch.") :]
        _sys.modules[alias] = module


_alias_submodules()


__all__ = list(_all)


def __getattr__(name: str) -> _Any:
    return getattr(_colsearch, name)


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(__all__) + ["__version__"]))
