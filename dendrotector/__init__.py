"""DendroDetector package utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


ENV_CACHE_PATH = "DENDROCACHE_PATH"
_DEFAULT_CACHE_PATH = Path("~/.dendrocache")


def resolve_cache_dir(
    override: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the directory used for storing downloaded model weights.

    The directory can be customised globally by setting the
    ``DENDROCACHE_PATH`` environment variable. Individual components may also
    provide an explicit ``override`` path which takes precedence over the
    environment variable.
    """

    if override is not None:
        base = Path(override)
    else:
        base = Path(os.environ.get(ENV_CACHE_PATH, _DEFAULT_CACHE_PATH))

    return base.expanduser().resolve()


__all__ = ["ENV_CACHE_PATH", "resolve_cache_dir"]
