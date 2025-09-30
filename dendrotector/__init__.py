"""DendroDetector package utilities.

This module centralises cache-path handling so that all heavyweight model
artifacts are stored in a user-controlled directory (typically a bind-mounted
volume when running inside Docker). The helpers make the Hugging Face Hub reuse
the same directory, preventing duplicate downloads in ephemeral containers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


ENV_CACHE_PATH = "DENDROCACHE_PATH"
_DEFAULT_CACHE_PATH = Path("~/.dendrocache")

_ENV_HF_HOME = "HF_HOME"
_ENV_HF_CACHE = "HUGGINGFACE_HUB_CACHE"
_HF_SUBDIR = "huggingface"


def _ensure_hf_environment(base: Path) -> None:
    """Ensure Hugging Face caches live inside the shared models directory."""

    hf_cache = base / _HF_SUBDIR
    hf_cache.mkdir(parents=True, exist_ok=True)

    if _ENV_HF_HOME not in os.environ:
        os.environ[_ENV_HF_HOME] = str(hf_cache)

    if _ENV_HF_CACHE not in os.environ:
        os.environ[_ENV_HF_CACHE] = str(hf_cache)


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

    resolved = base.expanduser().resolve()
    _ensure_hf_environment(resolved)
    return resolved


def resolve_hf_cache_dir(override: Optional[Union[str, Path]] = None) -> Path:
    """Return the directory used by the Hugging Face Hub cache."""

    base = resolve_cache_dir(override)
    return (base / _HF_SUBDIR).expanduser().resolve()


__all__ = ["ENV_CACHE_PATH", "resolve_cache_dir", "resolve_hf_cache_dir"]
