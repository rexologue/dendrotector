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

_ENV_PRIMARY_HF_TOKEN = "DENDROTECTOR_HF_TOKEN"
_ENV_ALT_HF_TOKENS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)

_hf_login_attempted = False


def _ensure_hf_environment(base: Path) -> None:
    """Ensure Hugging Face caches live inside the shared models directory."""

    hf_cache = base / _HF_SUBDIR
    hf_cache.mkdir(parents=True, exist_ok=True)

    if _ENV_HF_HOME not in os.environ:
        os.environ[_ENV_HF_HOME] = str(hf_cache)

    if _ENV_HF_CACHE not in os.environ:
        os.environ[_ENV_HF_CACHE] = str(hf_cache)


def _resolve_hf_token() -> Optional[str]:
    """Return the Hugging Face token from supported environment variables."""

    token = os.environ.get(_ENV_PRIMARY_HF_TOKEN)
    if token:
        return token.strip()

    for env_name in _ENV_ALT_HF_TOKENS:
        token = os.environ.get(env_name)
        if token:
            return token.strip()

    return None


def ensure_hf_login() -> None:
    """Authenticate with the Hugging Face Hub when a token is provided.

    The helper accepts a token via ``DENDROTECTOR_HF_TOKEN`` as the primary
    variable and mirrors it to the standard ``HUGGING_FACE_HUB_TOKEN`` so that
    downstream libraries automatically reuse the credential. ``HF_TOKEN`` and
    ``HUGGING_FACE_HUB_TOKEN`` remain backwards-compatible aliases. The login is
    attempted only once per process.
    """

    global _hf_login_attempted

    if _hf_login_attempted:
        return

    token = _resolve_hf_token()
    if not token:
        _hf_login_attempted = True
        return

    # Mirror the token to the canonical environment variable so that calls to
    # ``hf_hub_download`` automatically pick it up even if users only export the
    # Dendrotector-specific variable.
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    try:
        from huggingface_hub import login
    except Exception as exc:  # pragma: no cover - defensive guard
        print(
            "[Dendrotector] Warning: failed to import huggingface_hub for auth "
            f"({exc}).",
            flush=True,
        )
        _hf_login_attempted = True
        return

    try:
        login(token=token, add_to_git_credential=False, new_session=False)
    except Exception as exc:  # pragma: no cover - provide feedback without aborting
        print(
            "[Dendrotector] Warning: Hugging Face authentication failed. "
            f"Downloads may require 'huggingface-cli login'. Error: {exc}",
            flush=True,
        )

    _hf_login_attempted = True


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


__all__ = [
    "ENV_CACHE_PATH",
    "ensure_hf_login",
    "resolve_cache_dir",
    "resolve_hf_cache_dir",
]
