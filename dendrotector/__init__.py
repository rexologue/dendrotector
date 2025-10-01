"""DendroDetector package utilities.

This module centralises cache-path handling so that all heavyweight model
artifacts are stored in a user-controlled directory (typically a bind-mounted
volume when running inside Docker). The helpers make the Hugging Face Hub reuse
the same directory, preventing duplicate downloads in ephemeral containers.
"""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download


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
_CACHE_LOGGER = logging.getLogger("dendrotector.cache")
_AUTH_LOGGER = logging.getLogger("dendrotector.auth")


def _ensure_hf_environment(base: Path) -> None:
    """Ensure Hugging Face caches live inside the shared models directory."""

    hf_cache = base / _HF_SUBDIR
    hf_cache.mkdir(parents=True, exist_ok=True)

    if _ENV_HF_HOME not in os.environ:
        os.environ[_ENV_HF_HOME] = str(hf_cache)
        _CACHE_LOGGER.debug("Set %s=%s", _ENV_HF_HOME, hf_cache)

    if _ENV_HF_CACHE not in os.environ:
        os.environ[_ENV_HF_CACHE] = str(hf_cache)
        _CACHE_LOGGER.debug("Set %s=%s", _ENV_HF_CACHE, hf_cache)


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
        _AUTH_LOGGER.info("No Hugging Face token provided; proceeding without authentication")
        _hf_login_attempted = True
        return

    # Mirror the token to the canonical environment variable so that calls to
    # ``hf_hub_download`` automatically pick it up even if users only export the
    # Dendrotector-specific variable.
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    _AUTH_LOGGER.info("Attempting Hugging Face login with provided token")

    try:
        from huggingface_hub import login
    except Exception as exc:  # pragma: no cover - defensive guard
        _AUTH_LOGGER.warning("Failed to import huggingface_hub for auth: %s", exc)
        _hf_login_attempted = True
        return

    try:
        login(token=token, add_to_git_credential=False, new_session=False)
    except Exception as exc:  # pragma: no cover - provide feedback without aborting
        _AUTH_LOGGER.warning(
            "Hugging Face authentication failed; downloads may require 'huggingface-cli login'. Error: %s",
            exc,
        )

    _hf_login_attempted = True
    _AUTH_LOGGER.info("Hugging Face login attempt finished")


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
    _CACHE_LOGGER.debug("Resolved cache directory to %s", resolved)
    _ensure_hf_environment(resolved)
    return resolved


def resolve_hf_cache_dir(override: Optional[Union[str, Path]] = None) -> Path:
    """Return the directory used by the Hugging Face Hub cache."""

    base = resolve_cache_dir(override)
    return (base / _HF_SUBDIR).expanduser().resolve()


def ensure_local_hf_file(
    repo_id: str,
    filename: str,
    target_dir: Union[str, Path],
    *,
    subfolder: Optional[str] = None,
    local_filename: Optional[str] = None,
    **download_kwargs,
) -> Path:
    """Return the on-disk path of ``filename`` stored inside ``target_dir``.

    The helper first checks whether the expected file already exists to avoid
    redundant downloads when the user pre-populates the cache directory (for
    instance by copying ``*.pth`` checkpoints into
    ``~/.dendrocache/huggingface/``). When the file is missing it is fetched
    from the Hugging Face Hub with ``local_dir`` pointing to ``target_dir`` so
    that the resulting path matches the location being checked.
    """

    directory = Path(target_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _CACHE_LOGGER.debug(
        "Ensuring local HF file repo=%s filename=%s target_dir=%s", repo_id, filename, directory
    )

    local_name = local_filename or Path(filename).name
    local_path = directory / local_name

    if local_path.exists():
        _CACHE_LOGGER.info("Reusing existing file %s", local_path)
        return local_path

    download_params = {
        "repo_id": repo_id,
        "filename": filename,
        "cache_dir": str(directory),
        "local_dir": str(directory),
        "local_dir_use_symlinks": False,
        **download_kwargs,
    }
    _CACHE_LOGGER.info("Cached file missing, triggering Hugging Face download")
    _CACHE_LOGGER.debug("hf_hub_download parameters: %s", download_params)

    if subfolder is not None:
        download_params.setdefault("subfolder", subfolder)

    downloaded = Path(hf_hub_download(**download_params))
    _CACHE_LOGGER.info("Download finished; file available at %s", downloaded)

    if local_path.exists():
        _CACHE_LOGGER.debug("Local path %s now exists after download", local_path)
        return local_path

    if downloaded.exists() and downloaded != local_path:
        try:
            shutil.copy2(downloaded, local_path)
            _CACHE_LOGGER.debug("Copied downloaded file to %s", local_path)
            return local_path
        except OSError:
            _CACHE_LOGGER.warning("Failed to copy %s to %s", downloaded, local_path)
            pass

    return local_path if local_path.exists() else downloaded


__all__ = [
    "ENV_CACHE_PATH",
    "ensure_hf_login",
    "ensure_local_hf_file",
    "resolve_cache_dir",
    "resolve_hf_cache_dir",
]
