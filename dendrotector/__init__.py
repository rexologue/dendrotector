"""DendroDetector package utilities.

This module centralises cache-path handling so that all heavyweight model
artifacts are stored in a user-controlled directory (typically a bind-mounted
volume when running inside Docker). The helpers make the Hugging Face Hub reuse
the same directory, preventing duplicate downloads in ephemeral containers.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

def load_from_hf(repo_id: str, filename: str, save_path: Path) -> None:
    cached_path = hf_hub_download(repo_id, filename)
    shutil.copy(cached_path, str(save_path))

    try:
        os.remove(cached_path)
        
    except OSError:
        pass 

