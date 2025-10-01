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
