"""Species classification for detected tree and shrub instances."""
from __future__ import annotations

import json
from typing import Any, Union
from pathlib import Path

import timm
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

MODEL_NAME = "vit_large_patch16_384"
MODEL_REPO = "rexologue/vit_large_384_for_trees"

class SpeciesIdentifier:
    """Identify species using a fine-tuned vision transformer."""

    def __init__(
        self,
        device: str | None = None,
        models_dir: Union[Path, str, None] = None,
    ) -> None:
        self.device = device

        from . import resolve_cache_dir

        self._models_dir = resolve_cache_dir(models_dir)
        self._models_dir.mkdir(exist_ok=True)
        
        model_dir = self._models_dir / "specifier"
        model_dir.mkdir(parents=True, exist_ok=True)

        labels_path = hf_hub_download(MODEL_REPO, filename="labels.json", local_dir=model_dir)
        ckpt_path = hf_hub_download(MODEL_REPO, filename="pytorch_model.bin", local_dir=model_dir)

        with open(labels_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.labels = [raw[str(i)] for i in range(len(raw))] if isinstance(raw, dict) else list(raw)

        state = torch.load(ckpt_path, map_location="cpu")
        if any(k.startswith("module.") for k in state):  # DDP fix
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        self.model = timm.create_model(MODEL_NAME, num_classes=len(self.labels), pretrained=False)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        # preprocessing (ViT-L/16 @ 384 w/ ImageNet mean/std + bicubic)
        self.transform = create_transform(
            input_size=(3, 384, 384),
            interpolation="bicubic",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            is_training=False, 
        )
                
    def identify(
        self,
        image_path: Path | str,
        top_k: int = 1
    ) -> tuple[list[dict[str, Any]], int]:
        
        img = Image.open(image_path).convert("RGB")

        x = self.transform(img).unsqueeze(0).to(self.device) # type: ignore

        with torch.no_grad():
            logits = self.model(x)

        k = max(1, min(top_k, len(self.labels)))

        if k != top_k:
            print(f"\n{top_k} is not acceptable! Fallback on top_{k} predictions!")

        probs = torch.softmax(logits, dim=1)[0].cpu()
        topk = probs.topk(k=k)

        return [{"label": self.labels[i], "prob": float(probs[i])} for i in topk.indices], k
    