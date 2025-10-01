"""Species classification for detected tree and shrub instances."""
from __future__ import annotations

import json
from typing import Any, Union
from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from . import load_from_hf

MODEL_NAME = "vit_large_patch16_384"
MODEL_REPO = "rexologue/vit_large_384_for_trees"

class SpeciesIdentifier:
    """Identify species using a fine-tuned vision transformer."""

    def __init__(
        self,
        device: str | None = None,
        models_dir: Path = Path("~/.dendrocache"),
    ) -> None:
        self.device = device

        self._models_dir = models_dir.expanduser().resolve()
        self._models_dir.mkdir(parents=True, exist_ok=True)

        specifier_dir = self._models_dir / "specifier"

        labels_path = specifier_dir / "labels.json"
        ckpt_path = specifier_dir / "pytorch_model.bin"

        if not labels_path.exists():
            load_from_hf(MODEL_REPO, "labels.json", labels_path)

        if not ckpt_path.exists():
            load_from_hf(MODEL_REPO, "pytorch_model.bin", ckpt_path)

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
    