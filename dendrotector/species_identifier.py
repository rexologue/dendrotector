"""Region-aware species classification for the DendroDetector pipeline."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .detector import DetectionResult
from .species_config import MOSCOW_REGION_SPECIES, SpeciesDefinition

# Default CLIP checkpoint used for zero-shot classification over the curated list
# of Moscow and temperate-region tree/shrub species.
DEFAULT_SPECIES_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_BACKGROUND_COLOR = (255, 255, 255)
DEFAULT_PROMPT_TEMPLATES = (
    "a photo of a mature {name} tree",
    "a photo of the foliage of {name}",
    "city park planting of {name}",
    "close-up of the bark of {name}",
)


@dataclass
class SpeciesPrediction:
    """Species classification result for an extracted tree or shrub."""

    detection: DetectionResult
    label: str
    score: float
    top_k: Sequence[tuple[str, float]]
    crop_path: Path
    model_name: str
    species_id: str
    scientific_name: str

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "score": float(self.score),
            "top_k": [
                {"label": candidate_label, "score": float(candidate_score)}
                for candidate_label, candidate_score in self.top_k
            ],
            "crop_path": str(self.crop_path),
            "model": self.model_name,
            "species_id": self.species_id,
            "scientific_name": self.scientific_name,
            "detection": self.detection.to_json(),
        }


class SpeciesIdentifier:
    """Classify detected instances into region-specific tree/shrub species."""

    DEFAULT_MODEL_ID = DEFAULT_SPECIES_MODEL_ID

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_SPECIES_MODEL_ID,
        *,
        device: str | None = None,
        top_k: int = 5,
        crop_padding: float = 0.05,
        apply_mask: bool = True,
        background_color: tuple[int, int, int] = DEFAULT_BACKGROUND_COLOR,
        batch_size: int = 4,
        models_dir: os.PathLike[str] | str | None = None,
        torch_dtype: torch.dtype | str | None = None,
        species_definitions: Sequence[SpeciesDefinition] | None = None,
        text_prompt_templates: Sequence[str] | None = None,
    ) -> None:
        """Create a new species identifier tailored to temperate regions."""

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.crop_padding = max(float(crop_padding), 0.0)
        self.apply_mask = bool(apply_mask)
        self.background_color = np.array(background_color, dtype=np.uint8)
        if self.background_color.shape != (3,):
            raise ValueError("background_color must contain exactly three RGB values")
        self.batch_size = max(int(batch_size), 1)
        self._models_dir = Path(models_dir) if models_dir is not None else None
        self.model_name_or_path = model_name_or_path

        if isinstance(torch_dtype, str):
            try:
                torch_dtype = getattr(torch, torch_dtype)
            except AttributeError as error:
                raise ValueError(f"Unknown torch dtype alias: {torch_dtype!r}") from error
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        species = species_definitions or MOSCOW_REGION_SPECIES
        if not species:
            raise ValueError("species_definitions must define at least one species")
        self.species_definitions: tuple[SpeciesDefinition, ...] = tuple(species)
        self._prompt_templates = tuple(text_prompt_templates or DEFAULT_PROMPT_TEMPLATES)

        cache_kwargs = self._cache_kwargs("species")
        self._processor = CLIPProcessor.from_pretrained(
            model_name_or_path,
            **cache_kwargs,
        )
        self._model = CLIPModel.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            **cache_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()

        self._species_features = self._build_species_text_features()
        self._logit_scale = self._model.logit_scale.exp().detach().to(self.device, dtype=self.torch_dtype)

    def _cache_kwargs(self, subdir: str) -> dict:
        if self._models_dir is None:
            return {}
        target_dir = self._models_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        return {"cache_dir": str(target_dir)}

    def identify(
        self,
        image_path: Path | str,
        detections: Sequence[DetectionResult],
        output_dir: Path | str,
        *,
        metadata_filename: str | None = None,
    ) -> List[SpeciesPrediction]:
        """Identify the species for each detection and persist artifacts."""

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            image_np = np.array(image.convert("RGB"))
        height, width = image_np.shape[:2]

        crops: list[Image.Image] = []
        crop_paths: list[Path] = []
        valid_detections: list[DetectionResult] = []

        for idx, detection in enumerate(detections):
            crop, crop_path = self._extract_crop(
                image_np,
                width,
                height,
                detection,
                output_dir,
                image_path.stem,
                idx,
            )
            if crop is None:
                continue
            crops.append(crop)
            crop_paths.append(crop_path)
            valid_detections.append(detection)

        if not crops:
            return []

        predictions = self._classify_crops(crops, valid_detections, crop_paths)

        metadata = [prediction.to_json() for prediction in predictions]
        metadata_name = metadata_filename or f"{image_path.stem}_species.json"
        metadata_path = output_dir / metadata_name
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        return predictions

    def _extract_crop(
        self,
        image_np: np.ndarray,
        width: int,
        height: int,
        detection: DetectionResult,
        output_dir: Path,
        image_stem: str,
        index: int,
    ) -> tuple[Image.Image | None, Path]:
        x0, y0, x1, y1 = detection.bbox

        pad_x = int(round((x1 - x0) * self.crop_padding))
        pad_y = int(round((y1 - y0) * self.crop_padding))
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(width, x1 + pad_x)
        y1 = min(height, y1 + pad_y)

        if x1 <= x0 or y1 <= y0:
            return None, output_dir / f"{image_stem}_crop_{index:02d}.png"

        crop = image_np[y0:y1, x0:x1].copy()

        if self.apply_mask and detection.mask_path.exists():
            mask = self._load_mask(detection.mask_path, (height, width))
            if mask is not None:
                mask_crop = mask[y0:y1, x0:x1]
                if mask_crop.any():
                    crop = np.where(
                        mask_crop[..., None],
                        crop,
                        self.background_color.reshape(1, 1, 3),
                    )

        crop_image = Image.fromarray(crop)
        crop_path = output_dir / f"{image_stem}_crop_{index:02d}.png"
        crop_image.save(crop_path)
        return crop_image, crop_path

    @staticmethod
    def _load_mask(mask_path: Path, expected_shape: tuple[int, int] | None) -> np.ndarray | None:
        try:
            with Image.open(mask_path) as mask_image:
                mask_rgba = mask_image.convert("RGBA")
                mask_np = np.array(mask_rgba)

                if mask_np.shape[-1] == 4:
                    alpha = mask_np[..., 3] > 0
                else:
                    alpha = mask_np[..., 0] > 0

                if expected_shape is not None and alpha.shape != expected_shape:
                    resized = mask_rgba.resize(expected_shape[::-1], Image.NEAREST)
                    resized_np = np.array(resized)
                    if resized_np.shape[-1] == 4:
                        alpha = resized_np[..., 3] > 0
                    else:
                        alpha = resized_np[..., 0] > 0
        except FileNotFoundError:
            return None

        return alpha.astype(bool)

    def _classify_crops(
        self,
        crops: Sequence[Image.Image],
        detections: Sequence[DetectionResult],
        crop_paths: Sequence[Path],
    ) -> List[SpeciesPrediction]:
        predictions: list[SpeciesPrediction] = []

        for start in range(0, len(crops), self.batch_size):
            end = min(start + self.batch_size, len(crops))
            batch_images = crops[start:end]
            batch_detections = detections[start:end]
            batch_paths = crop_paths[start:end]

            inputs = self._processor(images=list(batch_images), return_tensors="pt")
            processed_inputs: dict[str, torch.Tensor] = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    processed_inputs[key] = value.to(self.device, dtype=self.torch_dtype)
                else:
                    processed_inputs[key] = value.to(self.device)

            with torch.inference_mode():
                image_features = self._model.get_image_features(**processed_inputs)
            image_features = self._normalize(image_features.to(torch.float32))

            logits = self._logit_scale * image_features.to(self.torch_dtype) @ self._species_features.T
            probabilities = torch.softmax(logits.to(torch.float32), dim=-1)

            k = min(self.top_k, probabilities.shape[-1])
            top_scores, top_indices = torch.topk(probabilities, k=k, dim=-1)

            for det, crop_path, scores, indices in zip(
                batch_detections,
                batch_paths,
                top_scores,
                top_indices,
            ):
                candidate_scores = scores.tolist()
                candidate_indices = [int(i) for i in indices.tolist()]
                candidate_labels = [self.species_definitions[idx].common_name for idx in candidate_indices]
                primary_index = candidate_indices[0]
                primary_species = self.species_definitions[primary_index]
                prediction = SpeciesPrediction(
                    detection=det,
                    label=primary_species.common_name,
                    score=float(candidate_scores[0]),
                    top_k=list(zip(candidate_labels, candidate_scores)),
                    crop_path=crop_path,
                    model_name=self.model_name_or_path,
                    species_id=primary_species.identifier,
                    scientific_name=primary_species.scientific_name,
                )
                predictions.append(prediction)

        return predictions

    def _build_species_text_features(self) -> torch.Tensor:
        prompts: list[str] = []
        slices: list[tuple[int, int]] = []
        for species in self.species_definitions:
            species_prompts = self._prompts_for_species(species)
            if not species_prompts:
                raise ValueError(f"No prompts generated for species '{species.identifier}'")
            start = len(prompts)
            prompts.extend(species_prompts)
            slices.append((start, len(prompts)))

        text_inputs = self._processor(text=prompts, padding=True, return_tensors="pt")
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        with torch.inference_mode():
            text_features = self._model.get_text_features(**text_inputs)
        text_features = self._normalize(text_features.to(torch.float32))

        aggregated: list[torch.Tensor] = []
        for (start, end) in slices:
            pooled = text_features[start:end].mean(dim=0)
            pooled = self._normalize(pooled.unsqueeze(0))[0]
            aggregated.append(pooled)

        stacked = torch.stack(aggregated, dim=0)
        return stacked.to(self.device, dtype=self.torch_dtype)

    def _prompts_for_species(self, species: SpeciesDefinition) -> list[str]:
        prompts: list[str] = []
        seen: set[str] = set()

        for name in species.all_names():
            for template in self._prompt_templates:
                prompt = template.format(
                    name=name,
                    common_name=species.common_name,
                    scientific_name=species.scientific_name,
                ).strip()
                if not prompt:
                    continue
                lower = prompt.lower()
                if lower in seen:
                    continue
                seen.add(lower)
                prompts.append(prompt)

        for prompt in species.prompts:
            normalized = prompt.strip()
            if not normalized:
                continue
            lower = normalized.lower()
            if lower in seen:
                continue
            seen.add(lower)
            prompts.append(normalized)

        return prompts

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_detections(metadata_path: Path | str) -> List[DetectionResult]:
    """Load serialized detection results from a metadata JSON file."""

    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    base_dir = metadata_path.parent
    detections = [DetectionResult.from_json(item, base_dir=base_dir) for item in payload]
    return detections
