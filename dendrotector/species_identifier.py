"""Species classification for detected tree and shrub instances."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .detector import DetectionResult

DEFAULT_SPECIES_MODEL_ID = "rexologue/vit_large_384_for_trees"
DEFAULT_BACKGROUND_COLOR = (255, 255, 255)


@dataclass
class SpeciesPrediction:
    """Species classification result for an extracted tree or shrub."""

    detection: DetectionResult
    instance_index: int
    label: str
    score: float
    top_k: Sequence[tuple[str, float]]
    crop_path: Path
    crop_size: tuple[int, int]
    model_name: str
    crop_mode: str

    def to_json(self) -> dict:
        return {
            "instance_index": int(self.instance_index),
            "label": self.label,
            "score": float(self.score),
            "top_k": [
                {"label": candidate_label, "score": float(candidate_score)}
                for candidate_label, candidate_score in self.top_k
            ],
            "crop_path": str(self.crop_path),
            "crop_size": list(self.crop_size),
            "model": self.model_name,
            "crop_mode": self.crop_mode,
            "detection": self.detection.to_json(),
        }


class SpeciesIdentifier:
    """Identify species using a fine-tuned vision transformer."""

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
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_k = max(int(top_k), 1)
        self.crop_padding = max(float(crop_padding), 0.0)
        self.apply_mask = bool(apply_mask)
        self.background_color = np.array(background_color, dtype=np.uint8)
        if self.background_color.shape != (3,):
            raise ValueError("background_color must contain three RGB components")
        self.batch_size = max(int(batch_size), 1)
        self._models_dir = Path(models_dir) if models_dir is not None else None

        if isinstance(torch_dtype, str):
            try:
                torch_dtype = getattr(torch, torch_dtype)
            except AttributeError as error:
                raise ValueError(f"Unknown torch dtype alias: {torch_dtype!r}") from error
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.torch_dtype = torch_dtype

        cache_kwargs = self._cache_kwargs("species")
        self._processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            **cache_kwargs,
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            **cache_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()

        config = self._model.config
        if getattr(config, "id2label", None):
            self._id2label = {int(idx): label for idx, label in config.id2label.items()}
        else:
            self._id2label = {idx: str(idx) for idx in range(config.num_labels)}

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
        crop_mode: Literal["bbox", "mask"] = "mask",
        metadata_filename: str | None = None,
    ) -> list[SpeciesPrediction]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            image_np = np.array(image.convert("RGB"))
        height, width = image_np.shape[:2]

        crops: list[Image.Image] = []
        crop_paths: list[Path] = []
        crop_sizes: list[tuple[int, int]] = []
        valid_detections: list[DetectionResult] = []
        valid_indices: list[int] = []

        for index, detection in enumerate(detections):
            crop, crop_path, crop_size = self._extract_crop(
                image_np,
                width,
                height,
                detection,
                output_dir,
                image_path.stem,
                index,
                crop_mode,
            )
            if crop is None:
                continue
            crops.append(crop)
            crop_paths.append(crop_path)
            crop_sizes.append(crop_size)
            valid_detections.append(detection)
            valid_indices.append(index)

        if not crops:
            return []

        predictions = self._classify_crops(
            crops,
            valid_detections,
            crop_paths,
            crop_sizes,
            valid_indices,
            crop_mode,
        )

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
        crop_mode: str,
    ) -> tuple[Image.Image | None, Path, tuple[int, int]]:
        x0, y0, x1, y1 = detection.bbox

        mask: np.ndarray | None = None
        if (self.apply_mask or crop_mode == "mask") and detection.mask_path.exists():
            mask = self._load_mask(detection.mask_path, (height, width))

        if crop_mode == "mask" and mask is not None and mask.any():
            coords = np.column_stack(np.nonzero(mask))
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            y0, y1 = int(y_coords.min()), int(y_coords.max()) + 1
            x0, x1 = int(x_coords.min()), int(x_coords.max()) + 1

        pad_x = int(round((x1 - x0) * self.crop_padding))
        pad_y = int(round((y1 - y0) * self.crop_padding))
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(width, x1 + pad_x)
        y1 = min(height, y1 + pad_y)

        if x1 <= x0 or y1 <= y0:
            return None, output_dir / f"{image_stem}_crop_{index:02d}.png", (0, 0)

        crop_np = image_np[y0:y1, x0:x1].copy()

        if self.apply_mask and mask is not None:
            mask_crop = mask[y0:y1, x0:x1]
            if mask_crop.any():
                crop_np = np.where(
                    mask_crop[..., None],
                    crop_np,
                    self.background_color.reshape(1, 1, 3),
                )

        crop_image = Image.fromarray(crop_np)
        crop_path = output_dir / f"{image_stem}_crop_{index:02d}.png"
        crop_image.save(crop_path)
        crop_size = (crop_image.height, crop_image.width)
        return crop_image, crop_path, crop_size

    def _classify_crops(
        self,
        crops: Sequence[Image.Image],
        detections: Sequence[DetectionResult],
        crop_paths: Sequence[Path],
        crop_sizes: Sequence[tuple[int, int]],
        indices: Sequence[int],
        crop_mode: str,
    ) -> list[SpeciesPrediction]:
        predictions: list[SpeciesPrediction] = []

        for start in range(0, len(crops), self.batch_size):
            end = min(start + self.batch_size, len(crops))
            batch_images = list(crops[start:end])
            batch_detections = detections[start:end]
            batch_paths = crop_paths[start:end]
            batch_sizes = crop_sizes[start:end]
            batch_indices = indices[start:end]

            inputs = self._processor(images=batch_images, return_tensors="pt")
            processed_inputs: dict[str, torch.Tensor] = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    processed_inputs[key] = value.to(self.device, dtype=self.torch_dtype)
                else:
                    processed_inputs[key] = value.to(self.device)

            with torch.inference_mode():
                outputs = self._model(**processed_inputs)
            logits = outputs.logits.to(torch.float32)
            probabilities = torch.softmax(logits, dim=-1)

            k = min(self.top_k, probabilities.shape[-1])
            top_scores, top_indices = torch.topk(probabilities, k=k, dim=-1)

            for det, crop_path, crop_size, scores, indices_row, instance_index in zip(
                batch_detections,
                batch_paths,
                batch_sizes,
                top_scores,
                top_indices,
                batch_indices,
            ):
                candidate_scores = scores.tolist()
                candidate_indices = [int(i) for i in indices_row.tolist()]
                candidate_labels = [self._id2label[idx] for idx in candidate_indices]
                prediction = SpeciesPrediction(
                    detection=det,
                    instance_index=instance_index,
                    label=candidate_labels[0],
                    score=float(candidate_scores[0]),
                    top_k=list(zip(candidate_labels, candidate_scores)),
                    crop_path=crop_path,
                    crop_size=crop_size,
                    model_name=self.model_name_or_path,
                    crop_mode=crop_mode,
                )
                predictions.append(prediction)

        return predictions

    @staticmethod
    def _load_mask(mask_path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
        try:
            with Image.open(mask_path) as mask_image:
                mask_rgba = mask_image.convert("RGBA")
                mask_np = np.array(mask_rgba)
        except FileNotFoundError:
            return None

        if mask_np.shape[-1] == 4:
            alpha = mask_np[..., 3] > 0
        else:
            alpha = mask_np[..., 0] > 0

        if alpha.shape != expected_shape:
            resized = mask_rgba.resize(expected_shape[::-1], Image.NEAREST)
            resized_np = np.array(resized)
            if resized_np.shape[-1] == 4:
                alpha = resized_np[..., 3] > 0
            else:
                alpha = resized_np[..., 0] > 0

        return alpha.astype(bool)


def load_detections(metadata_path: Path | str) -> list[DetectionResult]:
    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    base_dir = metadata_path.parent
    detections = [DetectionResult.from_json(item, base_dir=base_dir) for item in payload]
    return detections

