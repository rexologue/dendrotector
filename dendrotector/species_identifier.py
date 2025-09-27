"""Species classification for detected trees and shrubs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .detector import DetectionResult


@dataclass
class SpeciesPrediction:
    """Species classification result for an extracted tree or shrub."""

    detection: DetectionResult
    label: str
    score: float
    top_k: Sequence[tuple[str, float]]
    crop_path: Path

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "score": float(self.score),
            "top_k": [
                {"label": candidate_label, "score": float(candidate_score)}
                for candidate_label, candidate_score in self.top_k
            ],
            "crop_path": str(self.crop_path),
            "detection": self.detection.to_json(),
        }


class SpeciesIdentifier:
    """High-quality species identifier built on top of `transformers` models."""

    def __init__(
        self,
        model_name_or_path: str = "google/vit-huge-patch14-224-in21k",
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
        top_k: int = 5,
        crop_padding: float = 0.05,
        apply_mask: bool = True,
        background_color: tuple[int, int, int] = (255, 255, 255),
        batch_size: int = 4,
        models_dir: os.PathLike[str] | str | None = None,
    ) -> None:
        """Create a new identifier.

        Parameters
        ----------
        model_name_or_path:
            Hugging Face model name or path. Defaults to a high-capacity ViT model.
        device:
            Device identifier. If ``None`` the method automatically selects CUDA when
            available.
        trust_remote_code:
            Whether to trust remote code when loading the model. Some community models
            require this flag.
        top_k:
            Number of highest probability classes to keep per detection.
        crop_padding:
            Extra padding added around each detection bounding box before cropping,
            expressed as a fraction of the box dimensions.
        apply_mask:
            If ``True``, the SAM mask is used to remove background pixels from the
            crop before classification.
        background_color:
            RGB color used to fill areas outside the predicted mask.
        batch_size:
            Maximum number of crops to classify at once.
        """

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.crop_padding = max(float(crop_padding), 0.0)
        self.apply_mask = apply_mask
        self.background_color = np.array(background_color, dtype=np.uint8)
        self.batch_size = max(int(batch_size), 1)
        self._models_dir = Path(models_dir) if models_dir is not None else None

        cache_kwargs = self._cache_kwargs("species")
        self._processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cache_kwargs,
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cache_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()

        config = self._model.config
        self._id2label = {int(idx): label for idx, label in config.id2label.items()}

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
        """Identify the species for each detection and persist artifacts.

        Parameters
        ----------
        image_path:
            Path to the source image used to produce the detections.
        detections:
            Iterable of :class:`DetectionResult` objects emitted by
            :meth:`dendrotector.detector.DendroDetector.detect`.
        output_dir:
            Directory where cropped instances and metadata will be written.
        metadata_filename:
            Optional filename for the JSON metadata file. The default is derived from
            the input image stem.
        """

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
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.inference_mode():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            probabilities = probabilities.cpu()

            k = min(self.top_k, probabilities.shape[-1])
            top_scores, top_indices = torch.topk(probabilities, k=k, dim=-1)

            for det, crop_path, scores, indices in zip(
                batch_detections,
                batch_paths,
                top_scores,
                top_indices,
            ):
                candidate_scores = scores.tolist()
                candidate_indices = indices.tolist()
                candidate_labels = [self._id2label.get(int(idx), str(idx)) for idx in candidate_indices]
                prediction = SpeciesPrediction(
                    detection=det,
                    label=candidate_labels[0],
                    score=float(candidate_scores[0]),
                    top_k=list(zip(candidate_labels, candidate_scores)),
                    crop_path=crop_path,
                )
                predictions.append(prediction)

        return predictions


def load_detections(metadata_path: Path | str) -> List[DetectionResult]:
    """Load serialized detection results from a metadata JSON file."""

    metadata_path = Path(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    base_dir = metadata_path.parent
    detections = [DetectionResult.from_json(item, base_dir=base_dir) for item in payload]
    return detections
