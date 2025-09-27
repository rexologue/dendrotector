"""Species classification for detected trees and shrubs.

The previous placeholder implementation relied on a generic ImageNet model,
which yielded coarse labels such as ``"oak"`` or ``"pine"`` at best.  This file
now integrates a pretrained tree-focused classifier published on the Hugging
Face Hub: :mod:`OttoYu/TreeClassification`.  The model is a fine-tuned
`Swin Transformer <https://huggingface.co/docs/transformers/model_doc/swin>`_
that was trained on a curated dataset of subtropical tree and shrub species.
It predicts 13 classes including *Araucaria columnaris*, *Callistemon
viminalis*, *Hibiscus tiliaceus*, and other ornamental species commonly found
in public datasets such as TreeSpecies.

During inference we crop (and optionally mask) each detected instance before
feeding it through the classifier.  The module exposes a high-level
:class:`SpeciesIdentifier` class that mirrors
:class:`~dendrotector.detector.DendroDetector` in spirit: detections go in,
species predictions with probabilities come out, and all intermediary
crops/metadata are saved to disk.

The implementation is GPU-ready.  When CUDA is available we default to running
the classifier in ``float16`` precision, which fits comfortably on modern
accelerators such as the NVIDIA H100 bundled with this environment.
"""
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

# Publicly available Swin Transformer trained for tree species recognition.
DEFAULT_SPECIES_MODEL_ID = "OttoYu/TreeClassification"
DEFAULT_BACKGROUND_COLOR = (255, 255, 255)


@dataclass
class SpeciesPrediction:
    """Species classification result for an extracted tree or shrub."""

    detection: DetectionResult
    label: str
    score: float
    top_k: Sequence[tuple[str, float]]
    crop_path: Path
    model_name: str

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
            "detection": self.detection.to_json(),
        }


class SpeciesIdentifier:
    """Classify detected instances into tree/shrub species.

    The identifier is purposely opinionated: by default it loads the
    :data:`~dendrotector.species_identifier.DEFAULT_SPECIES_MODEL_ID` Swin
    Transformer fine-tuned for tree species recognition.  Users can override the
    ``model_name_or_path`` argument to experiment with alternative models (for
    example, broader PlantCLEF checkpoints) while retaining the same
    preprocessing and batching logic.
    """

    DEFAULT_MODEL_ID = DEFAULT_SPECIES_MODEL_ID

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_SPECIES_MODEL_ID,
        *,
        device: str | None = None,
        trust_remote_code: bool = False,
        top_k: int = 5,
        crop_padding: float = 0.05,
        apply_mask: bool = True,
        background_color: tuple[int, int, int] = DEFAULT_BACKGROUND_COLOR,
        batch_size: int = 4,
        models_dir: os.PathLike[str] | str | None = None,
        torch_dtype: torch.dtype | str | None = None,
    ) -> None:
        """Create a new identifier.

        Parameters
        ----------
        model_name_or_path:
            Hugging Face model name or path. Defaults to the
            ``OttoYu/TreeClassification`` species head.
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
        models_dir:
            Optional directory where downloaded model files should be cached.
        torch_dtype:
            Preferred :class:`torch.dtype`.  Defaults to ``float16`` on CUDA devices
            and ``float32`` otherwise.
        """

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

        cache_kwargs = self._cache_kwargs("species")
        self._processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cache_kwargs,
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=self.torch_dtype,
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
            processed_inputs: dict[str, torch.Tensor] = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    processed_inputs[key] = value.to(self.device, dtype=self.torch_dtype)
                else:
                    processed_inputs[key] = value.to(self.device)

            inputs = processed_inputs

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
                    model_name=self.model_name_or_path,
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
