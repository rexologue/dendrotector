"""Instance segmentation pipeline for detecting trees and shrubs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, load_model, predict
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .report import DendroReport, GeneralInfo, InstanceReport, SpeciesSummary

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .species_identifier import SpeciesPrediction

PROMPT = "tree . shrub . bush ."
GROUNDING_REPO = "ShilongLiu/GroundingDINO"
GROUNDING_CONFIG_CANDIDATES = (
    "GroundingDINO_SwinT_OGC.cfg.py",
    "GroundingDINO_SwinT_OGC.py",
)
GROUNDING_WEIGHTS = "groundingdino_swint_ogc.pth"
# Mapping derived from the repositories published at
# https://huggingface.co/facebook for the SAM 2 checkpoints.
SAM2_MODELS = {
    "hiera_t": "facebook/sam2-hiera-tiny",
    "hiera_s": "facebook/sam2-hiera-small",
    "hiera_b+": "facebook/sam2-hiera-base-plus",
    "hiera_l": "facebook/sam2-hiera-large",
    "hiera_t_2.1": "facebook/sam2.1-hiera-tiny",
    "hiera_s_2.1": "facebook/sam2.1-hiera-small",
    "hiera_b+_2.1": "facebook/sam2.1-hiera-base-plus",
    "hiera_l_2.1": "facebook/sam2.1-hiera-large",
    # Legacy aliases preserved for backwards compatibility with previous SAM models.
    "vit_b": "facebook/sam2-hiera-small",
    "vit_l": "facebook/sam2-hiera-base-plus",
    "vit_h": "facebook/sam2-hiera-large",
}

DEFAULT_SPECIES_MODEL = "rexologue/vit_large_384_for_trees"


@dataclass
class DetectionResult:
    """Holds the result of a single detection."""

    label: str
    score: float
    bbox: List[int]
    mask_path: Path

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "score": float(self.score),
            "bbox": self.bbox,
            "mask_path": str(self.mask_path),
        }

    @classmethod
    def from_json(cls, payload: dict, base_dir: Path | None = None) -> "DetectionResult":
        """Recreate a :class:`DetectionResult` from serialized metadata."""

        mask_path = Path(payload["mask_path"])
        if base_dir is not None and not mask_path.is_absolute():
            mask_path = base_dir / mask_path

        return cls(
            label=payload["label"],
            score=float(payload["score"]),
            bbox=[int(v) for v in payload["bbox"]],
            mask_path=mask_path,
        )


@dataclass
class DetectionArtifacts:
    """Container for detection outputs and associated metadata."""

    image_path: Path
    image_size: tuple[int, int]
    overlay_path: Path
    metadata_path: Path
    detections: List[DetectionResult]

    def to_json(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "image_size": list(self.image_size),
            "overlay_path": str(self.overlay_path),
            "metadata_path": str(self.metadata_path),
            "detections": [detection.to_json() for detection in self.detections],
        }


class DendroDetector:
    """High-level wrapper around GroundingDINO + SAM 2."""

    def __init__(
        self,
        device: Optional[str] = None,
        sam_model: str = "vit_h",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        models_dir: os.PathLike[str] | str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._models_dir = Path(models_dir) if models_dir is not None else None

        self._dino_model = self._load_groundingdino()
        self._sam_model_name = sam_model
        self._sam_predictor = self._load_sam(sam_model)

    def _load_groundingdino(self):
        download_kwargs = self._download_kwargs("groundingdino")
        config_path = self._download_groundingdino_config(download_kwargs)
        weights_path = hf_hub_download(GROUNDING_REPO, GROUNDING_WEIGHTS, **download_kwargs)
        model = load_model(config_path, weights_path)
        model.to(self.device)
        model.eval()
        return model

    def _download_groundingdino_config(self, download_kwargs: dict) -> str:
        last_error: EntryNotFoundError | None = None
        for filename in GROUNDING_CONFIG_CANDIDATES:
            try:
                return hf_hub_download(GROUNDING_REPO, filename, **download_kwargs)
            except EntryNotFoundError as error:
                last_error = error
        if last_error is not None:
            raise last_error
        raise RuntimeError("No GroundingDINO config candidates configured")

    def _load_sam(self, sam_model: str) -> SAM2ImagePredictor:
        download_kwargs = self._download_kwargs("sam2")

        try:
            repo_id = SAM2_MODELS[sam_model]
        except KeyError as error:
            known_models = ", ".join(sorted(SAM2_MODELS))
            raise ValueError(
                f"Unsupported SAM2 model '{sam_model}'. Known models: {known_models}."
            ) from error

        try:
            config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[repo_id]
        except KeyError as error:
            raise ValueError(
                f"No configuration mapping found for SAM2 repository '{repo_id}'."
            ) from error

        checkpoint_path = hf_hub_download(repo_id, checkpoint_name, **download_kwargs)
        sam_model_instance = build_sam2(
            config_file=config_name,
            ckpt_path=checkpoint_path,
            device=self.device,
        )
        return SAM2ImagePredictor(sam_model_instance)

    def _download_kwargs(self, subdir: str) -> dict:
        if self._models_dir is None:
            return {}
        target_dir = self._models_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        return {"local_dir": str(target_dir), "local_dir_use_symlinks": False}

    def detect(
        self,
        image_path: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
        *,
        prompt: str = PROMPT,
        multimask_output: bool = False,
    ) -> DetectionArtifacts:
        """Run tree and shrub instance segmentation on an image.

        Parameters
        ----------
        image_path:
            Path to the input image.
        output_dir:
            Directory where mask artifacts and metadata will be written.
        prompt:
            Text prompt for GroundingDINO. Defaults to a tree-focused prompt.
        multimask_output:
            If True, SAM 2 will return up to three mask proposals per box; the
            best scoring mask is selected otherwise.
        """

        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_source, image_tensor = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=self._dino_model,
            image=image_tensor,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        if boxes.shape[0] == 0:
            metadata_path = output_dir / f"{image_path.stem}_detections.json"
            with metadata_path.open("w", encoding="utf-8") as fp:
                json.dump([], fp, indent=2)
            overlay_path = output_dir / f"{image_path.stem}_overlay.png"
            cv2.imwrite(str(overlay_path), self._draw_overlay(image_source, np.empty((0, 4)), []))
            return DetectionArtifacts(
                image_path=image_path,
                image_size=image_source.shape[:2],
                overlay_path=overlay_path,
                metadata_path=metadata_path,
                detections=[],
            )

        h, w = image_source.shape[:2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes_xyxy *= torch.tensor([w, h, w, h], device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy.cpu()

        self._sam_predictor.set_image(image_source)

        mask_candidates: list[np.ndarray] = []
        iou_candidates: list[np.ndarray] = []
        for box in boxes_xyxy.numpy():
            masks_np, iou_predictions_np, _ = self._sam_predictor.predict(
                box=box,
                multimask_output=multimask_output,
                normalize_coords=True,
            )
            mask_candidates.append(masks_np)
            iou_candidates.append(iou_predictions_np)

        boxes_xyxy = boxes_xyxy.numpy()
        logits = logits.sigmoid().cpu().numpy()

        overlay = self._draw_overlay(image_source, boxes_xyxy, mask_candidates)
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        results: List[DetectionResult] = []
        for idx, (box, mask_set, iou_set, logit, phrase) in enumerate(
            zip(boxes_xyxy, mask_candidates, iou_candidates, logits, phrases)
        ):
            best_mask = mask_set[0]
            if multimask_output and mask_set.shape[0] > 1:
                best_idx = int(np.argmax(iou_set))
                best_mask = mask_set[best_idx]

            mask_binary = best_mask.astype(bool)
            mask_path = output_dir / f"{image_path.stem}_instance_{idx:02d}.png"
            self._save_mask(mask_binary, mask_path)

            bbox = [int(round(v)) for v in box.tolist()]
            label = phrase.strip() or "tree"
            score = float(np.max(logit))

            results.append(
                DetectionResult(
                    label=label,
                    score=score,
                    bbox=bbox,
                    mask_path=mask_path,
                )
            )

        metadata_path = output_dir / f"{image_path.stem}_detections.json"
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump([r.to_json() for r in results], fp, indent=2)

        return DetectionArtifacts(
            image_path=image_path,
            image_size=image_source.shape[:2],
            overlay_path=overlay_path,
            metadata_path=metadata_path,
            detections=results,
        )

    @staticmethod
    def _save_mask(mask: np.ndarray, path: Path) -> None:
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 34
        rgba[..., 1] = 139
        rgba[..., 2] = 34
        rgba[..., 3] = np.where(mask, 200, 0)
        cv2.imwrite(str(path), rgba)

    @staticmethod
    def _draw_overlay(
        image: np.ndarray,
        boxes: np.ndarray,
        masks: List[np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        overlay = image.copy()
        if overlay.dtype != np.uint8:
            overlay = (overlay * 255).astype(np.uint8)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        color = np.array((34, 139, 34), dtype=np.float32)
        for idx, box in enumerate(boxes):
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color.astype(np.uint8).tolist(), 2)
            cv2.putText(
                overlay,
                f"instance#{idx}",
                (x0, max(y0 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color.astype(np.uint8).tolist(),
                2,
                cv2.LINE_AA,
            )
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask.astype(bool)
            blended = overlay[mask_bool].astype(np.float32) * 0.5 + color * 0.5
            overlay[mask_bool] = blended.astype(np.uint8)
        return overlay

    @staticmethod
    def _infer_instance_type(label: str) -> str:
        lowered = label.lower()
        if any(keyword in lowered for keyword in ("shrub", "bush", "hedge")):
            return "shrub"
        return "tree"

    @staticmethod
    def _load_mask(mask_path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
        try:
            with Image.open(mask_path) as mask_image:  # type: ignore[name-defined]
                mask_rgba = mask_image.convert("RGBA")
                mask_np = np.array(mask_rgba)
        except FileNotFoundError:
            return None

        if mask_np.shape[-1] == 4:
            alpha = mask_np[..., 3] > 0
        else:
            alpha = mask_np[..., 0] > 0

        if alpha.shape != expected_shape:
            mask_rgba = mask_rgba.resize(expected_shape[::-1], Image.NEAREST)  # type: ignore[has-type]
            mask_np = np.array(mask_rgba)
            if mask_np.shape[-1] == 4:
                alpha = mask_np[..., 3] > 0
            else:
                alpha = mask_np[..., 0] > 0

        return alpha.astype(bool)

    @staticmethod
    def _mask_area(mask: np.ndarray | None) -> int | None:
        if mask is None:
            return None
        return int(mask.sum())

    @staticmethod
    def _lean_angle(mask: np.ndarray | None) -> float | None:
        if mask is None or not mask.any():
            return None

        coords = np.column_stack(np.nonzero(mask))
        if coords.shape[0] < 2:
            return None

        mean = coords.mean(axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
        vertical = np.array([1.0, 0.0])
        normalized = principal_vector / np.linalg.norm(principal_vector)
        dot = abs(float(np.dot(normalized, vertical)))
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = float(np.arccos(dot))
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def generate_report(
        self,
        image_path: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
        *,
        prompt: str = PROMPT,
        multimask_output: bool = False,
        crop_mode: str = "mask",
        species_model: str | None = DEFAULT_SPECIES_MODEL,
        species_device: str | None = None,
        species_top_k: int = 5,
        species_crop_padding: float = 0.05,
        species_batch_size: int = 4,
        species_apply_mask: bool = True,
        report_filename: str | None = None,
        extra_general_info: Dict[str, Any] | None = None,
    ) -> DendroReport:
        """Run detection, species identification and build a consolidated report."""

        artifacts = self.detect(
            image_path=image_path,
            output_dir=output_dir,
            prompt=prompt,
            multimask_output=multimask_output,
        )

        predictions: Sequence["SpeciesPrediction"] = []
        species_model_name: str | None = None
        prediction_map: Dict[int, "SpeciesPrediction"] = {}
        species_output_dir = Path(output_dir) / "species"

        if species_model is not None:
            from .species_identifier import SpeciesIdentifier  # pylint: disable=import-outside-toplevel

            identifier = SpeciesIdentifier(
                model_name_or_path=species_model,
                device=species_device,
                top_k=species_top_k,
                crop_padding=species_crop_padding,
                apply_mask=species_apply_mask,
                batch_size=species_batch_size,
                models_dir=self._models_dir,
            )
            predictions = identifier.identify(
                image_path=image_path,
                detections=artifacts.detections,
                output_dir=species_output_dir,
                crop_mode=crop_mode,
            )
            species_model_name = identifier.model_name_or_path
            prediction_map = {prediction.instance_index: prediction for prediction in predictions}

        total_instances = len(artifacts.detections)
        instances: list[InstanceReport] = []
        tree_count = 0
        shrub_count = 0
        total_mask_area = 0

        for index, detection in enumerate(artifacts.detections):
            instance_type = self._infer_instance_type(detection.label)
            if instance_type == "tree":
                tree_count += 1
            else:
                shrub_count += 1

            mask = self._load_mask(detection.mask_path, artifacts.image_size)
            mask_area = self._mask_area(mask)
            if mask_area is not None:
                total_mask_area += mask_area
            lean_angle = self._lean_angle(mask) if instance_type == "tree" else None

            species_prediction = prediction_map.get(index)
            species_summary: SpeciesSummary | None = None
            if species_prediction is not None:
                species_summary = SpeciesSummary(
                    label=species_prediction.label,
                    score=species_prediction.score,
                    top_k=species_prediction.top_k,
                    model_name=species_model_name or species_prediction.model_name,
                    crop_path=species_prediction.crop_path,
                    crop_size=species_prediction.crop_size,
                )

            extra: Dict[str, Any] = {
                "detection_mask_exists": detection.mask_path.exists(),
            }
            instances.append(
                InstanceReport(
                    index=index,
                    detection=detection,
                    instance_type=instance_type,
                    mask_path=detection.mask_path,
                    mask_area_px=mask_area,
                    lean_angle_degrees=lean_angle,
                    species=species_summary,
                    extra=extra,
                )
            )

        metadata = GeneralInfo(
            image_path=artifacts.image_path,
            overlay_path=artifacts.overlay_path,
            detection_metadata_path=artifacts.metadata_path,
            detection_model="GroundingDINO",
            segmentation_model=SAM2_MODELS.get(self._sam_model_name, self._sam_model_name),
            species_model=species_model_name,
            prompt=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            image_size=artifacts.image_size,
            crop_mode=crop_mode,
            total_instances=total_instances,
            tree_count=tree_count,
            shrub_count=shrub_count,
            additional={
                "multimask_output": multimask_output,
                "total_mask_area_px": total_mask_area,
                "species_top_k": species_top_k,
                "species_crop_padding": species_crop_padding,
                "species_output_dir": str(species_output_dir) if species_model is not None else None,
            } | (extra_general_info or {}),
        )

        report = DendroReport(general=metadata, instances=instances)
        report_filename = report_filename or f"{Path(image_path).stem}_report.json"
        report_path = Path(output_dir) / report_filename
        report.save(report_path)
        return report
