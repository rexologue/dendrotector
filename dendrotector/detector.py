"""Instance segmentation pipeline for detecting trees and shrubs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, load_model, predict
from huggingface_hub import hf_hub_download
from segment_anything import SamPredictor, sam_model_registry

PROMPT = "tree . shrub . bush ."
GROUNDING_REPO = "ShilongLiu/GroundingDINO"
GROUNDING_CONFIG = "GroundingDINO_SwinT_OGC.py"
GROUNDING_WEIGHTS = "groundingdino_swint_ogc.pth"
SAM_REPO = "facebook/sam"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"


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


class DendroDetector:
    """High-level wrapper around GroundingDINO + Segment Anything."""

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
        self._sam_predictor = self._load_sam(sam_model)

    def _load_groundingdino(self):
        download_kwargs = self._download_kwargs("groundingdino")
        config_path = hf_hub_download(GROUNDING_REPO, GROUNDING_CONFIG, **download_kwargs)
        weights_path = hf_hub_download(GROUNDING_REPO, GROUNDING_WEIGHTS, **download_kwargs)
        model = load_model(config_path, weights_path)
        model.to(self.device)
        model.eval()
        return model

    def _load_sam(self, sam_model: str) -> SamPredictor:
        download_kwargs = self._download_kwargs("sam")
        checkpoint_path = hf_hub_download(SAM_REPO, SAM_CHECKPOINT, **download_kwargs)
        sam = sam_model_registry[sam_model](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return SamPredictor(sam)

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
    ) -> List[DetectionResult]:
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
            If True, SAM will return up to three mask proposals per box; the
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
            return []

        h, w = image_source.shape[:2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes_xyxy *= torch.tensor([w, h, w, h], device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy.cpu()

        self._sam_predictor.set_image(image_source)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_source.shape[:2]
        )

        masks, iou_predictions, _ = self._sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=multimask_output,
        )

        masks = masks.cpu().numpy()
        iou_predictions = iou_predictions.cpu().numpy()
        boxes_xyxy = boxes_xyxy.numpy()
        logits = logits.sigmoid().cpu().numpy()

        overlay = self._draw_overlay(image_source, boxes_xyxy, masks)
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        results: List[DetectionResult] = []
        for idx, (box, mask, logit, phrase) in enumerate(
            zip(boxes_xyxy, masks, logits, phrases)
        ):
            best_mask = mask[0]
            if multimask_output and mask.shape[0] > 1:
                best_idx = int(np.argmax(iou_predictions[idx]))
                best_mask = mask[best_idx]

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

        return results

    @staticmethod
    def _save_mask(mask: np.ndarray, path: Path) -> None:
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 34
        rgba[..., 1] = 139
        rgba[..., 2] = 34
        rgba[..., 3] = np.where(mask, 200, 0)
        cv2.imwrite(str(path), rgba)

    @staticmethod
    def _draw_overlay(image: np.ndarray, boxes: np.ndarray, masks: np.ndarray) -> np.ndarray:
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
