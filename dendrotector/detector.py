from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import torch
import numpy as np
from PIL import Image

from groundingdino.util import box_ops
from groundingdino.util.inference import load_image, load_model, predict

from huggingface_hub import hf_hub_download

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2

from species_identifier import SpeciesIdentifier
from . import resolve_cache_dir, resolve_hf_cache_dir

PROMPT = "tree . shrub . bush ."

# Grounding DINO constants
GROUNDING_REPO    = "ShilongLiu/GroundingDINO"
GROUNDING_WEIGHTS = "groundingdino_swint_ogc.pth"
GROUNDING_CONFIG  = "GroundingDINO_SwinT_OGC.cfg.py"

SAM2_REPO = "facebook/sam2-hiera-large"


class DendroDetector:
    """High-level wrapper around GroundingDINO + SAM 2."""

    def __init__(
        self,
        device: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        models_dir: Union[Path, str, None] = None,
    ) -> None:

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self._models_dir = resolve_cache_dir(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._hf_cache_dir = resolve_hf_cache_dir(self._models_dir)

        self._dino_model = self._load_groundingdino()
        self._sam_predictor = self._load_sam2()
        self._species_identifier = SpeciesIdentifier(self.device, self._models_dir) # type: ignore

    ################
    # LOAD METHODS #
    ################

    def _load_groundingdino(self):
        groundingdino_dir = self._hf_cache_dir / "groundingdino"
        groundingdino_dir.mkdir(parents=True, exist_ok=True)

        config_path = hf_hub_download(
            GROUNDING_REPO,
            GROUNDING_CONFIG,
            cache_dir=str(groundingdino_dir),
        )
        weights_path = hf_hub_download(
            GROUNDING_REPO,
            GROUNDING_WEIGHTS,
            cache_dir=str(groundingdino_dir),
        )

        model = load_model(str(config_path), str(weights_path))

        model.to(self.device)
        model.eval()

        return model

    def _load_sam2(self) -> SAM2ImagePredictor:
        sam2_dir = self._hf_cache_dir / "sam2"
        sam2_dir.mkdir(parents=True, exist_ok=True)

        config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[SAM2_REPO]
        ckpt_path   = hf_hub_download(
            SAM2_REPO,
            filename=checkpoint_name,
            cache_dir=str(sam2_dir),
        )

        sam_model_instance = build_sam2(
            config_file=config_name,      
            ckpt_path=str(ckpt_path),
            device=str(self.device),      
        )

        return SAM2ImagePredictor(sam_model_instance)
    
    #############
    # BASIC API #
    #############

    def detect(
        self,
        image_path: os.PathLike[str] | str,
        output_dir: os.PathLike[str] | str,
        top_k: int,
        *,
        prompt: str = PROMPT,
        multimask_output: bool = False,
    ) -> List[Path]:
        """
        Запуск детекции деревьев/кустарников. Для КАЖДОГО instance создаётся
        отдельная папка в output_dir со следующими файлами:
        - overlay.png  (исходное изображение + маска и bbox данного instance)
        - mask.png     (только маска, RGBA -> сохраняется как BGRA для OpenCV)
        - bbox.png     (кроп по bbox)
        - report.json  (type: tree/shrub, score, bbox, lean_angle, species)
        Возвращает список путей к созданным папкам instance.
        """
        image_path = Path(image_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Детекция боксами GroundingDINO
        image_source, image_tensor = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=self._dino_model,
            image=image_tensor,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,  # type: ignore
        )

        # Если ничего не нашли — ничего не пишем
        if boxes.shape[0] == 0:
            return []

        # Приводим боксы к XYXY в пикселях
        h, w = image_source.shape[:2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes_xyxy *= torch.tensor([w, h, w, h], device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy.cpu().numpy()
        logits = logits.sigmoid().cpu().numpy()

        # 2) Маски SAM 2: получаем лучшую маску для каждого бокса
        self._sam_predictor.set_image(image_source)

        items: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, str]] = []
        # (area_ratio, box_xyxy, mask_bool, logit_vec, phrase)

        for box_xyxy, logit_vec, phrase in zip(boxes_xyxy, logits, phrases):
            masks_np, iou_predictions_np, _ = self._sam_predictor.predict(
                box=box_xyxy,
                multimask_output=multimask_output,
                normalize_coords=True,
            )
            # Выбор лучшей маски по IoU с подсказкой SAM2 (или первая)
            if multimask_output and masks_np.shape[0] > 1:
                best_idx = int(np.argmax(iou_predictions_np))
                best_mask = masks_np[best_idx]
            else:
                best_mask = masks_np[0]

            mask_bool = best_mask.astype(bool)
            area = int(mask_bool.sum())
            area_ratio = area / float(h * w) if h > 0 and w > 0 else 0.0
            items.append((area_ratio, box_xyxy, mask_bool, logit_vec, phrase))

        # 3) Сортируем инстансы по убыванию доли площади маски (крупные — первыми)
        items.sort(key=lambda t: t[0], reverse=True)

        created_instance_dirs: List[Path] = []

        # 4) Сохраняем артефакты для каждого инстанса
        for idx, (area_ratio, box_xyxy, mask_bool, logit_vec, phrase) in enumerate(items):
            # --- безопасный клэмп координат ---
            # Клэмп под кроп (эксклюзивный правый/нижний край) — минимум 1px по каждой оси
            x0_c = max(0, min(int(round(box_xyxy[0])), w - 1))
            y0_c = max(0, min(int(round(box_xyxy[1])), h - 1))
            x1_c = max(x0_c + 1, min(int(round(box_xyxy[2])), w))   # allow == w для слайсинга
            y1_c = max(y0_c + 1, min(int(round(box_xyxy[3])), h))   # allow == h для слайсинга

            # Клэмп под рисование рамки (инклюзивный правый/нижний край)
            x1_d = min(x1_c - 1, w - 1)
            y1_d = min(y1_c - 1, h - 1)

            # Директория instance (строго по ТЗ, без stem)
            instance_dir = output_dir / f"instance_{idx:02d}"
            instance_dir.mkdir(parents=True, exist_ok=True)

            # overlay.png — один бокс/маска поверх оригинала
            overlay_bgr = self._draw_overlay(
                image_source,
                np.asarray([[x0_c, y0_c, x1_d, y1_d]], dtype=np.float32),
                [mask_bool.astype(np.uint8)],
            )
            cv2.imwrite(str(instance_dir / "overlay.png"), overlay_bgr)

            # mask.png — RGBA (внутри _save_mask делается RGBA->BGRA для OpenCV)
            self._save_mask(mask_bool, instance_dir / "mask.png")

            # bbox.png — кроп (эксклюзивные границы)
            crop_rgb = image_source[y0_c:y1_c, x0_c:x1_c]
            if crop_rgb.size == 0:
                # На случай вырожденных границ
                crop_rgb = image_source[max(0, y0_c):max(0, y0_c + 1), max(0, x0_c):max(0, x0_c + 1)]
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            bbox_path = instance_dir / "bbox.png"
            cv2.imwrite(str(bbox_path), crop_bgr)

            # report.json — метаданные
            label = (phrase or "").strip() or "tree"
            instance_type = self._infer_instance_type(label)
            angle_deg = self._lean_angle(mask_bool)

            species, k = self._species_identifier.identify(bbox_path, top_k)

            report = {
                "type": instance_type,                           # "tree" | "shrub"
                "score": float(np.max(logit_vec)),               # уверенность DINO по этому боксу
                "bbox": [int(x0_c), int(y0_c), int(x1_c), int(y1_c)],
                "lean_angle": None if angle_deg is None else float(angle_deg),
                "top_k": k,                                      # фактический k после капа
                "species": species[0]["label"],
                "species_score": species[0]["prob"],
                "top_k_species": species
            }
            with (instance_dir / "report.json").open("w", encoding="utf-8") as fp:
                json.dump(report, fp, ensure_ascii=False, indent=2)

            created_instance_dirs.append(instance_dir)

        return created_instance_dirs

    
    ###########
    # HELPERS #
    ###########

    @staticmethod
    def _save_mask(mask: np.ndarray, path: Path) -> None:
        # Build RGBA logically (R, G, B, A)
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 34   # R
        rgba[..., 1] = 139  # G
        rgba[..., 2] = 34   # B
        rgba[..., 3] = np.where(mask, 200, 0)  # A

        # OpenCV expects BGRA
        bgra = rgba[..., [2, 1, 0, 3]]  # swap R<->B
        cv2.imwrite(str(path), bgra)

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
