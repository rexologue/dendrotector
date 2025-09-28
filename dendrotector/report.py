"""Structured reporting utilities for the DendroDetector pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .detector import DetectionResult


@dataclass
class SpeciesSummary:
    """Summary of the species classifier output for a single instance."""

    label: str
    score: float
    top_k: Sequence[tuple[str, float]]
    model_name: str
    crop_path: Path | None = None
    crop_size: tuple[int, int] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": float(self.score),
            "top_k": [
                {"label": candidate_label, "score": float(candidate_score)}
                for candidate_label, candidate_score in self.top_k
            ],
            "model": self.model_name,
            "crop_path": str(self.crop_path) if self.crop_path is not None else None,
            "crop_size": list(self.crop_size) if self.crop_size is not None else None,
        }


@dataclass
class InstanceReport:
    """Detailed report for a single detected tree or shrub instance."""

    index: int
    detection: DetectionResult
    instance_type: str
    mask_path: Path
    mask_area_px: int | None
    lean_angle_degrees: float | None
    species: SpeciesSummary | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "type": self.instance_type,
            "bbox": self.detection.bbox,
            "detection_label": self.detection.label,
            "detection_score": float(self.detection.score),
            "mask_path": str(self.mask_path),
            "mask_area_px": self.mask_area_px,
            "lean_angle_degrees": self.lean_angle_degrees,
            "species": self.species.to_json() if self.species is not None else None,
        }
        for key, value in self.extra.items():
            if key in payload:
                continue
            payload[key] = value
        return payload


@dataclass
class GeneralInfo:
    """Global metadata describing a detector run."""

    image_path: Path
    overlay_path: Path
    detection_metadata_path: Path
    detection_model: str
    segmentation_model: str
    species_model: str | None
    prompt: str
    box_threshold: float
    text_threshold: float
    generated_at: str
    image_size: tuple[int, int]
    crop_mode: str
    total_instances: int
    tree_count: int
    shrub_count: int
    additional: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_path": str(self.image_path),
            "overlay_path": str(self.overlay_path),
            "detection_metadata_path": str(self.detection_metadata_path),
            "detection_model": self.detection_model,
            "segmentation_model": self.segmentation_model,
            "species_model": self.species_model,
            "prompt": self.prompt,
            "box_threshold": float(self.box_threshold),
            "text_threshold": float(self.text_threshold),
            "generated_at": self.generated_at,
            "image_size": list(self.image_size),
            "crop_mode": self.crop_mode,
            "total_instances": int(self.total_instances),
            "tree_count": int(self.tree_count),
            "shrub_count": int(self.shrub_count),
        }
        for key, value in self.additional.items():
            if key in payload:
                continue
            payload[key] = value
        return payload


@dataclass
class DendroReport:
    """Aggregate report produced by :class:`DendroDetector.generate_report`."""

    general: GeneralInfo
    instances: Sequence[InstanceReport]
    report_path: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "general": self.general.to_json(),
            "instances": [instance.to_json() for instance in self.instances],
        }

    def save(self, path: Path | str) -> Path:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.to_json(), fp, indent=2)
        self.report_path = path
        return path

