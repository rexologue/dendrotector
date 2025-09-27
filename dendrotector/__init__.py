"""High-level interface for dendrotector instance segmentation and taxonomy."""

from .detector import DendroDetector, DetectionResult
from .species_config import MOSCOW_REGION_SPECIES, SpeciesDefinition
from .species_identifier import SpeciesIdentifier, SpeciesPrediction, load_detections

__all__ = [
    "DendroDetector",
    "DetectionResult",
    "SpeciesIdentifier",
    "SpeciesPrediction",
    "load_detections",
    "SpeciesDefinition",
    "MOSCOW_REGION_SPECIES",
]
