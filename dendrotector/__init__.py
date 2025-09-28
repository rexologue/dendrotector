"""High-level interface for dendrotector instance segmentation and taxonomy."""

from .detector import DendroDetector, DetectionArtifacts, DetectionResult
from .species_config import MOSCOW_REGION_SPECIES, SpeciesDefinition
from .species_identifier import SpeciesIdentifier, SpeciesPrediction, load_detections
from .report import DendroReport, GeneralInfo, InstanceReport, SpeciesSummary

__all__ = [
    "DendroDetector",
    "DetectionResult",
    "DetectionArtifacts",
    "SpeciesIdentifier",
    "SpeciesPrediction",
    "load_detections",
    "SpeciesDefinition",
    "MOSCOW_REGION_SPECIES",
    "DendroReport",
    "GeneralInfo",
    "InstanceReport",
    "SpeciesSummary",
]
