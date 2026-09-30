"""Response schemas for the FlameAnalyzer backend.

These dataclasses define the *only* public data contract.  ``to_dict`` returns
plain Python types (lists/floats/ints/bools/str) so the result can always be
handed straight to :func:`json.dumps` - no NumPy arrays ever escape the API.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "BoundingBox",
    "FireDetection",
    "SegmentationSummary",
    "FlameColor",
    "ClusteringResult",
    "FlameAnalysis",
    "MaterialScore",
    "MaterialAnalysis",
    "SuppressionInformation",
    "AnalysisTiming",
    "AnalysisResult",
    "jsonable",
]

_RGB_LAB_DECIMALS = 2
_SIMILARITY_DECIMALS = 4
_RATIO_DECIMALS = 6


def jsonable(value: Any) -> Any:
    """Recursively convert NumPy containers/scalars into JSON-native types.

    ``np.ndarray`` becomes ``list``, ``np.floating``/``np.integer`` become
    ``float``/``int``, and tuples/sets become lists.  Everything else is
    returned unchanged.
    """
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned fire bounding box in absolute pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class FireDetection:
    """Result of the fire detection stage."""

    detected: bool
    confidence: float
    bounding_box: BoundingBox | None = None


@dataclass(frozen=True)
class SegmentationSummary:
    """Result of the flame segmentation stage (the mask itself is not returned)."""

    available: bool
    fallback_used: bool
    flame_pixel_count: int
    mask_area_ratio: float
    confidence: float | None = None


@dataclass(frozen=True)
class FlameColor:
    """A representative flame colour, serialised in both RGB and LAB."""

    rgb: list[int] = field(default_factory=list)
    lab: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ClusteringResult:
    """A representative colour produced by one clustering algorithm."""

    rgb: list[int]
    lab: list[float]
    method: str
    cluster_count: int
    samples_used: int
    pixels_sampled: bool


@dataclass(frozen=True)
class FlameAnalysis:
    """Colour analysis of the segmented flame pixels."""

    kmeans: ClusteringResult | None
    gmm: ClusteringResult | None
    mean_color: FlameColor
    flame_pixel_count: int
    samples_used: int
    pixels_sampled: bool


@dataclass(frozen=True)
class MaterialScore:
    """A database material together with its heuristic similarity score."""

    material: str
    similarity: float


@dataclass(frozen=True)
class MaterialAnalysis:
    """Deterministic, database-derived material identification.

    ``similarity`` is a distance-based score in ``[0, 1]``; it is *not* a
    calibrated probability.
    """

    primary_material: str | None
    similarity: float
    alternatives: list[MaterialScore]
    database_notes: str | None
    score_basis: str = "mean LAB distance to flame_dataset.json reference colours"


@dataclass(frozen=True)
class SuppressionInformation:
    """Suppression data copied verbatim from ``flame_dataset.json``."""

    source: str
    material: str | None
    methods: list[str]
    database_notes: str | None


@dataclass(frozen=True)
class AnalysisTiming:
    """Per-stage wall-clock timings, useful for cloud observability."""

    total_ms: float
    detection_ms: float
    segmentation_ms: float
    color_ms: float
    material_ms: float


@dataclass(frozen=True)
class AnalysisResult:
    """Full, JSON-serialisable analysis result."""

    success: bool
    fire_detection: FireDetection
    segmentation: SegmentationSummary
    flame_analysis: FlameAnalysis | None = None
    material_analysis: MaterialAnalysis | None = None
    suppression_information: SuppressionInformation | None = None
    timing: AnalysisTiming | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a ``dict`` that :func:`json.dumps` accepts as-is."""
        return jsonable(dataclasses.asdict(self))
