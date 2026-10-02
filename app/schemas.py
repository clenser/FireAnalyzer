"""Response schemas for the FlameAnalyzer backend.

These dataclasses define the *only* public data contract.  ``to_dict`` returns
plain Python types (lists/floats/ints/bools/str) so the result can always be
handed straight to :func:`json.dumps` - no NumPy arrays ever escape the API.

Two naming details are deliberate:

* ``FireClassResult.class_`` is serialised as ``class`` (``class`` is a Python
  keyword, so the dataclass field needs a different name).  The rename happens
  in :meth:`AnalysisResult.to_dict`.
* ``fire_detection.bounding_box`` and ``segmentation.fallback_used`` are the
  project's original field names and are kept.  ``AnalysisResult.to_dict``
  additionally emits the shorter aliases ``bbox`` and ``fallback`` so a consumer
  that expects either spelling keeps working.
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
    "ClusterCentroid",
    "ClusteringResult",
    "FlameAnalysis",
    "MaterialScore",
    "MaterialAnalysis",
    "SuppressionInformation",
    "FireClassResult",
    "ExtinguishingAgent",
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
    """Axis-aligned fire bounding box in absolute pixel coordinates.

    This is the *detection* rectangle.  It is not the flame area - the segmented
    region is reported separately as ``SegmentationSummary.mask``.
    """

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
    """Result of the flame segmentation stage, including the mask itself.

    ``flame_pixel_count`` and ``mask_area_ratio`` are always measured from the
    mask, never from the bounding box.  When ``mask`` is set it is a base64 PNG
    whose decoded size is exactly ``mask_width`` x ``mask_height``, i.e. the
    analysed image's dimensions, so a client can overlay it directly.
    """

    available: bool
    fallback_used: bool
    flame_pixel_count: int
    mask_area_ratio: float
    confidence: float | None = None
    mask_width: int = 0
    mask_height: int = 0
    mask_encoding: str | None = None
    mask: str | None = None
    #: True when ``mask`` was produced by rasterising the detection box because
    #: the segmentation model was unavailable or returned nothing usable.
    bbox_fallback_reason: str | None = None
    #: Number of per-detection masks that contributed to ``mask``.
    mask_count: int = 0
    #: True when ``mask`` is the union of the per-detection masks (always the
    #: case for the multi-detection pipeline, even with a single detection).
    merged: bool = False
    #: One entry per detection, in detection order: where its mask came from
    #: (``segmentation`` / ``bbox_fallback`` / ``none``), the segmentation
    #: confidence and the pixel count of that individual mask.
    detection_masks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FlameColor:
    """A representative flame colour, serialised in both RGB and LAB."""

    rgb: list[int] = field(default_factory=list)
    lab: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ClusterCentroid:
    """One cluster of one algorithm: its centroid, size and share of the samples."""

    index: int
    size: int
    weight: float
    rgb: list[int] = field(default_factory=list)
    lab: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ClusteringResult:
    """A representative colour produced by one clustering algorithm.

    ``rgb``/``lab`` are the algorithm's representative colour, ``dominant_color``
    is the centroid of the dominant cluster, and ``centroids`` lists every
    cluster so the dominant-cluster extraction is inspectable.
    """

    rgb: list[int]
    lab: list[float]
    method: str
    cluster_count: int
    samples_used: int
    pixels_sampled: bool
    dominant_cluster: int = 0
    dominant_color: FlameColor = field(default_factory=FlameColor)
    centroids: list[ClusterCentroid] = field(default_factory=list)
    noise_count: int = 0
    #: How the representative colour was derived from the clusters.
    representative: str = ""
    #: True when the algorithm had to fall back (e.g. DBSCAN found no cluster).
    fallback: bool = False


@dataclass(frozen=True)
class FlameAnalysis:
    """Colour analysis of the segmented flame pixels.

    One entry per clustering algorithm that ran.  All five are always present
    for a mask with enough pixels: ``kmeans``, ``gmm``, ``bayesian_gmm``,
    ``dbscan`` and ``agglomerative``.  MeanShift is not used.
    """

    kmeans: ClusteringResult | None = None
    gmm: ClusteringResult | None = None
    bayesian_gmm: ClusteringResult | None = None
    dbscan: ClusteringResult | None = None
    agglomerative: ClusteringResult | None = None
    mean_color: FlameColor = field(default_factory=FlameColor)
    flame_pixel_count: int = 0
    samples_used: int = 0
    pixels_sampled: bool = False
    #: The ``k`` the k-based algorithms used.
    n_clusters: int = 0
    #: Names of the algorithms that produced a colour, in report order.
    algorithms: list[str] = field(default_factory=list)
    #: Methods that did not run because the mask was too small.
    skipped_reason: str | None = None


@dataclass(frozen=True)
class MaterialScore:
    """A database material together with its heuristic similarity score."""

    material: str
    similarity: float


@dataclass(frozen=True)
class MaterialAnalysis:
    """Deterministic, database-derived material identification.

    This is supporting evidence: the fire class derived from it is the primary
    safety classification.  ``similarity`` is a distance-based score in
    ``[0, 1]``; it is *not* a calibrated probability.
    """

    primary_material: str | None
    similarity: float
    alternatives: list[MaterialScore]
    database_notes: str | None
    score_basis: str = "mean LAB distance to flame_dataset.json reference colours"


@dataclass(frozen=True)
class SuppressionInformation:
    """Suppression data copied verbatim from ``flame_dataset.json``.

    Kept for backward compatibility with existing API consumers; the structured
    form is :class:`ExtinguishingAgent` in ``extinguishing_agents``.
    """

    source: str
    material: str | None
    methods: list[str]
    database_notes: str | None


@dataclass(frozen=True)
class FireClassResult:
    """The primary safety classification.

    ``class_`` is serialised as ``class``.  This value is **derived** from the
    material analysis through the documented mapping in
    :mod:`app.fire_classes`; the detection model does not predict it, and
    ``confidence`` is the material match's distance-based similarity rather than
    a calibrated probability.
    """

    class_: str
    description: str = ""
    confidence: float = 0.0
    material: str | None = None
    basis: str = ""
    mapping_source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ExtinguishingAgent:
    """One extinguishing agent associated with the determined fire class.

    ``name`` is the string recorded in ``flame_dataset.json``.  ``compound`` is
    always ``None`` for that reason: the dataset stores agent names, not
    chemical identities, and none are invented here.  ``type`` is the agent's
    suppression mechanism category.
    """

    name: str
    compound: str | None = None
    type: str = "unspecified"
    source: str = "flame_dataset.json"
    fire_class: str | None = None
    compound_basis: str = ""


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
    fire_class: FireClassResult | None = None
    extinguishing_agents: list[ExtinguishingAgent] = field(default_factory=list)
    timing: AnalysisTiming | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a ``dict`` that :func:`json.dumps` accepts as-is."""
        payload = jsonable(dataclasses.asdict(self))

        fire_class = payload.get("fire_class")
        if isinstance(fire_class, dict) and "class_" in fire_class:
            fire_class["class"] = fire_class.pop("class_")

        detection = payload.get("fire_detection")
        if isinstance(detection, dict) and detection.get("bounding_box") is not None:
            # Short alias; ``bounding_box`` stays the documented name.
            detection["bbox"] = detection["bounding_box"]

        segmentation = payload.get("segmentation")
        if isinstance(segmentation, dict):
            segmentation["fallback"] = bool(segmentation.get("fallback_used"))

        return payload
