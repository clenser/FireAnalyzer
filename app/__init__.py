"""FlameAnalyzer inference backend.

A headless, cloud-ready pipeline:

    YOLO detection (OBJ_best.pt)        -> detection bounding box
        -> YOLO segmentation (SEG_best.pt) -> actual flame mask
        -> flame colour extraction (K-Means / GMM / Bayesian GMM / DBSCAN /
           Agglomerative clustering in CIELAB; MeanShift is not used)
        -> material matching against flame_dataset.json
        -> fire class + extinguishing agents (derived, documented mapping)
        -> JSON-serialisable result

No GUI, no local web server, no tunnels, no LLM calls, no disk I/O.
"""

from __future__ import annotations

from .analyzer import FlameAnalyzer
from .config import Settings
from .errors import (
    AnalysisError,
    InferenceError,
    InvalidImageError,
    MissingDatabaseError,
    MissingModelError,
    NoFireDetectedError,
    NoFlamePixelsError,
)
from .fire_classes import MATERIAL_FIRE_CLASS, classify
from .imaging import decode_image_bytes, imread, validate_image
from .mask import MASK_ENCODING, decode_mask_base64, mask_from_base64
from .schemas import AnalysisResult

__all__ = [
    "FlameAnalyzer",
    "Settings",
    "AnalysisResult",
    "AnalysisError",
    "InferenceError",
    "InvalidImageError",
    "MissingDatabaseError",
    "MissingModelError",
    "NoFireDetectedError",
    "NoFlamePixelsError",
    "MATERIAL_FIRE_CLASS",
    "classify",
    "MASK_ENCODING",
    "decode_mask_base64",
    "mask_from_base64",
    "decode_image_bytes",
    "imread",
    "validate_image",
]

__version__ = "3.0.0"
