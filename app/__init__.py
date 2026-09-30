"""FlameAnalyzer inference backend.

A headless, cloud-ready pipeline:

    YOLO detection (OBJ_best.pt)
        -> YOLO segmentation (SEG_best.pt)
        -> flame colour extraction (K-Means / GMM in CIELAB)
        -> material matching against flame_dataset.json
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
from .imaging import decode_image_bytes, imread, validate_image
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
    "decode_image_bytes",
    "imread",
    "validate_image",
]

__version__ = "2.0.0"
