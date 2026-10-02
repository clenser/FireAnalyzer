"""Fire detection stage - wraps the custom ``OBJ_best.pt`` YOLO model.

Only detections of the fire class above the confidence threshold are
considered.  Up to :data:`MAX_DETECTIONS` of them are kept, ordered by
confidence (highest first).  Boxes are never merged: each detection keeps its
own rectangle and later receives its own segmentation mask.

:func:`detect_fire` keeps the original single-detection contract (the best
box) for callers that only need one.
"""

from __future__ import annotations

import logging

from .config import Settings
from .schemas import BoundingBox, FireDetection

__all__ = ["MAX_DETECTIONS", "detect_fires", "detect_fire", "boxes_from_detection"]

logger = logging.getLogger(__name__)


#: Hard upper bound on the number of flame detections kept per image/frame.
MAX_DETECTIONS = 3


def detect_fires(model, image_bgr, settings: Settings, max_detections: int = MAX_DETECTIONS) -> list[FireDetection]:
    """Run fire detection and return up to ``max_detections`` detections.

    Parameters
    ----------
    model:
        A loaded ``ultralytics.YOLO`` detection model.
    image_bgr:
        Input image as an OpenCV BGR array.
    settings:
        Active configuration (image size, thresholds, class id).
    max_detections:
        Upper bound, clamped to ``1..MAX_DETECTIONS``.

    Returns
    -------
    list[FireDetection]
        Highest confidence first; empty when nothing passes the threshold.
        Ties are broken by box position so the order is deterministic.
    """
    if model is None:
        raise RuntimeError("detect_fires() called without a loaded detection model")

    limit = max(1, min(MAX_DETECTIONS, int(max_detections)))
    results = model(
        image_bgr,
        imgsz=settings.imgsz,
        conf=settings.detection_conf,
        device=settings.device,
        verbose=settings.ultralytics_verbose,
    )

    candidates: list[tuple[float, list[int]]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            if class_id != settings.detection_class_id:
                continue
            if confidence < settings.detection_conf:
                continue
            candidates.append((confidence, [int(round(float(v))) for v in xyxy]))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        FireDetection(
            detected=True,
            confidence=round(float(confidence), 4),
            bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        )
        for confidence, (x1, y1, x2, y2) in candidates[:limit]
    ]


def detect_fire(model, image_bgr, settings: Settings) -> FireDetection:
    """Single best detection (original contract); ``detected=False`` when none."""
    detections = detect_fires(model, image_bgr, settings, max_detections=1)
    if not detections:
        return FireDetection(detected=False, confidence=0.0, bounding_box=None)
    return detections[0]


def boxes_from_detection(detection: FireDetection) -> list[list[int]]:
    """Return ``[[x1, y1, x2, y2]]`` for the detection, or ``[]`` if none."""
    if not detection.detected or detection.bounding_box is None:
        return []
    return [list(detection.bounding_box.as_tuple())]
