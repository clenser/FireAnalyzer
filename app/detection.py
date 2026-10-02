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

#: IoU above which two surviving boxes are treated as the same flame region
#: rather than two detections.  The model's own NMS already separates
#: distinct flames; this is a safety net for the rare near-identical
#: duplicate that slips through, so it is set deliberately high - high enough
#: that two genuinely separate (even adjacent or partially overlapping)
#: flames are never collapsed into one.
_DUPLICATE_IOU = 0.92


def _iou_xyxy(a: list[int], b: list[int]) -> float:
    """IoU between two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _drop_duplicate_boxes(
    candidates: list[tuple[float, list[int]]]
) -> list[tuple[float, list[int]]]:
    """Drop boxes that are near-duplicates of an already-kept, higher-confidence one.

    ``candidates`` must already be sorted highest-confidence first, so the
    first box seen at a given location is always the one kept.
    """
    kept: list[tuple[float, list[int]]] = []
    for confidence, box in candidates:
        if any(_iou_xyxy(box, kept_box) >= _DUPLICATE_IOU for _, kept_box in kept):
            continue
        kept.append((confidence, box))
    return kept


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
    height, width = image_bgr.shape[:2]
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
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            # Clip to the frame: letterbox/rounding artifacts must never hand
            # the API an out-of-bounds coordinate.
            x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
            y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
            if x2 <= x1 or y2 <= y1:
                # Zero/negative-area box: not a real detection.
                continue
            candidates.append((confidence, [x1, y1, x2, y2]))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    candidates = _drop_duplicate_boxes(candidates)
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
