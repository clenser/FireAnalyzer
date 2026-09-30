"""Fire detection stage - wraps the custom ``OBJ_best.pt`` YOLO model.

Behaviour preserved from the original implementation: only detections of the
fire class above the confidence threshold are considered and the single
highest-confidence detection is kept.
"""

from __future__ import annotations

import logging

from .config import Settings
from .schemas import BoundingBox, FireDetection

__all__ = ["detect_fire", "boxes_from_detection"]

logger = logging.getLogger(__name__)


def detect_fire(model, image_bgr, settings: Settings) -> FireDetection:
    """Run fire detection and return the best-scoring detection.

    Parameters
    ----------
    model:
        A loaded ``ultralytics.YOLO`` detection model.
    image_bgr:
        Input image as an OpenCV BGR array.
    settings:
        Active configuration (image size, thresholds, class id).

    Returns
    -------
    FireDetection
        ``detected=False`` when nothing passes the confidence threshold.
    """
    if model is None:
        raise RuntimeError("detect_fire() called without a loaded detection model")

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

    if not candidates:
        return FireDetection(detected=False, confidence=0.0, bounding_box=None)

    # Keep the highest-confidence detection, exactly as the original pipeline did.
    confidence, (x1, y1, x2, y2) = max(candidates, key=lambda item: item[0])
    return FireDetection(
        detected=True,
        confidence=round(float(confidence), 4),
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def boxes_from_detection(detection: FireDetection) -> list[list[int]]:
    """Return ``[[x1, y1, x2, y2]]`` for the detection, or ``[]`` if none."""
    if not detection.detected or detection.bounding_box is None:
        return []
    return [list(detection.bounding_box.as_tuple())]
