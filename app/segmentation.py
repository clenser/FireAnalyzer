"""Flame segmentation stage - wraps the custom ``SEG_best.pt`` YOLO model.

Behaviour preserved from the original implementation: the highest-confidence
mask wins, and if the segmentation model is unavailable / fails / returns an
empty mask, the detected bounding box is rasterised into a mask instead.  The
fallback is always reported explicitly in the API response.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from .config import Settings
from .schemas import SegmentationSummary

__all__ = ["segment_fire", "bounding_box_mask", "summarize_mask"]

logger = logging.getLogger(__name__)

BoxLike = Sequence[float]


def bounding_box_mask(image_shape: Sequence[int], boxes: Sequence[BoxLike]) -> np.ndarray:
    """Rasterise ``[x1, y1, x2, y2]`` boxes into a ``uint8`` mask (0/255).

    Coordinates are clipped to the image bounds, so a box that lies partially
    (or fully) outside the frame still yields the visible part instead of an
    inverted slice.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
        y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def _resize_mask(mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Resize a binary ``uint8`` mask to ``shape`` (nearest neighbour)."""
    height, width = int(shape[0]), int(shape[1])
    if mask.shape[:2] == (height, width):
        return mask
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8) * 255


def summarize_mask(mask: np.ndarray | None) -> tuple[int, float]:
    """Return ``(flame_pixel_count, mask_area_ratio)`` for a mask."""
    if mask is None or mask.size == 0:
        return 0, 0.0
    binary = mask > 0
    count = int(np.count_nonzero(binary))
    ratio = count / float(binary.size)
    return count, round(float(ratio), 6)


def _mask_confidences(result, count: int) -> list[float]:
    """Per-mask confidences for a segmentation result.

    Falls back to a neutral ``0.5`` when the result carries no usable box
    confidences, which keeps every mask eligible instead of dropping them.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) != count:
        return [0.5] * count
    return [float(value) for value in np.asarray(boxes.conf.cpu().numpy()).ravel()]


def segment_fire(
    model,
    image_bgr: np.ndarray,
    boxes: Sequence[BoxLike],
    settings: Settings,
) -> tuple[np.ndarray, SegmentationSummary]:
    """Produce a flame mask and a JSON-ready summary of how it was obtained.

    Parameters
    ----------
    model:
        Loaded ``ultralytics.YOLO`` segmentation model, or ``None``.
    image_bgr:
        Input image as an OpenCV BGR array.
    boxes:
        Detected boxes used for the fallback mask.
    settings:
        Active configuration.

    Returns
    -------
    tuple[np.ndarray, SegmentationSummary]
        ``(mask, summary)`` where ``summary.fallback_used`` tells the caller
        whether the bounding-box fallback produced the mask.
    """
    original_shape = image_bgr.shape[:2]
    best_mask: np.ndarray | None = None
    best_conf = -1.0

    if model is not None:
        try:
            results = model(
                image_bgr,
                imgsz=settings.imgsz,
                conf=settings.segmentation_conf,
                device=settings.device,
                retina_masks=settings.segmentation_retina_masks,
                verbose=settings.ultralytics_verbose,
            )
            for result in results:
                masks = getattr(result, "masks", None)
                if masks is None or len(masks) == 0:
                    continue
                mask_array = masks.data.cpu().numpy()
                confidences = _mask_confidences(result, len(masks))
                for index, raw_mask in enumerate(mask_array):
                    confidence = confidences[index] if index < len(confidences) else 0.5
                    if confidence <= best_conf:
                        continue
                    best_conf = confidence
                    best_mask = _resize_mask((raw_mask > 0.5).astype(np.uint8), original_shape)
        except Exception:  # noqa: BLE001 - segmentation failure must degrade, not crash
            logger.exception("Segmentation model failed; falling back to bounding-box mask")
            best_mask = None
    else:
        logger.warning("No segmentation model loaded; using bounding-box mask fallback")

    if best_mask is not None and np.count_nonzero(best_mask) > 0:
        count, ratio = summarize_mask(best_mask)
        return best_mask, SegmentationSummary(
            available=True,
            fallback_used=False,
            flame_pixel_count=count,
            mask_area_ratio=ratio,
            confidence=round(best_conf, 4) if best_conf > 0 else None,
        )

    if not settings.fallback_to_bbox_mask or not boxes:
        count, ratio = summarize_mask(best_mask)
        return (
            best_mask if best_mask is not None else np.zeros(original_shape, dtype=np.uint8),
            SegmentationSummary(
                available=False,
                fallback_used=False,
                flame_pixel_count=count,
                mask_area_ratio=ratio,
            ),
        )

    mask = bounding_box_mask(original_shape, boxes)
    count, ratio = summarize_mask(mask)
    return mask, SegmentationSummary(
        available=False,
        fallback_used=True,
        flame_pixel_count=count,
        mask_area_ratio=ratio,
    )
