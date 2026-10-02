"""Flame segmentation stage - wraps the custom ``SEG_best.pt`` YOLO model.

Behaviour preserved from the original implementation: the highest-confidence
mask wins, and if the segmentation model is unavailable / fails / returns an
empty mask, the detected bounding box is rasterised into a mask instead.  The
fallback is always reported explicitly in the API response.

The mask itself is returned in two forms:

* as a ``uint8`` array, which the colour stage consumes directly, and
* as a base64 PNG inside :class:`~app.schemas.SegmentationSummary`, so the API
  can hand the frontend the actual segmented flame region rather than a
  bounding box.

The mask is produced by the single segmentation inference pass - YOLO is never
run a second time just to obtain the mask.
"""

from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np

from .config import Settings
from .mask import MASK_ENCODING, encode_mask_base64
from .schemas import SegmentationSummary

__all__ = ["segment_fire", "segment_detections", "bounding_box_mask", "summarize_mask"]

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


def _summary(
    mask: np.ndarray | None,
    *,
    available: bool,
    fallback_used: bool,
    confidence: float | None,
    settings: Settings,
    bbox_fallback_reason: str | None = None,
    mask_count: int | None = None,
    merged: bool = False,
    detection_masks: list[dict] | None = None,
) -> SegmentationSummary:
    """Build the JSON-ready summary, attaching the encoded mask when enabled.

    ``flame_pixel_count`` and ``mask_area_ratio`` are always measured from the
    mask itself, never from the bounding box, so they describe the same pixels
    the frontend will draw.
    """
    count, ratio = summarize_mask(mask)
    height, width = (int(mask.shape[0]), int(mask.shape[1])) if mask is not None else (0, 0)

    payload: str | None = None
    encoding: str | None = None
    if settings.emit_mask and count > 0:
        payload = encode_mask_base64(mask, settings.mask_png_compression)
        encoding = MASK_ENCODING if payload is not None else None

    return SegmentationSummary(
        available=available,
        fallback_used=fallback_used,
        flame_pixel_count=count,
        mask_area_ratio=ratio,
        confidence=confidence,
        mask_width=width,
        mask_height=height,
        mask_encoding=encoding,
        mask=payload,
        bbox_fallback_reason=bbox_fallback_reason,
        mask_count=int(mask_count if mask_count is not None else (1 if count > 0 else 0)),
        merged=merged,
        detection_masks=list(detection_masks or []),
    )


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
    model_ran = False

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
            model_ran = True
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
            model_ran = False
    else:
        logger.warning("No segmentation model loaded; using bounding-box mask fallback")

    if best_mask is not None and np.count_nonzero(best_mask) > 0:
        return best_mask, _summary(
            best_mask,
            available=True,
            fallback_used=False,
            confidence=round(best_conf, 4) if best_conf > 0 else None,
            settings=settings,
        )

    reason = (
        "no segmentation model is loaded"
        if not model_ran
        else "the segmentation model returned no usable mask"
    )

    if not settings.fallback_to_bbox_mask or not boxes:
        empty = best_mask if best_mask is not None else np.zeros(original_shape, dtype=np.uint8)
        return empty, _summary(
            empty,
            available=False,
            fallback_used=False,
            confidence=None,
            settings=settings,
            bbox_fallback_reason=reason,
        )

    mask = bounding_box_mask(original_shape, boxes)
    return mask, _summary(
        mask,
        available=False,
        fallback_used=True,
        confidence=None,
        settings=settings,
        bbox_fallback_reason=reason,
    )


# ---------------------------------------------------------------------------
# Multi-detection segmentation
# ---------------------------------------------------------------------------
def _box_inside_fraction(mask: np.ndarray, box: BoxLike) -> float:
    """Share of a mask's pixels that fall inside ``box`` (0 when the mask is empty)."""
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    total = int(np.count_nonzero(mask))
    if total == 0 or x2 <= x1 or y2 <= y1:
        return 0.0
    return int(np.count_nonzero(mask[y1:y2, x1:x2])) / float(total)


def _segmentation_instances(
    model, image_bgr: np.ndarray, settings: Settings
) -> tuple[list[tuple[float, np.ndarray]], bool]:
    """Run the segmentation model once; return ``([(confidence, mask)], ran)``.

    Masks are binary ``uint8`` arrays at the image resolution.  Any model failure
    degrades to ``([], False)`` so the caller can fall back to box masks.
    """
    if model is None:
        logger.warning("No segmentation model loaded; using bounding-box mask fallback")
        return [], False
    original_shape = image_bgr.shape[:2]
    instances: list[tuple[float, np.ndarray]] = []
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
                mask = _resize_mask((raw_mask > 0.5).astype(np.uint8), original_shape)
                if np.count_nonzero(mask) > 0:
                    instances.append((float(confidence), mask))
    except Exception:  # noqa: BLE001 - segmentation failure must degrade, not crash
        logger.exception("Segmentation model failed; falling back to bounding-box masks")
        return [], False
    return instances, True


def segment_detections(
    model,
    image_bgr: np.ndarray,
    boxes: Sequence[BoxLike],
    settings: Settings,
) -> tuple[np.ndarray, SegmentationSummary, list[np.ndarray | None]]:
    """One mask per detection, merged into a single binary flame mask.

    The segmentation model runs **once** on the image.  Each detection box is
    then paired with the highest-confidence segmentation instance that overlaps
    it (ties broken by the larger share of that instance inside the box).  A
    detection with no overlapping instance gets its rasterised box as mask when
    ``fallback_to_bbox_mask`` is enabled.  Boxes themselves are never merged;
    only the masks are unioned, and the union drives every downstream
    measurement.

    With exactly one detection and no overlapping instance, the globally best
    instance is used - the original single-detection behaviour.

    Returns
    -------
    tuple
        ``(merged_mask, summary, per_detection_masks)``.  ``per_detection_masks``
        is index-aligned with ``boxes`` (``None`` where no mask was produced),
        preserving the detection-to-mask correspondence.
    """
    original_shape = image_bgr.shape[:2]
    instances, model_ran = _segmentation_instances(model, image_bgr, settings)

    per_detection: list[np.ndarray | None] = []
    records: list[dict] = []
    used_confidences: list[float] = []
    any_fallback = False
    for index, box in enumerate(boxes):
        choice: tuple[float, np.ndarray] | None = None
        best_key: tuple[float, float] | None = None
        for confidence, mask in instances:
            inside = _box_inside_fraction(mask, box)
            if inside <= 0.0:
                continue
            key = (confidence, inside)
            if best_key is None or key > best_key:
                best_key, choice = key, (confidence, mask)
        if choice is None and len(boxes) == 1 and instances:
            choice = max(instances, key=lambda item: item[0])

        mask_conf: float | None = None
        if choice is not None:
            confidence, mask = choice
            per_detection.append(mask)
            used_confidences.append(confidence)
            source = "segmentation"
            mask_conf = round(confidence, 4)
        elif settings.fallback_to_bbox_mask:
            mask = bounding_box_mask(original_shape, [box])
            has_pixels = bool(np.count_nonzero(mask))
            per_detection.append(mask if has_pixels else None)
            any_fallback = any_fallback or has_pixels
            source = "bbox_fallback" if has_pixels else "none"
        else:
            per_detection.append(None)
            source = "none"
        pixels, _ratio = summarize_mask(per_detection[-1])
        records.append(
            {
                "detection_index": index,
                "mask_source": source,
                "segmentation_confidence": mask_conf,
                "mask_pixel_count": pixels,
            }
        )

    merged = np.zeros(original_shape, dtype=np.uint8)
    valid = [mask for mask in per_detection if mask is not None]
    for mask in valid:
        merged = np.maximum(merged, (mask > 0).astype(np.uint8) * 255)

    if used_confidences:
        reason = (
            "the segmentation model returned no mask for some detections"
            if any_fallback
            else None
        )
    elif not model_ran:
        reason = "no segmentation model is loaded or it failed"
    else:
        reason = "the segmentation model returned no usable mask"

    summary = _summary(
        merged,
        available=bool(used_confidences),
        fallback_used=any_fallback,
        confidence=round(max(used_confidences), 4) if used_confidences else None,
        settings=settings,
        bbox_fallback_reason=reason,
        mask_count=len(valid),
        merged=True,
        detection_masks=records,
    )
    return merged, summary, per_detection
