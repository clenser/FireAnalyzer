"""Optional debug visualisations.

These helpers are **never** called by the inference pipeline or by the future
API.  They exist for local debugging and for generating documentation images::

    from app.visualization import create_detection_visualization
    cv2.imwrite("debug.jpg", create_detection_visualization(image_bgr, detection))

They always return a new array and never mutate their input.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schemas import FireDetection

__all__ = ["create_detection_visualization", "create_mask_visualization"]


def create_detection_visualization(
    image_bgr: np.ndarray,
    detection: FireDetection,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the detection bounding box and label on a copy of the image."""
    canvas = image_bgr.copy()
    box = detection.bounding_box
    if box is not None:
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
        label_y = box.y1 - 10 if box.y1 > 20 else box.y1 + 20
        cv2.putText(
            canvas,
            f"Fire: {detection.confidence:.3f}",
            (box.x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )
    return canvas


def create_mask_visualization(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.5,
    mask_color: tuple[int, int, int] = (0, 140, 255),
) -> np.ndarray:
    """Blend the flame mask over a copy of the image (debugging only)."""
    if mask is None or mask.size == 0:
        return image_bgr.copy()
    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))
    overlay = image_bgr.copy()
    selection = (mask > 0)[..., None]
    overlay[selection] = mask_color
    return cv2.addWeighted(overlay, float(alpha), image_bgr, 1.0 - float(alpha), 0.0)
