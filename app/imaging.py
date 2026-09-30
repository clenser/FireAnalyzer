"""Image decoding and validation helpers for the API layer.

The ML pipeline itself never touches the filesystem: it consumes an OpenCV BGR
array.  Uploaded bytes are turned into such an array here, in memory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .errors import InvalidImageError

__all__ = ["validate_image", "decode_image_bytes", "imread"]

logger = logging.getLogger(__name__)

MAX_DIMENSION = 20000  # sanity guard against decompression bombs


def validate_image(image: np.ndarray | None) -> np.ndarray:
    """Validate and normalise an image array.

    Accepts grayscale, BGRA and BGR arrays and always returns a 3-channel
    ``uint8`` BGR array.

    Raises
    ------
    InvalidImageError
        If the input is ``None``, empty, not an array, or too large.
    """
    if image is None:
        raise InvalidImageError("No image was supplied.")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Image data must be a numpy array.")
    if image.size == 0:
        raise InvalidImageError("The supplied image is empty.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"Unsupported image shape: {image.shape}.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if max(image.shape[:2]) > MAX_DIMENSION:
        raise InvalidImageError(
            f"Image is too large: {image.shape[1]}x{image.shape[0]} pixels."
        )
    return np.ascontiguousarray(image)


def decode_image_bytes(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode uploaded image bytes into a validated BGR array.

    Raises
    ------
    InvalidImageError
        If the payload is empty or is not a decodable image.
    """
    if not data:
        raise InvalidImageError("No image bytes were supplied.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("The uploaded data is not a decodable image.")
    return validate_image(image)


def imread(path: str | Path) -> np.ndarray:
    """Read an image from disk and validate it (used by the CLI only)."""
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"Image file not found: {path}")
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except OSError as exc:  # ultralytics patches cv2.imread and may raise
        raise InvalidImageError(f"Could not read image file: {path}") from exc
    if image is None:
        raise InvalidImageError(f"Could not decode image file: {path}")
    return validate_image(image)
