"""Binary flame-mask encoding for the API response.

The segmentation stage produces a ``uint8`` mask (0 or 255) at the analysed
image's resolution.  The API has to hand that mask to the frontend so it can
draw the *actual* segmented flame region - the detection bounding box is a
different thing and is reported separately.

Why PNG + base64
----------------
* A binary mask compresses to a tiny payload: a 4K mask with 45 000 flame
  pixels is a few kilobytes, whereas a raw pixel array would be hundreds of
  kilobytes of JSON numbers.
* PNG is lossless, so the client recovers the *exact* mask - and therefore the
  exact ``flame_pixel_count`` - by decoding the base64 blob.
* It keeps the response a single ``application/json`` document.  No
  ``image/png`` side channel and no base64 *upload* payload is involved: base64
  appears only in this generated response field.

Base64 is a transport detail of the mask, not of the analysis: the decoded
pixels are identical to the array the pipeline clustered on.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

__all__ = [
    "MASK_ENCODING",
    "encode_mask_base64",
    "decode_mask_base64",
    "mask_from_base64",
]

logger = logging.getLogger(__name__)

#: Value published as ``segmentation.mask_encoding``.
MASK_ENCODING = "png_base64"

#: zlib levels 0-9 are valid; anything else is clamped rather than rejected so a
#: bad environment variable cannot fail a request.
_MIN_COMPRESSION = 0
_MAX_COMPRESSION = 9


def _clamp_compression(level: Any) -> int:
    try:
        value = int(level)
    except (TypeError, ValueError):
        return 6
    return max(_MIN_COMPRESSION, min(_MAX_COMPRESSION, value))


def encode_mask_base64(mask: np.ndarray | None, compression: int = 6) -> str | None:
    """Encode a binary mask as a base64 PNG.

    Parameters
    ----------
    mask:
        ``uint8`` (or boolean) mask of shape ``(height, width)``.  Any non-zero
        value counts as flame.
    compression:
        PNG zlib level (0-9).  Binary masks compress extremely well, so this
        barely affects the encoded size.

    Returns
    -------
    str | None
        The base64 PNG, or ``None`` when the mask is absent/empty.  The decoded
        image has exactly the same width and height as ``mask``.
    """
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        logger.warning("Cannot encode a mask with shape %s", array.shape)
        return None

    binary = (array > 0).astype(np.uint8) * 255
    ok, buffer = cv2.imencode(
        ".png", binary, [int(cv2.IMWRITE_PNG_COMPRESSION), _clamp_compression(compression)]
    )
    if not ok:
        logger.error("PNG encoding of the flame mask failed")
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_mask_base64(payload: str | bytes | None) -> np.ndarray | None:
    """Decode a base64 PNG mask back into a ``uint8`` array of 0/255.

    Returns ``None`` for a missing or undecodable payload, so a client can never
    be handed a half-decoded mask.  This is the exact inverse of
    :func:`encode_mask_base64` and is used by the tests to prove the API
    contract holds.
    """
    if not payload:
        return None
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        logger.warning("Flame mask payload is not valid base64: %s", exc)
        return None

    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        logger.warning("Flame mask payload is not a decodable PNG")
        return None
    return (decoded > 0).astype(np.uint8) * 255


def mask_from_base64(payload: str | bytes | None) -> np.ndarray | None:
    """Decode a mask and return it as a boolean array (``True`` = flame).

    Convenience wrapper for consumers - the frontend overlay logic and the
    test-suite assertions both work in boolean space.
    """
    decoded = decode_mask_base64(payload)
    return None if decoded is None else decoded > 0
