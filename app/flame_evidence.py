"""Read-only extraction of flame evidence for the secondary Gemini analysis.

This module is strictly additive.  It never touches the deterministic
:class:`app.analyzer.FlameAnalyzer` pipeline and never recomputes flame colours:
it only reads the JSON-serialisable analysis result the analyzer already
produced and repackages the available measurements into a single evidence dict
for :func:`app.gemini_analysis.gemini_material_analysis`.

What it does:

* forwards the analyzer's own ``mean_color`` (final RGB/LAB) verbatim;
* derives HSV, perceived brightness and saturation from that mean RGB with a
  small isolated helper (the only values computed here - everything else is
  read straight from the result);
* forwards flame-region statistics (pixel counts, mask area, mask dimensions),
  detection/segmentation confidences, the detection bounding box and its
  aspect ratio, and the already-calculated colour clusters / dominant colours
  of every clustering algorithm that ran.

What it never does:

* invents values - any field the analyzer did not produce is omitted;
* sends images, masks-as-images or base64 payloads to Gemini;
* modifies the deterministic result in any way.
"""

from __future__ import annotations

import colorsys
import logging
import math
from collections.abc import Sequence
from typing import Any

__all__ = ["extract_flame_evidence", "rgb_to_hsv"]

logger = logging.getLogger(__name__)

#: Perceived-luminance weights (ITU-R BT.601) for the 0-255 RGB brightness.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def rgb_to_hsv(rgb: Sequence[float] | None) -> dict[str, float] | None:
    """Convert an RGB triple (0-255 per channel) to HSV.

    Returns ``{"h": 0-360, "s": 0-100, "v": 0-100}`` or ``None`` when the input
    is missing or not a 3-channel numeric triple.  Uses the stdlib
    :mod:`colorsys` - no new dependency, no image involved.
    """
    if rgb is None:
        return None
    try:
        r, g, b = (float(channel) for channel in rgb)  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= channel <= 255.0 for channel in (r, g, b)):
        return None
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return {
        "h": round(h * 360.0, 1),
        "s": round(s * 100.0, 1),
        "v": round(v * 100.0, 1),
    }


def _brightness(rgb: Sequence[float] | None) -> float | None:
    """Perceived luminance of an RGB triple on a 0-255 scale."""
    if rgb is None:
        return None
    try:
        r, g, b = (float(channel) for channel in rgb)  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    return round(_LUMA_WEIGHTS[0] * r + _LUMA_WEIGHTS[1] * g + _LUMA_WEIGHTS[2] * b, 1)


def _number(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` (never invents a value)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _mean_color(flame_analysis: dict[str, Any]) -> dict[str, Any] | None:
    mean = flame_analysis.get("mean_color")
    if not isinstance(mean, dict):
        return None
    rgb = mean.get("rgb")
    lab = mean.get("lab")
    if not isinstance(rgb, list) or not isinstance(lab, list):
        return None
    return {"rgb": rgb, "lab": lab}


def _bounding_box_evidence(fire_detection: dict[str, Any]) -> dict[str, Any] | None:
    box = fire_detection.get("bounding_box") or fire_detection.get("bbox")
    if not isinstance(box, dict):
        return None
    x1, y1 = _number(box.get("x1")), _number(box.get("y1"))
    x2, y2 = _number(box.get("x2")), _number(box.get("y2"))
    if None in (x1, y1, x2, y2):
        return None
    width, height = abs(x2 - x1), abs(y2 - y1)
    aspect = round(width / height, 3) if height else None
    return {
        "width_px": int(width),
        "height_px": int(height),
        "aspect_ratio": aspect,
    }


def _clustering_evidence(flame_analysis: dict[str, Any]) -> dict[str, Any]:
    """Forward the already-calculated clusters and dominant colours, per algorithm."""
    per_algorithm: dict[str, Any] = {}
    for method, entry in flame_analysis.items():
        if not isinstance(entry, dict) or "method" not in entry:
            continue
        record: dict[str, Any] = {
            "representative_rgb": entry.get("rgb"),
            "representative_lab": entry.get("lab"),
            "cluster_count": _int(entry.get("cluster_count")),
        }
        dominant = entry.get("dominant_color")
        if isinstance(dominant, dict):
            record["dominant_color"] = {"rgb": dominant.get("rgb"), "lab": dominant.get("lab")}
        centroids = entry.get("centroids")
        if isinstance(centroids, list):
            record["clusters"] = [
                {
                    "rgb": centroid.get("rgb"),
                    "lab": centroid.get("lab"),
                    "pixel_share": _number(centroid.get("weight")),
                }
                for centroid in centroids
                if isinstance(centroid, dict)
            ]
        per_algorithm[method] = record

    return {
        "algorithms": flame_analysis.get("algorithms"),
        "n_clusters": _int(flame_analysis.get("n_clusters")),
        "skipped_reason": flame_analysis.get("skipped_reason"),
        "per_algorithm": per_algorithm,
    }


def extract_flame_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Build the evidence dict for the secondary Gemini analysis.

    Parameters
    ----------
    result:
        A successful ``FlameAnalyzer.analyze_image`` result dict.  Every value is
        read verbatim from it; the only derived quantities are HSV, brightness
        and saturation, computed from the mean flame RGB.

    Returns
    -------
    dict
        A JSON-safe evidence dict.  Fields the analyzer did not produce are
        omitted - nothing is invented.  Always contains ``mean_color`` when the
        analyzer reported one; the API layer treats its absence as
        "no evidence" and skips the Gemini call.
    """
    if not isinstance(result, dict):
        return {}

    flame_analysis = result.get("flame_analysis")
    flame_analysis = flame_analysis if isinstance(flame_analysis, dict) else {}
    fire_detection = result.get("fire_detection")
    fire_detection = fire_detection if isinstance(fire_detection, dict) else {}
    segmentation = result.get("segmentation")
    segmentation = segmentation if isinstance(segmentation, dict) else {}

    evidence: dict[str, Any] = {}

    mean_color = _mean_color(flame_analysis)
    if mean_color is not None:
        evidence["mean_color"] = mean_color
        hsv = rgb_to_hsv(mean_color.get("rgb"))
        if hsv is not None:
            evidence["hsv"] = hsv
        brightness = _brightness(mean_color.get("rgb"))
        if brightness is not None:
            evidence["brightness_0_255"] = brightness
        if hsv is not None:
            evidence["saturation_0_100"] = hsv["s"]

    flame_region = {
        "flame_pixel_count": _int(flame_analysis.get("flame_pixel_count")),
        "samples_used": _int(flame_analysis.get("samples_used")),
        "pixels_sampled": flame_analysis.get("pixels_sampled"),
        "mask_area_ratio": _number(segmentation.get("mask_area_ratio")),
        "mask_width_px": _int(segmentation.get("mask_width")),
        "mask_height_px": _int(segmentation.get("mask_height")),
    }
    flame_region = {key: value for key, value in flame_region.items() if value is not None}
    if flame_region:
        evidence["flame_region"] = flame_region

    detection_confidence = _number(fire_detection.get("confidence"))
    if detection_confidence is not None:
        evidence["detection_confidence"] = detection_confidence
    segmentation_confidence = _number(segmentation.get("confidence"))
    if segmentation_confidence is not None:
        evidence["segmentation_confidence"] = segmentation_confidence

    bounding_box = _bounding_box_evidence(fire_detection)
    if bounding_box is not None:
        evidence["detection_bounding_box"] = bounding_box

    evidence["color_distribution"] = _clustering_evidence(flame_analysis)

    return evidence
