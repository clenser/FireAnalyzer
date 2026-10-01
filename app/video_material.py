"""Video material analysis: deterministic identification + ONE Gemini assessment.

This module backs the two endpoints the video frontend uses.  It adds no new
classifier, no new dataset and no second copy of the material vocabulary: it
reuses the deterministic matcher, the canonical material database, the
extinguishing/fire-class mapping and the Gemini client boundary that ``/analyze``
already uses.

``POST /material-identification`` (deterministic)
    The client measures the video frames itself, aggregates them into one flame
    colour and sends ``{"rgb": [R, G, B], "lab": [L, a, b]}``.  That colour is
    fed straight into :func:`app.material_matching.match_material` - the exact
    matcher and the exact dataset ``/analyze`` uses - and the ranked result,
    the alternatives, the fire class and the extinguishing agents are returned in
    the same structure the image workflow returns.  **No YOLO detection and no
    segmentation run here**: the client already has the evidence, and the video
    fire class is derived from this deterministic match (never from Gemini).

``POST /video-material-analysis`` (one consolidated AI assessment)
    The client collected structured, per-frame evidence from every frame it
    analysed with ``/analyze`` and sends the whole collection here.  All of it
    goes into **exactly one** Gemini request and exactly **one** consolidated,
    video-level material assessment comes back, in the same canonical structure as
    the image AI analysis.

Invariants this module enforces
--------------------------------
* **One Gemini call per video.**  Never one call per frame, never an average of
  per-frame Gemini answers, never "the first frame" or "the highest-confidence
  frame" - see :func:`app.gemini_analysis.gemini_video_material_analysis`.
* **Numbers only.**  The request bodies carry measurements, never image files,
  base64 or masks; unknown request fields are rejected outright.  The Gemini
  prompt is built from the same numbers.
* **The deterministic answer is the source of truth.**  Gemini output is an
  independent opinion that can never overwrite or influence the fire class.
* **No secrets in responses.**  ``GEMINI_API_KEY`` is read from the backend
  settings only; it is never logged and never serialised.
* **Confidence is never inflated.**  Whatever Gemini reports is returned as-is,
  including values below the frontend's display threshold.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Sequence
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from .color_analysis import RepresentativeColor
from .config import Settings
from .fire_classes import classify
from .flame_evidence import perceived_brightness, rgb_to_hsv
from .gemini_analysis import gemini_video_material_analysis
from .material_matching import (
    MaterialDatabase,
    match_material,
)
from .schemas import (
    ExtinguishingAgent,
    FireClassResult,
    MaterialAnalysis,
    SuppressionInformation,
    jsonable,
)

__all__ = [
    "MAX_VIDEO_FRAMES",
    "VIDEO_DISPLAY_THRESHOLD_PERCENT",
    "AGGREGATED_COLOR_METHOD",
    "FrameEvidence",
    "MaterialIdentificationRequest",
    "VideoMaterialAnalysisRequest",
    "identify_material",
    "material_identification_payload",
    "frame_evidence",
    "build_video_evidence",
    "video_material_analysis",
]

logger = logging.getLogger(__name__)

#: Upper bound on the number of frames one video request may carry.  The frontend
#: analyses a handful of sampled frames; the cap only exists so one request can
#: never build an unbounded prompt.
MAX_VIDEO_FRAMES = 120

#: The frontend's own display rule for the video AI card.  It is reported with the
#: result so the threshold is defined once, on the backend.  A result below it is
#: still returned unchanged - this is never used to alter Gemini's confidence.
VIDEO_DISPLAY_THRESHOLD_PERCENT = 45.0

#: ``RepresentativeColor.method`` label for a colour the client measured and sent,
#: as opposed to one a clustering algorithm produced inside the pipeline.
AGGREGATED_COLOR_METHOD = "client_aggregated"


# ---------------------------------------------------------------------------
# Request models
#
# These are the transport contracts for the two video endpoints.  They validate
# and normalise *shape and range only* - the material decision is always made by
# the existing deterministic matcher or by Gemini, never here.
# ---------------------------------------------------------------------------
def _reject_boolean(value: Any) -> Any:
    """``true`` is not a colour channel and not a confidence value."""
    if isinstance(value, bool):
        raise ValueError("must be a number, not a boolean")
    return value


#: A plain number.  Booleans are rejected explicitly because JSON ``true`` would
#: otherwise coerce to ``1.0`` and silently look like a valid measurement.
_Number = Annotated[float, BeforeValidator(_reject_boolean)]

#: One RGB channel on a 0-255 scale.
_RgbChannel = Annotated[_Number, Field(ge=0.0, le=255.0)]
_RgbTriple = Annotated[list[_RgbChannel], Field(min_length=3, max_length=3)]

#: CIELAB components in the ``scikit-image`` convention the dataset uses:
#: L* in 0-100, a*/b* in the signed -128..127 range.
_Lightness = Annotated[_Number, Field(ge=0.0, le=100.0)]
_LabComponent = Annotated[_Number, Field(ge=-128.0, le=127.0)]
_LabTriple = Annotated[
    tuple[_Lightness, _LabComponent, _LabComponent],
    Field(min_length=3, max_length=3),
]

#: A model/detection confidence or an area ratio: a 0-1 fraction.
_Confidence = Annotated[_Number, Field(ge=0.0, le=1.0)]


class _StrictBody(BaseModel):
    """Base for the request bodies: unknown fields are a client error.

    ``extra="forbid"`` is what makes "no images, no base64" enforceable rather
    than aspirational: a client that posts ``image_base64`` gets a clean 422
    instead of a silently ignored field (or a huge payload).
    """

    model_config = ConfigDict(extra="forbid")


class MaterialIdentificationRequest(_StrictBody):
    """The aggregated flame colour measured by the client across video frames."""

    rgb: _RgbTriple = Field(
        description="Mean flame RGB of the aggregated video evidence, 0-255 per channel."
    )
    lab: _LabTriple = Field(
        description="Mean flame CIELAB [L, a, b] of the aggregated video evidence "
        "(scikit-image convention: L 0-100, a/b -128..127)."
    )


class FrameEvidence(_StrictBody):
    """Structured evidence measured for one analysed video frame."""

    frame_index: Annotated[int, Field(ge=0)] = Field(
        description="Index of the frame within the video (0-based or 1-based, as the client prefers)."
    )
    rgb: _RgbTriple = Field(description="Mean flame RGB for this frame, 0-255 per channel.")
    lab: _LabTriple = Field(description="Mean flame CIELAB [L, a, b] for this frame.")
    timestamp_seconds: Annotated[_Number, Field(ge=0.0)] | None = Field(
        default=None, description="Where the frame sits in the video, in seconds."
    )
    detection_confidence: _Confidence | None = Field(
        default=None, description="Fire-detection confidence for this frame (0-1)."
    )
    segmentation_confidence: _Confidence | None = Field(
        default=None, description="Flame-segmentation confidence for this frame (0-1)."
    )
    flame_area_ratio: _Confidence | None = Field(
        default=None, description="Segmented flame area as a fraction of the frame (0-1)."
    )


class VideoMaterialAnalysisRequest(_StrictBody):
    """The whole frame collection, to be assessed as one fire event."""

    frames: Annotated[
        list[FrameEvidence],
        Field(min_length=1, max_length=MAX_VIDEO_FRAMES),
    ] = Field(
        description="Frame-level evidence collected from every successfully analysed frame."
    )


# ---------------------------------------------------------------------------
# Deterministic identification (no YOLO, no segmentation, no LLM)
# ---------------------------------------------------------------------------
def _fire_class_payload(fire_class: FireClassResult) -> dict[str, Any]:
    """``FireClassResult`` as JSON, with ``class_`` renamed to ``class``.

    Same rename :meth:`app.schemas.AnalysisResult.to_dict` performs for
    ``/analyze``; it is repeated here because this endpoint has no full
    ``AnalysisResult`` to route through.
    """
    payload = jsonable(dataclasses.asdict(fire_class))
    payload["class"] = payload.pop("class_")
    return payload


def material_identification_payload(
    analysis: MaterialAnalysis,
    suppression: SuppressionInformation,
    fire_class: FireClassResult,
    agents: Sequence[ExtinguishingAgent],
) -> dict[str, Any]:
    """Build the deterministic response body.

    ``material_analysis`` is the identical structure ``/analyze`` returns, and
    ``fire_class`` / ``extinguishing_agents`` come from the same mapping in
    :mod:`app.fire_classes`.  The three top-level fields (``primary_material``,
    ``similarity``, ``alternatives``) are mirrors of the nested ones, following the
    project's existing alias convention (``bbox`` mirrors ``bounding_box``).
    """
    material = jsonable(dataclasses.asdict(analysis))
    return {
        "success": True,
        "material_analysis": material,
        "primary_material": material["primary_material"],
        "similarity": material["similarity"],
        "alternatives": material["alternatives"],
        "suppression_information": jsonable(dataclasses.asdict(suppression)),
        "fire_class": _fire_class_payload(fire_class),
        "extinguishing_agents": [jsonable(dataclasses.asdict(agent)) for agent in agents],
    }


def identify_material(
    rgb: Sequence[float],
    lab: Sequence[float],
    database: MaterialDatabase,
    settings: Settings,
) -> dict[str, Any]:
    """Identify the material for one aggregated flame colour.

    Runs the **existing** deterministic matcher - the same
    :func:`app.material_matching.match_material` used by ``/analyze`` - against the
    client-supplied colour, against the **same** canonical dataset.  No new
    classifier, no duplicated material list, no YOLO and no segmentation: the
    matcher only ever needed the flame LAB colour, and that is what arrives here.

    Parameters
    ----------
    rgb, lab:
        The aggregated flame colour, already validated by
        :class:`MaterialIdentificationRequest`.
    database:
        The loaded ``flame_dataset.json``.
    settings:
        Active configuration (distance scale, number of alternatives).
    """
    color = RepresentativeColor(
        method=AGGREGATED_COLOR_METHOD,
        rgb=np.asarray([int(round(channel)) for channel in rgb], dtype=np.uint8),
        lab=np.asarray([float(channel) for channel in lab], dtype=np.float64),
    )
    analysis, suppression = match_material([color], database, settings)
    fire_class, agents = classify(database, analysis.primary_material, analysis.similarity)
    return material_identification_payload(analysis, suppression, fire_class, agents)


# ---------------------------------------------------------------------------
# Frame evidence normalisation
# ---------------------------------------------------------------------------
def _mean(values: Sequence[Sequence[float]]) -> list[float]:
    """Column-wise mean of equally sized numeric rows."""
    width = len(values[0])
    return [round(sum(row[channel] for row in values) / len(values), 2) for channel in range(width)]


def _std_dev(values: Sequence[Sequence[float]]) -> list[float]:
    """Column-wise population standard deviation: a consistency measure.

    Reported so Gemini can judge how stable the flame was across the video, not
    as a substitute for a per-frame classification.
    """
    width = len(values[0])
    count = len(values)
    means = _mean(values)
    spread: list[float] = []
    for channel in range(width):
        variance = sum((row[channel] - means[channel]) ** 2 for row in values) / count
        spread.append(round(math.sqrt(variance), 2))
    return spread


def _optional_mean(values: Sequence[float]) -> float | None:
    """Mean of the values that were actually supplied, or ``None`` if there were none."""
    return round(sum(values) / len(values), 4) if values else None


def frame_evidence(frame: FrameEvidence) -> dict[str, Any]:
    """Normalise one validated frame into the image path's evidence vocabulary.

    The keys mirror what :func:`app.flame_evidence.extract_flame_evidence` produces
    for a single image (``mean_color``, ``hsv``, ``brightness_0_255``,
    ``saturation_0_100``, ``detection_confidence``, ``segmentation_confidence``,
    ``flame_region.mask_area_ratio``) so one Gemini prompt can reason about
    frames and images with the same concepts.  HSV and brightness are derived by
    the existing helpers; nothing is invented and absent measurements are simply
    omitted.
    """
    rgb = [int(round(channel)) for channel in frame.rgb]
    lab = [round(float(channel), 2) for channel in frame.lab]

    evidence: dict[str, Any] = {
        "frame_index": int(frame.frame_index),
        "mean_color": {"rgb": rgb, "lab": lab},
    }
    if frame.timestamp_seconds is not None:
        evidence["timestamp_seconds"] = round(float(frame.timestamp_seconds), 3)

    hsv = rgb_to_hsv(rgb)
    if hsv is not None:
        evidence["hsv"] = hsv
        evidence["saturation_0_100"] = hsv["s"]
    brightness = perceived_brightness(rgb)
    if brightness is not None:
        evidence["brightness_0_255"] = brightness

    if frame.detection_confidence is not None:
        evidence["detection_confidence"] = round(float(frame.detection_confidence), 4)
    if frame.segmentation_confidence is not None:
        evidence["segmentation_confidence"] = round(float(frame.segmentation_confidence), 4)
    if frame.flame_area_ratio is not None:
        evidence["flame_region"] = {"mask_area_ratio": round(float(frame.flame_area_ratio), 6)}

    return evidence


def _collection_summary(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Whole-collection statistics, i.e. what a video adds over a single frame.

    Spread, frame count and time span are the measurable form of "consistency and
    amount of evidence".  They are measurements of the collection, never a
    per-frame classification and never an average of model opinions.
    """
    rgbs = [frame["mean_color"]["rgb"] for frame in frames]
    labs = [frame["mean_color"]["lab"] for frame in frames]
    timestamps = [frame["timestamp_seconds"] for frame in frames if "timestamp_seconds" in frame]

    summary: dict[str, Any] = {
        "frame_count": len(frames),
        "mean_rgb": [float(channel) for channel in _mean(rgbs)],
        "mean_lab": _mean(labs),
        "lab_std_dev": _std_dev(labs),
    }
    if len(timestamps) == len(frames) and len(timestamps) > 1:
        summary["time_span_seconds"] = round(max(timestamps) - min(timestamps), 3)

    for key in (
        "detection_confidence",
        "segmentation_confidence",
        "saturation_0_100",
        "brightness_0_255",
    ):
        values = [frame[key] for frame in frames if key in frame]
        average = _optional_mean(values)
        if average is not None:
            summary[f"mean_{key}"] = average

    areas = [
        frame["flame_region"]["mask_area_ratio"]
        for frame in frames
        if "flame_region" in frame
    ]
    mean_area = _optional_mean(areas)
    if mean_area is not None:
        summary["mean_flame_area_ratio"] = mean_area

    return summary


def build_video_evidence(frames: Sequence[FrameEvidence]) -> dict[str, Any]:
    """Aggregate every supplied frame into a single evidence payload.

    Returns ``{}`` when no frame carries usable colour evidence, which the caller
    reports as an unavailable AI analysis rather than calling Gemini with nothing.
    """
    per_frame = [frame_evidence(frame) for frame in frames]
    per_frame = [frame for frame in per_frame if frame.get("mean_color")]
    if not per_frame:
        return {}
    return {
        "frames": per_frame,
        "collection": _collection_summary(per_frame),
        "frame_count": len(per_frame),
    }


# ---------------------------------------------------------------------------
# Video-level orchestration
# ---------------------------------------------------------------------------
def video_material_analysis(
    frames: Sequence[FrameEvidence],
    settings: Settings,
    database: MaterialDatabase | None = None,
) -> dict[str, Any]:
    """Produce ONE consolidated material assessment for a whole video.

    All frame evidence is forwarded to Gemini in a single request and exactly one
    consolidated result is returned, in the same canonical structure as the image
    AI analysis: ``available``, ``primary_material``, ``matches``,
    ``overall_confidence_level``, ``uncertain``, ``evidence_quality`` and
    ``reasoning_summary``.

    Any Gemini failure (disabled, missing key, quota, timeout, malformed payload)
    degrades to ``{"available": false, "error": ...}`` - this function never
    raises, exactly like :func:`app.gemini_analysis.gemini_material_analysis`.
    The frame counts and the frontend's display threshold are reported alongside
    the result; Gemini's confidence is passed through unchanged.
    """
    evidence = build_video_evidence(frames)
    analysed = int(evidence.get("frame_count", 0))
    if analysed == 0:
        payload: dict[str, Any] = {
            "available": False,
            "error": "No usable flame colour evidence was supplied for the video analysis.",
        }
    else:
        try:
            payload = gemini_video_material_analysis(evidence, settings, database)
        except Exception:  # noqa: BLE001 - the AI analysis must never fail the request
            logger.exception("Video material analysis failed unexpectedly")
            payload = {
                "available": False,
                "error": "The video AI material analysis could not be completed.",
            }

    payload["frames_supplied"] = len(frames)
    payload["frames_analyzed"] = analysed
    payload["display_threshold_percent"] = VIDEO_DISPLAY_THRESHOLD_PERCENT
    return payload