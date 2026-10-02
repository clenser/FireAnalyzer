"""JSON-only material endpoints: deterministic identification and video voting.

``POST /material-identification``
    The client sends one aggregated flame colour ``{"rgb", "lab"}``.  It is
    ranked by the same LAB matcher ``/analyze`` uses, normalised into
    deterministic evidence and decided by the same Python fusion engine
    (:mod:`app.material_fusion`).  No vision evidence is available for a bare
    colour, so the decision rests on colour alone and is reported as uncertain
    whenever colour cannot separate the leading materials.

``POST /video-material-analysis``
    The client sends per-frame measurements it collected.  Every frame is
    decided independently by the Python fusion engine and the video result is
    the Python majority/consistency vote over those frame decisions
    (:func:`app.material_fusion.aggregate_video_frames`).  **No LLM produces a
    video conclusion** - the former single-Gemini-call consolidation was
    removed.  For full server-side video analysis (detection, masks and
    vision evidence per frame) use ``POST /analyze-video``.

Request bodies carry numbers only; unknown fields (images, base64) are rejected.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Annotated, Any, Sequence

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from .color_analysis import RepresentativeColor
from .config import Settings
from .fire_classes import classify_decision
from .material_fusion import (
    aggregate_video_frames,
    deterministic_evidence,
    fuse_material,
    video_fire_class,
)
from .material_matching import MaterialDatabase, rank_materials
from .schemas import jsonable
from .vision_providers import unavailable_vision

__all__ = [
    "MAX_VIDEO_FRAMES",
    "VIDEO_DISPLAY_THRESHOLD_PERCENT",
    "AGGREGATED_COLOR_METHOD",
    "FrameEvidence",
    "MaterialIdentificationRequest",
    "VideoMaterialAnalysisRequest",
    "identify_material",
    "video_material_analysis",
]

logger = logging.getLogger(__name__)

#: Upper bound on the number of frames one video request may carry.
MAX_VIDEO_FRAMES = 120

#: Kept in the response for frontend compatibility (AI-card display rule).
VIDEO_DISPLAY_THRESHOLD_PERCENT = 45.0

#: ``RepresentativeColor.method`` label for a colour the client measured.
AGGREGATED_COLOR_METHOD = "client_aggregated"

_NO_VISION = "no image is available on this endpoint; colour evidence only"


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
# Deterministic decisions
# ---------------------------------------------------------------------------
def _decide_color(
    rgb: Sequence[float],
    lab: Sequence[float],
    database: MaterialDatabase,
    settings: Settings,
    *,
    quality: str,
    detection_confidence: float = 0.0,
    mask_area_ratio: float = 0.0,
) -> dict[str, Any]:
    """Rank one measured colour and fuse it (colour evidence only)."""
    color = RepresentativeColor(
        method=AGGREGATED_COLOR_METHOD,
        rgb=np.asarray([int(round(channel)) for channel in rgb], dtype=np.uint8),
        lab=np.asarray([float(channel) for channel in lab], dtype=np.float64),
    )
    ranking = rank_materials([color], database, settings)
    evidence = deterministic_evidence(
        ranking,
        settings,
        flame_pixel_count=0,
        mask_area_ratio=mask_area_ratio,
        real_mask=False,
        detection_confidence=detection_confidence,
        detection_count=0,
        mask_count=0,
        mean_rgb=[int(round(channel)) for channel in rgb],
        mean_lab=[round(float(channel), 2) for channel in lab],
        quality=quality,
    )
    vision = unavailable_vision(_NO_VISION)
    decision = fuse_material(evidence, vision, database)
    fire_class, agents = classify_decision(
        database, decision["final_material"], decision["leading_candidates"], decision["confidence"]
    )
    fire_payload = jsonable(dataclasses.asdict(fire_class))
    fire_payload["class"] = fire_payload.pop("class_")
    return {
        "deterministic_evidence": evidence,
        "vision_evidence": vision,
        "vision_provider": "none",
        **decision,
        "fire_class": fire_payload,
        "extinguishing_agents": [jsonable(dataclasses.asdict(agent)) for agent in agents],
    }


def identify_material(
    rgb: Sequence[float],
    lab: Sequence[float],
    database: MaterialDatabase,
    settings: Settings,
) -> dict[str, Any]:
    """Deterministic material decision for one aggregated flame colour."""
    body = _decide_color(rgb, lab, database, settings, quality="moderate")
    final = body["final_material"]
    similarity = {row["material"]: row["similarity"] for row in body["deterministic_evidence"]["ranking"]}
    entry = database.get(final) if final else None
    alternatives = [
        {"material": row["material"], "similarity": similarity.get(row["material"], 0.0)}
        for row in body["candidate_materials"]
        if row["material"] != final
    ][: max(0, settings.max_alternatives)]
    material_analysis = {
        "primary_material": final,
        "similarity": similarity.get(final, 0.0) if final else 0.0,
        "alternatives": alternatives,
        "database_notes": entry.notes if entry else None,
        "score_basis": "Python fusion of CIELAB colour matching (no vision evidence on this endpoint)",
        "uncertain": body["uncertain"],
    }
    return {
        "success": True,
        "analysis_type": "material_identification",
        "cached": False,
        "detection_count": 0,
        "mask_count": 0,
        **body,
        "material_analysis": material_analysis,
        "primary_material": final,
        "similarity": material_analysis["similarity"],
        "alternatives": alternatives,
        "suppression_information": {
            "source": database.source,
            "material": final,
            "methods": [agent["name"] for agent in body["extinguishing_agents"]],
            "database_notes": entry.notes if entry else None,
        },
        "error": None,
    }


def video_material_analysis(
    frames: Sequence["FrameEvidence"],
    settings: Settings,
    database: MaterialDatabase,
) -> dict[str, Any]:
    """Decide every supplied frame in Python, then majority-vote the video."""
    frame_results: list[dict[str, Any]] = []
    for frame in frames:
        detection = float(frame.detection_confidence) if frame.detection_confidence is not None else 0.0
        quality = "moderate" if frame.detection_confidence is None or detection >= 0.45 else "limited"
        result = _decide_color(
            frame.rgb,
            frame.lab,
            database,
            settings,
            quality=quality,
            detection_confidence=detection,
            mask_area_ratio=float(frame.flame_area_ratio or 0.0),
        )
        frame_results.append(
            {
                "success": True,
                "frame_index": int(frame.frame_index),
                "timestamp_seconds": frame.timestamp_seconds,
                **result,
            }
        )

    aggregate = aggregate_video_frames(frame_results, len(frames), database)
    fire_class, agents = video_fire_class(database, aggregate, frame_results)
    analysed = aggregate["frames_analyzed"]
    matches = [
        {
            "rank": rank,
            "material": vote["material"],
            "confidence_percent": round(100.0 * vote["frames"] / float(analysed), 1) if analysed else 0.0,
            "reason": f"chosen by {vote['frames']} of {analysed} frames (Python per-frame fusion)",
        }
        for rank, vote in enumerate(aggregate["votes"], start=1)
    ]
    if aggregate["final_material"]:
        summary = (
            f"{aggregate['final_material']} won the per-frame majority vote "
            f"({aggregate['votes'][0]['frames']} of {analysed} frames)."
        )
    else:
        summary = "No material won the per-frame vote clearly: " + "; ".join(
            aggregate["uncertainty_reasons"] or ["insufficient evidence"]
        )
    return {
        "success": True,
        "analysis_type": "video_material_analysis",
        "cached": False,
        # Legacy fields kept for existing clients.
        "available": analysed > 0,
        "primary_material": aggregate["final_material"],
        "matches": matches,
        "overall_confidence_level": aggregate["confidence_level"],
        "evidence_quality": "moderate" if analysed else "insufficient",
        "reasoning_summary": summary,
        "frames_supplied": len(frames),
        "frames_analyzed": analysed,
        "display_threshold_percent": VIDEO_DISPLAY_THRESHOLD_PERCENT,
        # Common response fields.
        "detection_count": 0,
        "mask_count": 0,
        "vision_provider": "none",
        "final_material": aggregate["final_material"],
        "confidence": aggregate["confidence"],
        "confidence_percent": aggregate["confidence_percent"],
        "confidence_level": aggregate["confidence_level"],
        "uncertain": aggregate["uncertain"],
        "uncertainty_reasons": aggregate["uncertainty_reasons"],
        "leading_candidates": aggregate["leading_candidates"],
        "candidate_materials": aggregate["votes"],
        "consolidated": aggregate,
        "fire_class": fire_class,
        "extinguishing_agents": agents,
        "frames": frame_results,
        "error": None,
    }
