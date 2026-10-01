"""Secondary, Gemini-powered material identification.

This module is an *independent* second opinion on the burning material.  It is
strictly additive:

* The deterministic :mod:`app.material_matching` result is never touched.
* No image and no base64 data is ever sent to Gemini - only the flame evidence
  extracted from the analyzer's own result by :mod:`app.flame_evidence`
  (final RGB/LAB, HSV, brightness, saturation, flame-region statistics,
  detection/segmentation confidences, bounding-box geometry and the
  already-calculated colour clusters), plus the canonical material vocabulary
  loaded from the same ``flame_dataset.json``.
* Gemini may only choose from the canonical dataset categories.  Every response
  is validated against the vocabulary, the rank sequence, the confidence range
  and the uncertainty consistency rules; anything malformed becomes a clean
  ``{"available": false, ...}`` state instead of leaking malformed output.
* A failure here (disabled, missing key, quota, timeout, bad payload) never
  fails the ``/analyze`` request - the deterministic analysis is returned
  untouched.

Uncertainty model
-----------------
Gemini is *not* asked to name the fuel from flame colour.  It is asked to
evaluate whether the supplied evidence distinguishes among the candidate
materials, and it is explicitly allowed - and expected - to answer
``primary_material: null`` with ``uncertain: true`` when it does not.  Flame
colour is treated as weak, indirect evidence: different fuels can produce
overlapping flame colours, and camera exposure, white balance and background
all shift the measured values.

``confidence_percent`` values are the model's *relative heuristic* confidence
allocation.  They are not calibrated probabilities, are never derived from
token probabilities, and must reflect evidence quality rather than mere
colour closeness.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .config import Settings
from .errors import MissingDatabaseError
from .material_matching import MaterialDatabase, load_material_database

__all__ = [
    "SYSTEM_INSTRUCTION",
    "VIDEO_SYSTEM_INSTRUCTION",
    "VIDEO_CONSOLIDATION_INSTRUCTION",
    "REQUIRED_CANDIDATES",
    "EVIDENCE_QUALITY_LEVELS",
    "CONFIDENCE_LEVELS",
    "GeminiMatch",
    "GeminiTimeoutError",
    "gemini_material_analysis",
    "gemini_video_material_analysis",
]

logger = logging.getLogger(__name__)

#: Number of ranked candidates requested from (and validated from) Gemini.  The
#: dataset holds 18 materials, so five are always available; if a custom
#: dataset ever held fewer, that many are required instead.
REQUIRED_CANDIDATES = 5

#: Allowed ``overall_confidence_level`` values.
CONFIDENCE_LEVELS = ("high", "medium", "low")

#: Allowed ``evidence_quality`` values, from least to most distinguishing.
EVIDENCE_QUALITY_LEVELS = ("insufficient", "limited", "moderate", "strong")

#: Strict system instruction.  It frames the task as an evidence-sufficiency
#: evaluation (not colour matching), fixes the vocabulary rule, authorises the
#: uncertain/null outcome and the "heuristic, not probabilities" meaning of the
#: percentage values.
#:
#: :data:`_EVIDENCE_INSTRUCTION` holds the rules that apply to every request - a
#: single image and a whole video alike - and each caller appends what is specific
#: to it.  The single-image assembly below is byte-for-byte the instruction the
#: image workflow has always used.
_EVIDENCE_INSTRUCTION = """You are performing secondary material analysis for a flame-analysis system.

The supplied measurements describe the observed flame, not necessarily the fuel itself.

Evaluate whether the available visual flame evidence provides enough information to distinguish among the candidate materials.

Do not infer material identity from flame color alone. Treat RGB/LAB/HSV and flame morphology as indirect evidence.

Consider multiple possible materials when evidence overlaps.

Only use materials from the supplied canonical material vocabulary.

If the evidence is insufficient to distinguish materials, explicitly return an uncertain result rather than guessing: set primary_material to null, uncertain to true, overall_confidence_level to "low", and evidence_quality to "insufficient" or "limited".

Do not fabricate evidence that is not supplied.

Do not assume that the visually closest material is the actual burning material.

A flame's color is weak evidence for material identity: different fuels can produce overlapping flame colors, and camera exposure, white balance and the background environment affect the measured values."""

SYSTEM_INSTRUCTION = f"""{_EVIDENCE_INSTRUCTION}

Return structured JSON only."""

#: The one rule that makes a video assessment different from an image assessment:
#: the frames are samples of a single event, not independent classifications to be
#: scored and averaged.
VIDEO_CONSOLIDATION_INSTRUCTION = """The supplied observations are multiple samples of the same fire event. Evaluate the complete collection jointly and return ONE consolidated material assessment for the video. Do not independently classify each frame and average confidence percentages. Confidence must reflect the consistency, quality, and amount of evidence across the complete set of observations."""

VIDEO_SYSTEM_INSTRUCTION = f"""{_EVIDENCE_INSTRUCTION}

{VIDEO_CONSOLIDATION_INSTRUCTION}

Return exactly one consolidated assessment for the video. Never return one assessment per observation, and never derive it by averaging per-observation confidences.

Return structured JSON only."""

#: Structured-output contract enforced by the SDK (``response_mime_type``+
#: ``response_schema``) and re-validated locally after the call.
#:
#: Schema note: the Gemini API accepts only the restricted JSON-Schema subset
#: documented at https://ai.google.dev/gemini-api/docs/structured-output -
#: ``type`` must be a single type name, so a nullable field is expressed with
#: ``"nullable": true`` rather than a union such as ``["string", "null"]``
#: (the latter fails SDK validation with a pydantic ValidationError before
#: any request is sent).
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_material": {"type": "string", "nullable": True},
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "material": {"type": "string"},
                    "confidence_percent": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["rank", "material", "confidence_percent", "reason"],
            },
        },
        "overall_confidence_level": {"type": "string", "enum": ["high", "medium", "low"]},
        "uncertain": {"type": "boolean"},
        "evidence_quality": {
            "type": "string",
            "enum": ["insufficient", "limited", "moderate", "strong"],
        },
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "primary_material",
        "matches",
        "overall_confidence_level",
        "uncertain",
        "evidence_quality",
        "reasoning_summary",
    ],
}


class GeminiTimeoutError(TimeoutError):
    """The Gemini request exceeded the configured timeout."""


@dataclass(frozen=True)
class GeminiMatch:
    """One validated Gemini candidate."""

    rank: int
    material: str
    confidence_percent: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "material": self.material,
            "confidence_percent": self.confidence_percent,
            "reason": self.reason,
        }


def _unavailable(error: str) -> dict[str, Any]:
    """The single failure shape: a clean, JSON-safe unavailable state."""
    return {"available": False, "error": error}


# ---------------------------------------------------------------------------
# SDK boundary (lazy imports: this module must import without google-genai)
# ---------------------------------------------------------------------------
def _create_client(settings: Settings):
    """Build the ``google-genai`` client.  Lazy import keeps the dependency optional."""
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def _generate(
    client: Any,
    settings: Settings,
    contents: str,
    system_instruction: str = SYSTEM_INSTRUCTION,
) -> Any:
    """One structured-output content generation call.

    ``system_instruction`` defaults to the single-image instruction, so the image
    workflow's Gemini configuration is unchanged; the video workflow passes
    :data:`VIDEO_SYSTEM_INSTRUCTION`.
    """
    from google.genai import types

    return client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            # HttpOptions.timeout is in milliseconds.
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_s * 1000)),
        ),
    )


def _run_with_timeout(func: Callable[[], Any], timeout_s: float) -> Any:
    """Run ``func`` on a worker thread, giving up after ``timeout_s`` seconds.

    The worker is a daemon thread, so an abandoned (slow) request can never
    block the backend or the test process from finishing.
    """
    outcome: list[Any] = []
    failure: list[BaseException] = []

    def target() -> None:
        try:
            outcome.append(func())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            failure.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise GeminiTimeoutError(f"the Gemini request exceeded {timeout_s:g}s")
    if failure:
        raise failure[0]
    return outcome[0]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _format_color_list(values: Any) -> str:
    """Render a colour value for the prompt, or ``(not supplied)``."""
    if isinstance(values, list) and values:
        return str([round(v, 2) if isinstance(v, float) else v for v in values])
    return "(not supplied)"


def _format_clusters(clusters: Any) -> list[str]:
    """One line per cluster centroid: its RGB and its share of the samples."""
    if not isinstance(clusters, list):
        return []
    lines: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        share = cluster.get("pixel_share")
        share_text = f"{100.0 * float(share):.1f}% of samples" if isinstance(share, (int, float)) else "share unknown"
        lines.append(f"RGB {_format_color_list(cluster.get('rgb'))} ({share_text})")
    return lines


def _vocabulary_lines(database: MaterialDatabase) -> list[str]:
    """The canonical material vocabulary and its dataset notes.

    Shared by the image and video prompts so both offer Gemini exactly the same
    closed vocabulary from the same ``flame_dataset.json``.
    """
    lines = ["", f"Canonical material vocabulary ({len(database)} categories):"]
    lines.extend(f"{index}. {entry.name}" for index, entry in enumerate(database.entries, 1))
    lines.append("")
    lines.append("Dataset descriptions/notes for each canonical material:")
    lines.extend(f"- {entry.name}: {entry.notes or '(no notes)'}" for entry in database.entries)
    return lines


def _build_user_content(evidence: dict[str, Any], database: MaterialDatabase) -> str:
    """The user message: measured flame evidence + canonical vocabulary + rules.

    ``evidence`` is the dict built by :func:`app.flame_evidence.extract_flame_evidence`
    from the analyzer's own result; nothing is recomputed here.
    """
    mean_color = evidence.get("mean_color") or {}
    flame_region = evidence.get("flame_region") or {}
    bounding_box = evidence.get("detection_bounding_box") or {}
    distribution = evidence.get("color_distribution") or {}
    per_algorithm = distribution.get("per_algorithm") or {}

    lines = [
        "Measured flame evidence (observations of the flame, not the fuel):",
        "",
        f"- Final flame RGB: {_format_color_list(mean_color.get('rgb'))}",
        f"- Final flame LAB: {_format_color_list(mean_color.get('lab'))}",
    ]
    hsv = evidence.get("hsv")
    if isinstance(hsv, dict):
        lines.append(
            f"- Final flame HSV: H={hsv.get('h')} deg, S={hsv.get('s')}%, V={hsv.get('v')}%"
        )
    if evidence.get("brightness_0_255") is not None:
        lines.append(f"- Flame brightness (perceived luminance, 0-255): {evidence['brightness_0_255']}")
    if evidence.get("saturation_0_100") is not None:
        lines.append(f"- Flame saturation (0-100): {evidence['saturation_0_100']}")

    region_bits: list[str] = []
    if flame_region.get("flame_pixel_count") is not None:
        region_bits.append(f"{flame_region['flame_pixel_count']} flame pixels")
    if flame_region.get("mask_area_ratio") is not None:
        region_bits.append(f"{100.0 * float(flame_region['mask_area_ratio']):.2f}% of the image area")
    if flame_region.get("mask_width_px") is not None and flame_region.get("mask_height_px") is not None:
        region_bits.append(
            f"mask spans {flame_region['mask_width_px']}x{flame_region['mask_height_px']} px"
        )
    if region_bits:
        lines.append(f"- Flame region: {', '.join(region_bits)}")
    if flame_region.get("samples_used") is not None:
        sampled = " (subsampled)" if flame_region.get("pixels_sampled") else ""
        lines.append(f"- Colour samples: {flame_region['samples_used']}{sampled}")

    if evidence.get("detection_confidence") is not None:
        lines.append(f"- Fire detection confidence: {evidence['detection_confidence']}")
    if evidence.get("segmentation_confidence") is not None:
        lines.append(f"- Segmentation confidence: {evidence['segmentation_confidence']}")
    if bounding_box.get("width_px") is not None and bounding_box.get("height_px") is not None:
        aspect = bounding_box.get("aspect_ratio")
        aspect_text = f", aspect ratio {aspect}" if aspect is not None else ""
        lines.append(
            f"- Detection bounding box: {bounding_box['width_px']}x{bounding_box['height_px']} px{aspect_text}"
        )

    if per_algorithm:
        lines.append("- Colour distribution across the flame region (already calculated):")
        for method, record in per_algorithm.items():
            if not isinstance(record, dict):
                continue
            lines.append(
                f"  - {method}: {record.get('cluster_count')} clusters, "
                f"representative RGB {_format_color_list(record.get('representative_rgb'))} / "
                f"LAB {_format_color_list(record.get('representative_lab'))}"
            )
            dominant = record.get("dominant_color")
            if isinstance(dominant, dict):
                lines.append(
                    f"    dominant cluster: RGB {_format_color_list(dominant.get('rgb'))} / "
                    f"LAB {_format_color_list(dominant.get('lab'))}"
                )
            cluster_lines = _format_clusters(record.get("clusters"))
            if cluster_lines:
                lines.append("    clusters: " + "; ".join(cluster_lines))
    elif distribution.get("skipped_reason"):
        lines.append(f"- Colour distribution: not calculated ({distribution['skipped_reason']})")

    lines.extend(_vocabulary_lines(database))
    lines.extend(
        [
            "",
            "Instructions:",
            "- Evaluate whether this evidence distinguishes among the candidate materials.",
            "- Flame color is weak evidence for material identity; do not select a material merely "
            "because its documented flame color is closest to the measurement.",
            "- If several materials remain plausible, return them as ranked candidates with low or "
            "medium confidence and set uncertain to true.",
            "- If the evidence cannot reliably distinguish materials, set primary_material to null, "
            "uncertain to true, overall_confidence_level to \"low\" and evidence_quality to "
            "\"insufficient\" (or \"limited\" if some weak signal exists).",
            "- confidence_percent expresses how strongly the evidence supports each candidate, not "
            "color closeness; it is a heuristic score, not a calibrated probability.",
            f"- Return exactly {REQUIRED_CANDIDATES} ranked candidates, copying every material name "
            "verbatim from the canonical vocabulary above.",
        ]
    )
    return "\n".join(lines)


def _format_frame_evidence(frame: dict[str, Any]) -> str:
    """One line per frame: its measured colour and whatever else was supplied."""
    mean_color = frame.get("mean_color") or {}
    position = f"frame {frame.get('frame_index')}"
    if frame.get("timestamp_seconds") is not None:
        position += f" at t={frame['timestamp_seconds']}s"

    bits = [f"- {position}: RGB {_format_color_list(mean_color.get('rgb'))}"]
    bits.append(f"LAB {_format_color_list(mean_color.get('lab'))}")
    hsv = frame.get("hsv")
    if isinstance(hsv, dict):
        bits.append(f"HSV H={hsv.get('h')} deg S={hsv.get('s')}% V={hsv.get('v')}%")
    if frame.get("brightness_0_255") is not None:
        bits.append(f"brightness {frame['brightness_0_255']} (0-255)")
    if frame.get("saturation_0_100") is not None:
        bits.append(f"saturation {frame['saturation_0_100']}%")
    if frame.get("detection_confidence") is not None:
        bits.append(f"detection confidence {frame['detection_confidence']}")
    if frame.get("segmentation_confidence") is not None:
        bits.append(f"segmentation confidence {frame['segmentation_confidence']}")
    region = frame.get("flame_region") or {}
    if region.get("mask_area_ratio") is not None:
        bits.append(f"flame area {100.0 * float(region['mask_area_ratio']):.2f}% of the frame")
    return ", ".join(bits)


def _format_collection_summary(collection: dict[str, Any]) -> list[str]:
    """Whole-collection statistics: consistency and amount of evidence."""
    lines: list[str] = []
    lines.append(
        f"- Observations analysed: {collection.get('frame_count')} frames of one fire event"
    )
    if collection.get("time_span_seconds") is not None:
        lines.append(f"- Observed time span: {collection['time_span_seconds']}s")
    lines.append(f"- Mean RGB across the frames: {_format_color_list(collection.get('mean_rgb'))}")
    lines.append(f"- Mean LAB across the frames: {_format_color_list(collection.get('mean_lab'))}")
    lines.append(
        f"- Per-channel LAB standard deviation across the frames "
        f"(0 means every frame measured the same colour): "
        f"{_format_color_list(collection.get('lab_std_dev'))}"
    )
    if collection.get("mean_detection_confidence") is not None:
        lines.append(
            f"- Mean fire detection confidence across the frames: "
            f"{collection['mean_detection_confidence']}"
        )
    if collection.get("mean_segmentation_confidence") is not None:
        lines.append(
            f"- Mean flame segmentation confidence across the frames: "
            f"{collection['mean_segmentation_confidence']}"
        )
    if collection.get("mean_flame_area_ratio") is not None:
        lines.append(
            f"- Mean flame area across the frames: "
            f"{100.0 * float(collection['mean_flame_area_ratio']):.2f}% of the frame"
        )
    return lines


def _build_video_user_content(evidence: dict[str, Any], database: MaterialDatabase) -> str:
    """The user message for a whole video: every frame's evidence, plus the rules.

    ``evidence`` is the collection built by
    :func:`app.video_material.build_video_evidence`.  Every analysed frame appears
    individually, and the cross-frame statistics are supplied alongside so the
    model can judge consistency - the consolidation
    :data:`VIDEO_CONSOLIDATION_INSTRUCTION` demands.  Only numbers are sent: no
    image, no mask, no base64, not even a filename.
    """
    frames = evidence.get("frames") or []
    collection = evidence.get("collection") or {}

    lines = [
        f"Structured flame evidence for one video: {evidence.get('frame_count', len(frames))} "
        "frames sampled from a single fire event.",
        "These are numerical measurements of the flame, not the fuel, and no images are supplied.",
        "",
        "Per-frame observations:",
    ]
    lines.extend(_format_frame_evidence(frame) for frame in frames)

    lines.append("")
    lines.append("Whole-collection statistics (measured across all frames, not a per-frame verdict):")
    lines.extend(_format_collection_summary(collection))

    lines.extend(_vocabulary_lines(database))
    lines.extend(
        [
            "",
            VIDEO_CONSOLIDATION_INSTRUCTION,
            "",
            "Instructions:",
            "- Evaluate the complete collection of frames jointly as one fire event.",
            "- Do not classify the frames independently and do not average their confidence "
            "percentages; return ONE consolidated material assessment for the video.",
            "- Let the consistency of the observations across the frames, the amount of evidence "
            "and the measurement confidences drive the reported confidence and "
            "evidence_quality.",
            "- Flame color is weak evidence for material identity; do not select a material merely "
            "because its documented flame color is closest to the measurements.",
            "- If several materials remain plausible, return them as ranked candidates with low or "
            "medium confidence and set uncertain to true.",
            "- If the observations cannot reliably distinguish materials, set primary_material to "
            "null, uncertain to true, overall_confidence_level to \"low\" and evidence_quality to "
            "\"insufficient\" (or \"limited\" if some weak signal exists).",
            "- confidence_percent expresses how strongly the complete set of observations supports "
            "each candidate, not color closeness; it is a heuristic score, not a calibrated "
            "probability.",
            f"- Return exactly {REQUIRED_CANDIDATES} ranked candidates, copying every material name "
            "verbatim from the canonical vocabulary above.",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response validation (hallucination guard)
# ---------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_payload(
    payload: Any, canonical: set[str], expected: int
) -> tuple[list[GeminiMatch], str | None, bool]:
    """Validate Gemini's structured payload against the canonical vocabulary.

    Returns ``(matches, error, primary_is_null)``: ``(None, reason, False)`` on
    the first validation failure, or the validated candidates with a flag
    telling the caller whether ``primary_material`` was ``null``.  Never raises.
    """
    if not isinstance(payload, dict):
        return [], "the response is not a JSON object", False

    primary = payload.get("primary_material")
    primary_is_null = primary is None
    if primary is not None and (not isinstance(primary, str) or primary not in canonical):
        return [], "primary_material is not a canonical material name", False

    matches = payload.get("matches")
    if not isinstance(matches, list):
        return [], "matches is not a list", False
    if len(matches) != expected:
        return [], f"expected exactly {expected} candidates, got {len(matches)}", False

    seen: set[str] = set()
    validated: list[GeminiMatch] = []
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            return [], f"candidate {index + 1} is not an object", False
        rank = match.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            return [], f"candidate {index + 1} has a non-integer rank", False
        material = match.get("material")
        if not isinstance(material, str) or material not in canonical:
            return [], f"candidate {index + 1} material {material!r} is not in the canonical vocabulary", False
        if material in seen:
            return [], f"duplicate material {material!r}", False
        seen.add(material)
        confidence = match.get("confidence_percent")
        if not _is_number(confidence) or not 0.0 <= float(confidence) <= 100.0:
            return [], f"candidate {index + 1} confidence_percent is outside [0, 100]", False
        reason = match.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return [], f"candidate {index + 1} has an empty reason", False
        validated.append(
            GeminiMatch(
                rank=rank,
                material=material,
                confidence_percent=round(float(confidence), 1),
                reason=reason,
            )
        )

    ranks = sorted(match.rank for match in validated)
    if ranks != list(range(1, expected + 1)):
        return [], f"ranks must be exactly 1..{expected}, got {ranks}", False
    if not primary_is_null and validated[0].material != primary:
        return [], "rank 1 must be the primary_material", False

    level = payload.get("overall_confidence_level")
    if level not in CONFIDENCE_LEVELS:
        return [], "overall_confidence_level must be one of high|medium|low", False
    if not isinstance(payload.get("uncertain"), bool):
        return [], "uncertain must be a boolean", False
    quality = payload.get("evidence_quality")
    if quality not in EVIDENCE_QUALITY_LEVELS:
        return [], "evidence_quality must be one of insufficient|limited|moderate|strong", False
    summary = payload.get("reasoning_summary")
    if not isinstance(summary, str) or not summary.strip():
        return [], "reasoning_summary must be a non-empty string", False

    # Uncertainty consistency: a null primary is the model saying "the evidence
    # cannot distinguish materials", which by contract means an uncertain,
    # low-confidence answer.  A response that claims a null primary while
    # asserting certainty contradicts itself and is rejected.
    if primary_is_null and (payload["uncertain"] is not True or level != "low"):
        return [], "a null primary_material requires uncertain=true and overall_confidence_level=low", False

    return validated, None, primary_is_null


# ---------------------------------------------------------------------------
# Shared request/response plumbing
# ---------------------------------------------------------------------------
def _prepare_database(
    settings: Settings, database: MaterialDatabase | None
) -> tuple[MaterialDatabase | None, dict[str, Any] | None]:
    """Apply the guard clauses shared by both Gemini entry points.

    Returns ``(database, failure)``: exactly one of the two is ``None``.  The
    checks run in the same order for images and videos - disabled, then missing
    key, then database - so both entry points fail the same way for the same
    reason.
    """
    if not settings.gemini_enabled:
        return None, _unavailable(
            "Gemini material analysis is disabled (GEMINI_ENABLED is not enabled)."
        )
    if not settings.gemini_api_key:
        return None, _unavailable("GEMINI_API_KEY is not configured.")

    try:
        return (database if database is not None else load_material_database(settings.dataset_path), None)
    except MissingDatabaseError as exc:
        return None, _unavailable(f"The material database could not be loaded: {exc}")


def _expected_candidates(db: MaterialDatabase) -> tuple[set[str], int, dict[str, Any] | None]:
    """The canonical vocabulary and the number of candidates to expect from it."""
    canonical = {entry.name for entry in db.entries}
    expected = min(REQUIRED_CANDIDATES, len(canonical))
    if expected == 0:
        return canonical, expected, _unavailable("The material database contains no materials.")
    return canonical, expected, None


def _call_gemini(
    contents: str, settings: Settings, system_instruction: str
) -> tuple[Any | None, dict[str, Any] | None]:
    """Perform **one** structured Gemini request.

    Returns ``(response, failure)``: exactly one of the two is ``None``.  Every
    transport failure (timeout, quota, auth, network, SDK) becomes a clean
    unavailable payload rather than an exception.
    """
    try:
        response = _run_with_timeout(
            lambda: _generate(_create_client(settings), settings, contents, system_instruction),
            settings.gemini_timeout_s,
        )
    except GeminiTimeoutError:
        logger.warning("Gemini request timed out after %ss", settings.gemini_timeout_s)
        return None, _unavailable(
            f"The Gemini request timed out after {settings.gemini_timeout_s:g}s."
        )
    except Exception as exc:  # noqa: BLE001 - quota, auth, network, SDK errors: all become unavailable
        logger.warning("Gemini request failed: %s: %s", type(exc).__name__, exc)
        return None, _unavailable(f"The Gemini API request failed ({type(exc).__name__}).")
    return response, None


def _validated_result(response: Any, canonical: set[str], expected: int) -> dict[str, Any]:
    """Validate Gemini's answer and build the canonical result.

    Identical for images and videos: the structured contract, the canonical
    vocabulary, the rank sequence, the confidence range and the uncertainty
    consistency rules are the same, so the same result shape comes back either
    way.  Nothing is averaged, selected or adjusted here - Gemini's own
    confidence is passed through unchanged.
    """
    try:
        payload = json.loads(getattr(response, "text", None) or "")
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Gemini response was not valid JSON")
        return _unavailable("The Gemini response was not valid JSON.")

    matches, error, _primary_is_null = _validate_payload(payload, canonical, expected)
    if error is not None:
        logger.warning("Gemini response failed validation: %s", error)
        return _unavailable(f"The Gemini response failed validation: {error}.")

    return {
        "available": True,
        "primary_material": payload["primary_material"],
        "matches": [match.to_dict() for match in matches],
        "overall_confidence_level": payload["overall_confidence_level"],
        "uncertain": payload["uncertain"],
        "evidence_quality": payload["evidence_quality"],
        "reasoning_summary": payload["reasoning_summary"],
    }


# ---------------------------------------------------------------------------
# Public entry point - single image
# ---------------------------------------------------------------------------
def gemini_material_analysis(
    evidence: dict[str, Any],
    settings: Settings,
    database: MaterialDatabase | None = None,
) -> dict[str, Any]:
    """Run the secondary Gemini material analysis for one extracted flame.

    Parameters
    ----------
    evidence:
        The flame evidence built by
        :func:`app.flame_evidence.extract_flame_evidence` from the analyzer's
        own result (final RGB/LAB, HSV, brightness, saturation, region
        statistics, confidences, bounding box, colour clusters).  Used
        verbatim - never recomputed.
    settings:
        Active configuration (``gemini_enabled``, ``gemini_api_key``,
        ``gemini_model``, ``gemini_timeout_s``).
    database:
        The loaded material database.  Loaded from ``settings.dataset_path``
        when not supplied.

    Returns
    -------
    dict
        ``{"available": True, ...}`` with the validated candidates (and
        ``primary_material: null`` when the model reports insufficient
        evidence), or ``{"available": False, "error": "..."}``.  This function
        never raises: the deterministic analysis must survive any Gemini
        failure.
    """
    db, failure = _prepare_database(settings, database)
    if failure is not None:
        return failure

    canonical, expected, failure = _expected_candidates(db)
    if failure is not None:
        return failure

    if not isinstance(evidence, dict) or not evidence.get("mean_color"):
        return _unavailable("No flame colour evidence was extracted for the AI analysis.")

    contents = _build_user_content(evidence, db)
    response, failure = _call_gemini(contents, settings, SYSTEM_INSTRUCTION)
    if failure is not None:
        return failure

    return _validated_result(response, canonical, expected)


# ---------------------------------------------------------------------------
# Public entry point - whole video, one consolidated result
# ---------------------------------------------------------------------------
def gemini_video_material_analysis(
    evidence: dict[str, Any],
    settings: Settings,
    database: MaterialDatabase | None = None,
) -> dict[str, Any]:
    """Run **one** Gemini request over a whole collection of frame observations.

    The supplied evidence is the collection built by
    :func:`app.video_material.build_video_evidence`: every analysed frame with its
    measured RGB/LAB, HSV, brightness, saturation, detection/segmentation
    confidence and flame-area ratio, plus the cross-frame statistics.

    What this function guarantees:

    * **Exactly one** ``generate_content`` call for the entire collection, and
      exactly one consolidated result.  It never calls Gemini per frame, never
      averages per-frame confidences or classifications, and never picks the
      first or the highest-confidence frame's answer - those are all
      per-frame strategies this entry point exists to replace.
    * The prompt carries :data:`VIDEO_CONSOLIDATION_INSTRUCTION`, so the model is
      told to judge the collection jointly and to let consistency, quality and
      amount of evidence drive the confidence.
    * **No images, masks or base64** are sent, and the frames are not re-encoded
      or re-measured here; only the supplied numbers are rendered into the prompt.
    * Gemini's ``confidence_percent`` is returned unchanged, including values
      below any client display threshold.

    Parameters
    ----------
    evidence:
        The whole-collection evidence payload (see
        :func:`app.video_material.build_video_evidence`).
    settings:
        Active configuration (``gemini_enabled``, ``gemini_api_key``,
        ``gemini_model``, ``gemini_timeout_s``) - the same Gemini configuration
        the image workflow uses.
    database:
        The loaded material database.  Loaded from ``settings.dataset_path``
        when not supplied.

    Returns
    -------
    dict
        The same structure :func:`gemini_material_analysis` returns: one
        consolidated ``{"available": True, ...}`` result, or
        ``{"available": False, "error": "..."}``.  Never raises.
    """
    db, failure = _prepare_database(settings, database)
    if failure is not None:
        return failure

    canonical, expected, failure = _expected_candidates(db)
    if failure is not None:
        return failure

    frames = evidence.get("frames") if isinstance(evidence, dict) else None
    if not isinstance(frames, list) or not frames:
        return _unavailable("No flame colour evidence was supplied for the AI analysis.")

    contents = _build_video_user_content(evidence, db)
    # One request for the whole collection.  See the module docstring.
    response, failure = _call_gemini(contents, settings, VIDEO_SYSTEM_INSTRUCTION)
    if failure is not None:
        return failure

    return _validated_result(response, canonical, expected)
