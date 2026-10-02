"""Vision material evidence: Groq (primary) with Gemini vision fallback.

AI is an **evidence provider only**.  A vision model is shown the segmented
flame region and asked for ranked candidate materials from the canonical
``flame_dataset.json`` vocabulary.  It never decides the final material - that
is :mod:`app.material_fusion`'s job.

Flow (:func:`collect_vision_evidence`)::

    masked flame crop (merged mask, dimmed surroundings, downscaled JPEG)
      -> Groq  (GROQ_API_KEY + GROQ_MODEL)
      -> on ANY failure (missing key/model, network, timeout, HTTP error,
         invalid JSON, schema violation, non-canonical material):
         Gemini vision (GEMINI_ENABLED + GEMINI_API_KEY + GEMINI_MODEL)
      -> on failure too: ``vision_provider: "none"`` and the deterministic
         analysis continues untouched.

Every provider answer is validated strictly by :func:`validate_vision_payload`.
No API key ever appears in a returned structure or in a log line: failures are
reported as short, fixed reason codes.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .config import Settings
from .material_matching import MaterialDatabase

__all__ = [
    "VISION_PROVIDERS",
    "VisionProviderError",
    "build_flame_crop",
    "vision_prompt",
    "parse_json_object",
    "validate_vision_payload",
    "collect_vision_evidence",
    "vision_evidence_for_crop",
    "safe_flame_crop",
    "unavailable_vision",
]

logger = logging.getLogger(__name__)

VISION_PROVIDERS = ("groq", "gemini", "none")

_MAX_CANDIDATES = 5
_UNCERTAINTY_LEVELS = ("low", "medium", "high")
_MAX_TEXT = 300
_CROP_PADDING = 0.25
_CROP_MAX_SIDE = 768
_CROP_JPEG_QUALITY = 85
#: Brightness kept outside the flame mask: enough to show the burning object
#: (the strongest fuel cue), dim enough that the flame is clearly the subject.
_BACKGROUND_KEEP = 0.45


class VisionProviderError(Exception):
    """A provider failed.  ``reason`` is a short, secret-free code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------
def build_flame_crop(image_bgr: np.ndarray, mask: np.ndarray | None) -> tuple[bytes, dict[str, Any]] | None:
    """JPEG of the flame region cut from the merged mask.

    The crop is the mask's bounding rectangle plus padding; pixels outside the
    mask are dimmed, not removed, so the burning object stays visible.  Returns
    ``None`` when there is no usable mask.
    """
    if image_bgr is None or mask is None or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    height, width = image_bgr.shape[:2]
    y1, y2, x1, x2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    pad_y = int(round((y2 - y1) * _CROP_PADDING))
    pad_x = int(round((x2 - x1) * _CROP_PADDING))
    y1, y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
    x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)

    crop = image_bgr[y1:y2, x1:x2].astype(np.float32)
    inside = (mask[y1:y2, x1:x2] > 0)[..., None]
    crop = np.where(inside, crop, crop * _BACKGROUND_KEEP).clip(0, 255).astype(np.uint8)

    longest = max(crop.shape[:2])
    if longest > _CROP_MAX_SIDE:
        scale = _CROP_MAX_SIDE / float(longest)
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), _CROP_JPEG_QUALITY])
    if not ok:
        return None
    return buffer.tobytes(), {
        "type": "masked_flame_crop",
        "width": int(crop.shape[1]),
        "height": int(crop.shape[0]),
    }


# ---------------------------------------------------------------------------
# Prompt and validation (shared by both providers)
# ---------------------------------------------------------------------------
def vision_prompt(materials: Sequence[str]) -> str:
    """The single evidence-extraction prompt both providers receive."""
    vocabulary = "\n".join(f"- {name}" for name in materials)
    return f"""You are a visual evidence extractor in a fire-analysis system.
The image is a crop around a segmented flame region; pixels outside the flame
mask are dimmed so the flame is the subject, but the burning object stays visible.

Report which burning materials the VISIBLE evidence supports (flame colour,
smoke colour and density, soot, flame shape and steadiness, and the burning
object or container if it is visible).  You do NOT decide the final answer:
a deterministic program combines your evidence with colour measurements.

Allowed material names (copy them EXACTLY, character for character):
{vocabulary}

Return ONLY a JSON object, no markdown, with this structure:
{{
  "candidates": [
    {{"material": "<exact allowed name>", "confidence": <number 0.0-1.0>,
      "evidence": "<one short sentence about what you see>",
      "uncertainty": "low" | "medium" | "high"}}
  ],
  "uncertain": <true|false>,
  "observations": "<one or two neutral sentences describing the flame>"
}}

Rules:
- 1 to {_MAX_CANDIDATES} candidates, most supported first, each material at most once.
- "confidence" values are your relative support for each candidate; they must sum to at most 1.0.
- Use "uncertain": true when the image cannot distinguish the materials
  (e.g. only flame colour is visible, overexposed, smoke-obscured, tiny flame).
- Never invent a material name outside the allowed list."""


def parse_json_object(text: Any) -> dict[str, Any] | None:
    """Parse a JSON object from model text, tolerating a markdown fence."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _canonical_lookup(materials: Sequence[str]) -> dict[str, str]:
    return {" ".join(name.split()).casefold(): name for name in materials}


def _text(value: Any) -> str:
    return " ".join(str(value).split())[:_MAX_TEXT] if isinstance(value, str) else ""


def validate_vision_payload(
    payload: Any, materials: Sequence[str]
) -> tuple[dict[str, Any] | None, str | None]:
    """Strictly validate a provider answer.

    Returns ``(normalised, None)`` or ``(None, reason)``.  Rejected: a non-object,
    missing/empty/oversized ``candidates``, a material outside the canonical
    vocabulary (only case/whitespace differences are tolerated), duplicated
    materials, a confidence that is not a number in ``[0, 1]``, an unknown
    uncertainty level, or a non-boolean ``uncertain``.  Confidences that sum
    above 1 are rescaled proportionally to a distribution.
    """
    if not isinstance(payload, dict):
        return None, "not_an_object"
    raw = payload.get("candidates")
    if not isinstance(raw, list) or not raw:
        return None, "missing_candidates"
    if len(raw) > _MAX_CANDIDATES:
        return None, "too_many_candidates"

    lookup = _canonical_lookup(materials)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            return None, "candidate_not_an_object"
        name = item.get("material")
        if not isinstance(name, str):
            return None, "material_not_a_string"
        canonical = lookup.get(" ".join(name.split()).casefold())
        if canonical is None:
            return None, "non_canonical_material"
        if canonical in seen:
            return None, "duplicate_material"
        seen.add(canonical)
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None, "confidence_not_a_number"
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0) or confidence != confidence:
            return None, "confidence_out_of_range"
        uncertainty = item.get("uncertainty", "medium")
        if not isinstance(uncertainty, str) or uncertainty.strip().lower() not in _UNCERTAINTY_LEVELS:
            return None, "invalid_uncertainty"
        candidates.append(
            {
                "material": canonical,
                "confidence": confidence,
                "evidence": _text(item.get("evidence")),
                "uncertainty": uncertainty.strip().lower(),
            }
        )

    uncertain = payload.get("uncertain", False)
    if not isinstance(uncertain, bool):
        return None, "uncertain_not_boolean"

    total = sum(candidate["confidence"] for candidate in candidates)
    if total > 1.0:
        for candidate in candidates:
            candidate["confidence"] = candidate["confidence"] / total
    for candidate in candidates:
        candidate["confidence"] = round(candidate["confidence"], 4)
    # Highest support first; provider order breaks exact ties.
    candidates = [
        candidate
        for _, candidate in sorted(
            enumerate(candidates), key=lambda pair: (-pair[1]["confidence"], pair[0])
        )
    ]
    return {
        "candidates": candidates,
        "uncertain": uncertain,
        "observations": _text(payload.get("observations")),
    }, None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def unavailable_vision(reason: str, attempts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The ``vision_provider: none`` evidence block."""
    return {
        "vision_provider": "none",
        "available": False,
        "model": None,
        "candidates": [],
        "uncertain": True,
        "observations": "",
        "attempts": attempts or [],
        "image": None,
        "reason": reason,
    }


ProviderCall = Callable[[bytes, str, Settings], tuple[Any, str]]


def _attempt(
    name: str,
    call: ProviderCall,
    image: bytes,
    prompt: str,
    settings: Settings,
    materials: Sequence[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    model: str | None = None
    try:
        raw, model = call(image, prompt, settings)
        validated, reason = validate_vision_payload(parse_json_object(raw), materials)
        if validated is None:
            raise VisionProviderError(reason or "invalid_response")
    except VisionProviderError as exc:
        logger.warning("Vision provider %s failed (%s)", name, exc.reason)
        return None, {
            "provider": name,
            "status": "failed",
            "reason": exc.reason,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    except Exception as exc:  # noqa: BLE001 - provider failures never propagate
        logger.warning("Vision provider %s failed unexpectedly (%s)", name, type(exc).__name__)
        return None, {
            "provider": name,
            "status": "failed",
            "reason": "unexpected_error",
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    validated["model"] = model
    return validated, {
        "provider": name,
        "status": "ok",
        "reason": None,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def safe_flame_crop(image_bgr: np.ndarray | None, mask: np.ndarray | None) -> tuple[bytes, dict[str, Any]] | None:
    """:func:`build_flame_crop` that never raises."""
    try:
        return build_flame_crop(image_bgr, mask)
    except Exception:  # noqa: BLE001
        logger.exception("Could not build the flame crop for vision evidence")
        return None


def collect_vision_evidence(
    image_bgr: np.ndarray | None,
    mask: np.ndarray | None,
    settings: Settings,
    database: MaterialDatabase,
) -> dict[str, Any]:
    """Groq first, Gemini vision on any Groq failure, ``none`` if both fail.

    Never raises.  The result is evidence only; see the module docstring.
    """
    if not settings.vision_enabled:
        return unavailable_vision("vision evidence is disabled (FLAME_VISION_ENABLED=0)")
    return vision_evidence_for_crop(safe_flame_crop(image_bgr, mask), settings, database)


def vision_evidence_for_crop(
    crop: tuple[bytes, dict[str, Any]] | None,
    settings: Settings,
    database: MaterialDatabase,
) -> dict[str, Any]:
    """Same as :func:`collect_vision_evidence` for an already built crop."""
    from .gemini_vision_analysis import call_gemini_vision
    from .groq_analysis import call_groq_vision

    if not settings.vision_enabled:
        return unavailable_vision("vision evidence is disabled (FLAME_VISION_ENABLED=0)")
    if crop is None:
        return unavailable_vision("no flame region was available to send to a vision model")
    image_bytes, image_info = crop

    materials = [entry.name for entry in database.entries]
    prompt = vision_prompt(materials)
    attempts: list[dict[str, Any]] = []
    for name, call in (("groq", call_groq_vision), ("gemini", call_gemini_vision)):
        validated, record = _attempt(name, call, image_bytes, prompt, settings, materials)
        attempts.append(record)
        if validated is not None:
            return {
                "vision_provider": name,
                "available": True,
                "model": validated.pop("model", None),
                "candidates": validated["candidates"],
                "uncertain": validated["uncertain"],
                "observations": validated["observations"],
                "attempts": attempts,
                "image": image_info,
                "reason": None,
            }
    result = unavailable_vision("all vision providers failed; deterministic evidence only", attempts)
    result["image"] = image_info
    return result
