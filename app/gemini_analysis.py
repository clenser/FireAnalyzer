"""Secondary, Gemini-powered material identification.

This module is an *independent* second opinion on the burning material.  It is
strictly additive:

* The deterministic :mod:`app.material_matching` result is never touched.
* No image, no segmentation mask and no base64 data is sent to Gemini - only the
  final whole-mask flame colour (mean RGB and mean LAB) that
  :class:`app.analyzer.FlameAnalyzer` already extracted, plus the canonical
  material vocabulary loaded from the same ``flame_dataset.json``.
* Gemini may only choose from the canonical dataset categories.  Every response
  is validated against the vocabulary, the rank sequence and the confidence
  range; anything malformed becomes a clean ``{"available": false, ...}``
  state instead of leaking malformed output.
* A failure here (disabled, missing key, quota, timeout, bad payload) never
  fails the ``/analyze`` request - the deterministic analysis is returned
  untouched.

The ``confidence_percent`` values are the model's *relative heuristic*
confidence allocation.  They are not calibrated probabilities and are never
derived from token probabilities.
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
    "REQUIRED_CANDIDATES",
    "GeminiMatch",
    "GeminiTimeoutError",
    "gemini_material_analysis",
]

logger = logging.getLogger(__name__)

#: Number of ranked candidates requested from (and validated from) Gemini.  The
#: dataset holds 18 materials, so five are always available; if a custom
#: dataset ever held fewer, that many are required instead.
REQUIRED_CANDIDATES = 5

#: Strict system instruction.  It fixes the vocabulary rule, the "no certainty"
#: caveat, the five-candidate shape and the "heuristic, not probabilities"
#: meaning of the percentage values.
SYSTEM_INSTRUCTION = """You are a flame-color material analysis assistant.

Identify possible burning material categories using ONLY the supplied
canonical material vocabulary and the measured RGB/LAB flame-color values.

You are analyzing flame appearance only. Flame color alone is not a
reliable physical identification of fuel/material, so do not claim certainty.

Do not invent materials. Every material name you return must be copied
verbatim from the supplied canonical material vocabulary.

Return exactly five ranked candidates when five canonical candidates
are available.

The percentage values are heuristic confidence scores, NOT calibrated
probabilities and must not be described as probabilities. They are a relative
confidence allocation across the five candidates, not experimentally measured
likelihoods.

Use the supplied RGB and CIELAB measurements as the primary evidence.

Prefer candidates whose documented flame characteristics are compatible
with the measured color.

When the evidence is weak or ambiguous, explicitly say so in the result."""

#: Structured-output contract enforced by the SDK (``response_mime_type`` +
#: ``response_schema``) and re-validated locally after the call.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_material": {"type": "string"},
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
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "primary_material",
        "matches",
        "overall_confidence_level",
        "uncertain",
        "reasoning_summary",
    ],
}

_CONFIDENCE_LEVELS = ("high", "medium", "low")


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


def _generate(client: Any, settings: Settings, contents: str) -> Any:
    """One structured-output content generation call."""
    from google.genai import types

    return client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
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
def _build_user_content(mean_color: dict[str, Any], database: MaterialDatabase) -> str:
    """The user message: measured colour evidence + canonical vocabulary + notes.

    ``mean_color`` is the analyzer's own ``flame_analysis.mean_color`` record;
    nothing is recomputed here.
    """
    lines = [
        f"Final Flame RGB: {mean_color.get('rgb')}",
        f"Final Flame LAB: {mean_color.get('lab')}",
        "",
        f"Canonical material vocabulary ({len(database)} categories):",
    ]
    lines.extend(f"{index}. {entry.name}" for index, entry in enumerate(database.entries, 1))
    lines.append("")
    lines.append("Dataset descriptions/notes for each canonical material:")
    lines.extend(
        f"- {entry.name}: {entry.notes or '(no notes)'}" for entry in database.entries
    )
    lines.append("")
    lines.append(
        f"Return exactly {REQUIRED_CANDIDATES} ranked candidates, copying every "
        "material name verbatim from the canonical vocabulary above."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response validation (hallucination guard)
# ---------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_payload(
    payload: Any, canonical: set[str], expected: int
) -> tuple[list[GeminiMatch], str | None]:
    """Validate Gemini's structured payload against the canonical vocabulary.

    Returns ``(matches, None)`` on success or ``(None, reason)`` on the first
    validation failure.  Never raises.
    """
    if not isinstance(payload, dict):
        return [], "the response is not a JSON object"

    primary = payload.get("primary_material")
    if not isinstance(primary, str) or primary not in canonical:
        return [], "primary_material is not a canonical material name"

    matches = payload.get("matches")
    if not isinstance(matches, list):
        return [], "matches is not a list"
    if len(matches) != expected:
        return [], f"expected exactly {expected} candidates, got {len(matches)}"

    seen: set[str] = set()
    validated: list[GeminiMatch] = []
    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            return [], f"candidate {index + 1} is not an object"
        rank = match.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            return [], f"candidate {index + 1} has a non-integer rank"
        material = match.get("material")
        if not isinstance(material, str) or material not in canonical:
            return [], f"candidate {index + 1} material {material!r} is not in the canonical vocabulary"
        if material in seen:
            return [], f"duplicate material {material!r}"
        seen.add(material)
        confidence = match.get("confidence_percent")
        if not _is_number(confidence) or not 0.0 <= float(confidence) <= 100.0:
            return [], f"candidate {index + 1} confidence_percent is outside [0, 100]"
        reason = match.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return [], f"candidate {index + 1} has an empty reason"
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
        return [], f"ranks must be exactly 1..{expected}, got {ranks}"
    if validated[0].material != primary:
        return [], "rank 1 must be the primary_material"

    level = payload.get("overall_confidence_level")
    if level not in _CONFIDENCE_LEVELS:
        return [], "overall_confidence_level must be one of high|medium|low"
    if not isinstance(payload.get("uncertain"), bool):
        return [], "uncertain must be a boolean"
    summary = payload.get("reasoning_summary")
    if not isinstance(summary, str) or not summary.strip():
        return [], "reasoning_summary must be a non-empty string"

    return validated, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def gemini_material_analysis(
    mean_color: dict[str, Any],
    settings: Settings,
    database: MaterialDatabase | None = None,
) -> dict[str, Any]:
    """Run the secondary Gemini material analysis for one extracted flame colour.

    Parameters
    ----------
    mean_color:
        The analyzer's own ``flame_analysis.mean_color`` record
        (``{"rgb": [R, G, B], "lab": [L, A, B]}``).  Used verbatim - never
        recomputed.
    settings:
        Active configuration (``gemini_enabled``, ``gemini_api_key``,
        ``gemini_model``, ``gemini_timeout_s``).
    database:
        The loaded material database.  Loaded from ``settings.dataset_path``
        when not supplied.

    Returns
    -------
    dict
        ``{"available": True, ...}`` with the validated candidates, or
        ``{"available": False, "error": "..."}``.  This function never raises:
        the deterministic analysis must survive any Gemini failure.
    """
    if not settings.gemini_enabled:
        return _unavailable("Gemini material analysis is disabled (GEMINI_ENABLED is not enabled).")
    if not settings.gemini_api_key:
        return _unavailable("GEMINI_API_KEY is not configured.")

    try:
        db = database if database is not None else load_material_database(settings.dataset_path)
    except MissingDatabaseError as exc:
        return _unavailable(f"The material database could not be loaded: {exc}")

    canonical = {entry.name for entry in db.entries}
    expected = min(REQUIRED_CANDIDATES, len(canonical))
    if expected == 0:
        return _unavailable("The material database contains no materials.")

    contents = _build_user_content(mean_color, db)

    def request() -> Any:
        return _generate(_create_client(settings), settings, contents)

    try:
        response = _run_with_timeout(request, settings.gemini_timeout_s)
    except GeminiTimeoutError:
        logger.warning("Gemini request timed out after %ss", settings.gemini_timeout_s)
        return _unavailable(f"The Gemini request timed out after {settings.gemini_timeout_s:g}s.")
    except Exception as exc:  # noqa: BLE001 - quota, auth, network, SDK errors: all become unavailable
        logger.warning("Gemini request failed: %s: %s", type(exc).__name__, exc)
        return _unavailable(f"The Gemini API request failed ({type(exc).__name__}).")

    try:
        payload = json.loads(getattr(response, "text", None) or "")
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Gemini response was not valid JSON")
        return _unavailable("The Gemini response was not valid JSON.")

    matches, error = _validate_payload(payload, canonical, expected)
    if error is not None:
        logger.warning("Gemini response failed validation: %s", error)
        return _unavailable(f"The Gemini response failed validation: {error}.")

    return {
        "available": True,
        "primary_material": payload["primary_material"],
        "matches": [match.to_dict() for match in matches],
        "overall_confidence_level": payload["overall_confidence_level"],
        "uncertain": payload["uncertain"],
        "reasoning_summary": payload["reasoning_summary"],
    }
