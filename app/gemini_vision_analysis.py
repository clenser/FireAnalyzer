"""Gemini vision call (fallback vision evidence provider).

Used only when Groq fails.  Reuses the existing Gemini configuration
(``GEMINI_ENABLED``, ``GEMINI_API_KEY``, ``GEMINI_MODEL``, ``GEMINI_TIMEOUT_S``)
and the client/timeout helpers of :mod:`app.gemini_analysis`, which itself stays
untouched: the numerical Gemini analysis is a separate evidence stream.

Failures are raised as :class:`~app.vision_providers.VisionProviderError`
with a short reason code; exception messages (which could echo request
details) are never surfaced.
"""

from __future__ import annotations

from .config import Settings
from .gemini_analysis import GeminiTimeoutError, _create_client, _run_with_timeout
from .vision_providers import VisionProviderError

__all__ = ["call_gemini_vision"]


def call_gemini_vision(image_jpeg: bytes, prompt: str, settings: Settings) -> tuple[str, str]:
    """Send one image + prompt to Gemini; return ``(response_text, model)``."""
    if not settings.gemini_enabled:
        raise VisionProviderError("disabled")
    if not settings.gemini_api_key:
        raise VisionProviderError("missing_api_key")
    model = settings.gemini_model

    def generate():
        from google.genai import types

        client = _create_client(settings)
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_s * 1000)),
            ),
        )

    try:
        response = _run_with_timeout(generate, settings.gemini_timeout_s)
    except GeminiTimeoutError:
        raise VisionProviderError("timeout") from None
    except ImportError:
        raise VisionProviderError("sdk_unavailable") from None
    except Exception:  # noqa: BLE001 - quota/auth/network/SDK errors
        raise VisionProviderError("unavailable") from None

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise VisionProviderError("empty_response")
    return text, model
