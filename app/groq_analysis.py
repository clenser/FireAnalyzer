"""Groq vision call (primary vision evidence provider).

Standard library only (``urllib``) against Groq's OpenAI-compatible chat
completions endpoint.  Configured by ``GROQ_API_KEY`` and ``GROQ_MODEL``; both
are required.  Every failure is raised as
:class:`~app.vision_providers.VisionProviderError` with a short reason code so
the caller can fall back to Gemini.  The key is only ever placed in the
``Authorization`` header - never logged, never returned.
"""

from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.request
from typing import Any

from .config import Settings
from .vision_providers import VisionProviderError

__all__ = ["GROQ_CHAT_COMPLETIONS_URL", "call_groq_vision"]

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
_MAX_RESPONSE_BYTES = 1024 * 1024


def call_groq_vision(image_jpeg: bytes, prompt: str, settings: Settings) -> tuple[str, str]:
    """Send one image + prompt to Groq; return ``(response_text, model)``."""
    api_key = (settings.groq_api_key or "").strip()
    model = (settings.groq_model or "").strip()
    if not api_key:
        raise VisionProviderError("missing_api_key")
    if not model:
        raise VisionProviderError("missing_model")

    data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": 800,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        GROQ_CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "FlameAnalyzer/vision",
        },
        method="POST",
    )
    timeout = max(1.0, float(settings.groq_timeout_s))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as exc:
        raise VisionProviderError(f"http_{exc.code}") from None
    except (socket.timeout, TimeoutError):
        raise VisionProviderError("timeout") from None
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
            raise VisionProviderError("timeout") from None
        raise VisionProviderError("unavailable") from None
    except OSError:
        raise VisionProviderError("unavailable") from None

    try:
        payload = json.loads(raw.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError):
        raise VisionProviderError("malformed_response") from None
    if not isinstance(content, str) or not content.strip():
        raise VisionProviderError("empty_response")
    return content, model
