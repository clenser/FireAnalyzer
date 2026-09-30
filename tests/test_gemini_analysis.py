"""Tests for the secondary Gemini material analysis (``app/gemini_analysis.py``).

Every test mocks the Gemini client - the real API is never called.  The stub
client mimics the ``google-genai`` surface the module uses
(``client.models.generate_content(...)`` returning an object with ``.text``).

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import create_app  # noqa: E402
from app.config import Settings  # noqa: E402
from app.gemini_analysis import gemini_material_analysis  # noqa: E402
from app.material_matching import load_material_database  # noqa: E402
from tests.test_api import SUCCESS_RESULT, StubAnalyzer, _png_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "flame_dataset.json"

#: The exact evidence the real Gemini run uses (task specification).
MEAN_COLOR = {"rgb": [185, 150, 110], "lab": [64.3, 7.9, 26.0]}

CANONICAL_MATERIALS = [
    "Wood Materials",
    "Paper Products(Wood material)",
    "Natural Fibers",
    "Organic Waste",
    "Composite Materials",
]


# ---------------------------------------------------------------------------
# Mock Gemini client
# ---------------------------------------------------------------------------
class _StubModels:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._handler(**kwargs)


class _StubClient:
    def __init__(self, handler) -> None:
        self.models = _StubModels(handler)


def _ok_handler(payload):
    def handler(**kwargs):
        return SimpleNamespace(text=json.dumps(payload))

    return handler


def _valid_payload(materials=None, **overrides) -> dict:
    materials = materials or CANONICAL_MATERIALS
    payload = {
        "primary_material": materials[0],
        "matches": [
            {
                "rank": index + 1,
                "material": material,
                "confidence_percent": 80.0 - index * 7.5,
                "reason": f"documented flame characteristics compatible with candidate {index + 1}",
            }
            for index, material in enumerate(materials)
        ],
        "overall_confidence_level": "medium",
        "uncertain": False,
        "reasoning_summary": "The measured yellow-orange flame colour is most compatible with organic, carbon-based materials.",
    }
    payload.update(overrides)
    return payload


def _settings(**overrides) -> Settings:
    overrides.setdefault("gemini_enabled", True)
    overrides.setdefault("gemini_api_key", "test-key-do-not-leak")
    return Settings.from_env().with_overrides(**overrides)


def _database():
    return load_material_database(DATASET)


def _analyze(mean_color=None, settings=None, database=None, handler=None, monkeypatch=None):
    """Run the Gemini analysis with a mocked client; returns (payload, calls)."""
    settings = settings or _settings()
    database = database if database is not None else _database()
    if handler is not None:
        monkeypatch.setattr(
            "app.gemini_analysis._create_client", lambda _settings: _StubClient(handler)
        )
    payload = gemini_material_analysis(mean_color or MEAN_COLOR, settings, database)
    return payload


# ---------------------------------------------------------------------------
# 1-2. Disabled / missing key
# ---------------------------------------------------------------------------
def test_gemini_disabled_returns_unavailable(monkeypatch):
    payload = _analyze(settings=_settings(gemini_enabled=False), monkeypatch=monkeypatch)
    assert payload["available"] is False
    assert "disabled" in payload["error"]


def test_missing_api_key_returns_unavailable(monkeypatch):
    payload = _analyze(settings=_settings(gemini_api_key=None), monkeypatch=monkeypatch)
    assert payload["available"] is False
    assert "GEMINI_API_KEY" in payload["error"]


# ---------------------------------------------------------------------------
# 3-4. Success + structured JSON parsing
# ---------------------------------------------------------------------------
def test_successful_gemini_response(monkeypatch):
    payload = _analyze(handler=_ok_handler(_valid_payload()), monkeypatch=monkeypatch)
    assert payload["available"] is True
    assert payload["primary_material"] == "Wood Materials"
    assert payload["overall_confidence_level"] == "medium"
    assert payload["uncertain"] is False
    assert payload["reasoning_summary"]
    assert len(payload["matches"]) == 5


def test_structured_json_is_parsed_into_typed_fields(monkeypatch):
    payload = _analyze(handler=_ok_handler(_valid_payload()), monkeypatch=monkeypatch)
    for match in payload["matches"]:
        assert set(match) == {"rank", "material", "confidence_percent", "reason"}
        assert isinstance(match["rank"], int)
        assert isinstance(match["material"], str)
        assert isinstance(match["confidence_percent"], float)
        assert isinstance(match["reason"], str)


def test_malformed_json_is_rejected_cleanly(monkeypatch):
    payload = _analyze(handler=_ok_handler(None), monkeypatch=monkeypatch)
    # _ok_handler(None) still returns valid JSON; break the text explicitly.
    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(lambda **kw: SimpleNamespace(text="not json {")),
    )
    payload = gemini_material_analysis(MEAN_COLOR, _settings(), _database())
    assert payload["available"] is False
    assert "not valid JSON" in payload["error"]


# ---------------------------------------------------------------------------
# 5-8. Validation: unknown material, duplicates, ranks, confidence
# ---------------------------------------------------------------------------
def test_unknown_material_is_rejected(monkeypatch):
    payload = _valid_payload()
    payload["matches"][2]["material"] = "Unobtainium"
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "canonical vocabulary" in result["error"]


def test_unknown_primary_material_is_rejected(monkeypatch):
    payload = _valid_payload()
    payload["primary_material"] = "Wood (invented)"
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False


def test_duplicate_material_is_rejected(monkeypatch):
    payload = _valid_payload()
    payload["matches"][3]["material"] = payload["matches"][0]["material"]
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "duplicate" in result["error"]


def test_invalid_rank_is_rejected(monkeypatch):
    payload = _valid_payload()
    payload["matches"][4]["rank"] = 6
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "ranks" in result["error"]


def test_missing_rank_is_rejected(monkeypatch):
    payload = _valid_payload()
    del payload["matches"][1]["rank"]
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False


def test_invalid_confidence_is_rejected(monkeypatch):
    for bad in (-5.0, 150.0, "high", True):
        payload = _valid_payload()
        payload["matches"][0]["confidence_percent"] = bad
        result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
        assert result["available"] is False, bad
        assert "confidence_percent" in result["error"]


def test_wrong_candidate_count_is_rejected(monkeypatch):
    payload = _valid_payload()
    payload["matches"] = payload["matches"][:4]
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "exactly 5" in result["error"]


def test_rank1_must_be_the_primary(monkeypatch):
    payload = _valid_payload()
    payload["matches"][0]["material"] = CANONICAL_MATERIALS[1]
    payload["matches"][1]["material"] = CANONICAL_MATERIALS[0]
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "rank 1" in result["error"]


# ---------------------------------------------------------------------------
# 9-10. Timeout / API exception
# ---------------------------------------------------------------------------
def test_gemini_timeout_returns_unavailable(monkeypatch):
    def slow_handler(**kwargs):
        time.sleep(0.5)
        return SimpleNamespace(text=json.dumps(_valid_payload()))

    settings = _settings(gemini_timeout_s=0.05)
    result = _analyze(settings=settings, handler=slow_handler, monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "timed out" in result["error"]


def test_gemini_api_exception_returns_unavailable(monkeypatch):
    def exploding_handler(**kwargs):
        raise RuntimeError("quota exhausted")

    result = _analyze(handler=exploding_handler, monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "Gemini API request failed" in result["error"]


# ---------------------------------------------------------------------------
# 14-16. Shape, ordering, vocabulary
# ---------------------------------------------------------------------------
def test_exactly_five_candidates_returned(monkeypatch):
    payload = _analyze(handler=_ok_handler(_valid_payload()), monkeypatch=monkeypatch)
    assert len(payload["matches"]) == 5


def test_candidate_ordering_is_preserved(monkeypatch):
    payload = _analyze(handler=_ok_handler(_valid_payload()), monkeypatch=monkeypatch)
    assert [m["rank"] for m in payload["matches"]] == [1, 2, 3, 4, 5]
    assert [m["material"] for m in payload["matches"]] == CANONICAL_MATERIALS
    assert payload["primary_material"] == payload["matches"][0]["material"]


def test_dataset_vocabulary_is_enforced(monkeypatch):
    database = _database()
    canonical = {entry.name for entry in database.entries}
    assert len(canonical) == 18
    payload = _analyze(handler=_ok_handler(_valid_payload()), monkeypatch=monkeypatch)
    assert {m["material"] for m in payload["matches"]} <= canonical
    assert payload["primary_material"] in canonical


def test_prompt_carries_evidence_and_vocabulary_but_no_image(monkeypatch):
    captured: list[dict] = []

    def capturing_handler(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(text=json.dumps(_valid_payload()))

    _analyze(handler=capturing_handler, monkeypatch=monkeypatch)

    assert len(captured) == 1
    call = captured[0]
    contents = call["contents"]
    # The measured evidence is forwarded verbatim.
    assert "[185, 150, 110]" in contents
    assert "[64.3, 7.9, 26.0]" in contents
    # The full canonical vocabulary is supplied.
    database = _database()
    for entry in database.entries:
        assert entry.name in contents
    # Structured output is requested, and no image/base64 data is sent.
    assert "base64" not in contents.lower()
    assert call["config"].response_mime_type == "application/json"


# ---------------------------------------------------------------------------
# 11-13. API-level: deterministic result survives, structure unchanged
# ---------------------------------------------------------------------------
def _api_client(monkeypatch, **settings_overrides) -> TestClient:
    settings_overrides.setdefault("gemini_enabled", True)
    settings_overrides.setdefault("gemini_api_key", "test-key-do-not-leak")
    settings = Settings.from_env().with_overrides(**settings_overrides)
    # Deep-copy: the attach helper mutates the result dict, and StubAnalyzer
    # would otherwise hand every test the same shared module-level object.
    application = create_app(
        settings=settings,
        analyzer_factory=lambda: StubAnalyzer(result=copy.deepcopy(SUCCESS_RESULT)),
    )
    return TestClient(application)


def _analyzed_body(client: TestClient) -> dict:
    response = client.post("/analyze", files={"image": ("f.png", _png_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    return response.json()


def test_deterministic_analysis_survives_gemini_failure(monkeypatch):
    def exploding_handler(**kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(
        "app.gemini_analysis._create_client", lambda _s: _StubClient(exploding_handler)
    )
    with _api_client(monkeypatch) as client:
        body = _analyzed_body(client)

    assert body["success"] is True
    assert body["ai_material_analysis"]["available"] is False
    assert "Gemini API request failed" in body["ai_material_analysis"]["error"]


def test_original_material_analysis_is_unchanged(monkeypatch):
    from tests.test_api import SUCCESS_RESULT

    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(_ok_handler(_valid_payload())),
    )
    with _api_client(monkeypatch) as client:
        body = _analyzed_body(client)

    assert body["material_analysis"] == SUCCESS_RESULT["material_analysis"]
    assert body["fire_class"] == SUCCESS_RESULT["fire_class"]
    assert body["extinguishing_agents"] == SUCCESS_RESULT["extinguishing_agents"]


def test_ai_material_analysis_present_when_gemini_succeeds(monkeypatch):
    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(_ok_handler(_valid_payload())),
    )
    with _api_client(monkeypatch) as client:
        body = _analyzed_body(client)

    ai = body["ai_material_analysis"]
    assert ai["available"] is True
    assert ai["primary_material"] == "Wood Materials"
    assert len(ai["matches"]) == 5
    assert ai["overall_confidence_level"] == "medium"
    assert "ai_material_ms" in body["timing"]
    assert body["timing"]["ai_material_ms"] >= 0


def test_api_key_is_never_exposed_in_the_response(monkeypatch):
    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(_ok_handler(_valid_payload())),
    )
    with _api_client(monkeypatch) as client:
        response = client.post(
            "/analyze", files={"image": ("f.png", _png_bytes(), "image/png")}
        )

    assert "test-key-do-not-leak" not in response.text


def test_gemini_disabled_leaves_timing_untouched(monkeypatch):
    with _api_client(monkeypatch, gemini_enabled=False) as client:
        body = _analyzed_body(client)

    assert body["ai_material_analysis"]["available"] is False
    assert "ai_material_ms" not in body["timing"]
