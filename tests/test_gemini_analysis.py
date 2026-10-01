"""Tests for the secondary Gemini material analysis (``app/gemini_analysis.py``).

Every test mocks the Gemini client - the real API is never called.  The stub
client mimics the ``google-genai`` surface the module uses
(``client.models.generate_content(...)`` returning an object with ``.text``).

The evidence fed to Gemini is built with the real
:func:`app.flame_evidence.extract_flame_evidence` from a realistic analyzer
result, so the tests exercise the same evidence path the API layer uses.

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
from app.flame_evidence import extract_flame_evidence, rgb_to_hsv  # noqa: E402
from app.gemini_analysis import gemini_material_analysis  # noqa: E402
from app.material_matching import load_material_database  # noqa: E402
from tests.test_api import SUCCESS_RESULT, StubAnalyzer, _png_bytes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "flame_dataset.json"

#: The exact evidence the real Gemini run uses (task specification): a pale,
#: low-saturation flame whose RGB/LAB could be mistaken for Wood/Paper.
MEAN_COLOR = {"rgb": [194, 174, 143], "lab": [72.0, 2.6, 18.5]}

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
        "evidence_quality": "moderate",
        "reasoning_summary": "The measured yellow-orange flame colour is most compatible with organic, carbon-based materials.",
    }
    payload.update(overrides)
    return payload


def _uncertain_payload(**overrides) -> dict:
    """A payload for evidence that cannot distinguish materials (tests A/C/D)."""
    payload = _valid_payload(
        primary_material=None,
        overall_confidence_level="low",
        uncertain=True,
        evidence_quality="insufficient",
        reasoning_summary=(
            "The measured flame characteristics are insufficient for reliable material "
            "identification: several canonical materials produce visually similar flames, "
            "and flame colour alone cannot distinguish them."
        ),
    )
    # Low, evenly spread confidences: the candidates are plausible, not supported.
    for index, match in enumerate(payload["matches"]):
        match["confidence_percent"] = 25.0 - index * 3.0
        match["reason"] = (
            f"candidate {index + 1} is visually plausible but the evidence does not "
            "distinguish it from the other candidates"
        )
    payload.update(overrides)
    return payload


def _settings(**overrides) -> Settings:
    overrides.setdefault("gemini_enabled", True)
    overrides.setdefault("gemini_api_key", "test-key-do-not-leak")
    return Settings.from_env().with_overrides(**overrides)


def _database():
    return load_material_database(DATASET)


def _sample_result() -> dict:
    """A realistic analyzer result carrying the representative evidence."""
    result = copy.deepcopy(SUCCESS_RESULT)
    result["flame_analysis"]["mean_color"] = {
        "rgb": list(MEAN_COLOR["rgb"]),
        "lab": list(MEAN_COLOR["lab"]),
    }
    return result


def _evidence() -> dict:
    """The evidence the API layer would extract from ``_sample_result``."""
    return extract_flame_evidence(_sample_result())


def _analyze(evidence=None, settings=None, database=None, handler=None, monkeypatch=None):
    """Run the Gemini analysis with a mocked client; returns the payload."""
    settings = settings or _settings()
    database = database if database is not None else _database()
    if handler is not None:
        monkeypatch.setattr(
            "app.gemini_analysis._create_client", lambda _settings: _StubClient(handler)
        )
    return gemini_material_analysis(evidence if evidence is not None else _evidence(), settings, database)


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


def test_missing_evidence_returns_unavailable(monkeypatch):
    payload = _analyze(evidence={}, settings=_settings(), monkeypatch=monkeypatch)
    assert payload["available"] is False
    assert "No flame colour evidence" in payload["error"]


# ---------------------------------------------------------------------------
# 3-4. Success + structured JSON parsing
# ---------------------------------------------------------------------------
def test_successful_gemini_response(monkeypatch):
    payload = _analyze(handler=_ok_handler(_valid_payload()), monkeypatch=monkeypatch)
    assert payload["available"] is True
    assert payload["primary_material"] == "Wood Materials"
    assert payload["overall_confidence_level"] == "medium"
    assert payload["uncertain"] is False
    assert payload["evidence_quality"] == "moderate"
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
    # _ok_handler(None) still returns valid JSON; break the text explicitly.
    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(lambda **kw: SimpleNamespace(text="not json {")),
    )
    payload = gemini_material_analysis(_evidence(), _settings(), _database())
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
# Uncertainty behaviour (task tests A-D): null primary, ambiguous evidence
# ---------------------------------------------------------------------------
def test_uncertain_payload_with_null_primary_is_accepted(monkeypatch):
    """A. Strongly ambiguous evidence -> Gemini may return uncertain/null."""
    payload = _analyze(handler=_ok_handler(_uncertain_payload()), monkeypatch=monkeypatch)
    assert payload["available"] is True
    assert payload["primary_material"] is None
    assert payload["uncertain"] is True
    assert payload["overall_confidence_level"] == "low"
    assert payload["evidence_quality"] == "insufficient"
    assert "insufficient" in payload["reasoning_summary"].lower()
    assert len(payload["matches"]) == 5


def test_multiple_plausible_candidates_with_low_confidence(monkeypatch):
    """C. Multiple plausible materials -> several candidates, low confidence."""
    payload = _analyze(
        handler=_ok_handler(_uncertain_payload(evidence_quality="limited")),
        monkeypatch=monkeypatch,
    )
    assert payload["available"] is True
    assert payload["primary_material"] is None
    assert payload["uncertain"] is True
    assert payload["evidence_quality"] == "limited"
    assert len(payload["matches"]) == 5
    assert all(match["confidence_percent"] <= 30.0 for match in payload["matches"])
    assert {match["material"] for match in payload["matches"]} <= {
        entry.name for entry in _database().entries
    }


def test_null_primary_claiming_certainty_is_rejected(monkeypatch):
    """D. A null primary must come with uncertain=true and level=low."""
    payload = _uncertain_payload(uncertain=False)
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "null primary_material" in result["error"]


def test_null_primary_with_non_low_level_is_rejected(monkeypatch):
    payload = _uncertain_payload(overall_confidence_level="medium")
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "null primary_material" in result["error"]


def test_invalid_evidence_quality_is_rejected(monkeypatch):
    payload = _valid_payload(evidence_quality="excellent")
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "evidence_quality" in result["error"]


def test_missing_evidence_quality_is_rejected(monkeypatch):
    payload = _valid_payload()
    del payload["evidence_quality"]
    result = _analyze(handler=_ok_handler(payload), monkeypatch=monkeypatch)
    assert result["available"] is False
    assert "evidence_quality" in result["error"]


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


# ---------------------------------------------------------------------------
# Regression: the real-API ValidationError (union type in response_schema)
# ---------------------------------------------------------------------------
def _walk_schema(node: Any):
    """Yield every dict/list node of a JSON schema, recursively."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_schema(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_schema(item)


def test_response_schema_uses_no_union_types():
    """A JSON-Schema union ({"type": ["string", "null"]}) is rejected by the
    Gemini SDK before any request is sent - the exact real-API ValidationError.
    """
    from app.gemini_analysis import RESPONSE_SCHEMA

    for node in _walk_schema(RESPONSE_SCHEMA):
        type_value = node.get("type")
        assert not isinstance(type_value, list), (
            f"union type {type_value!r} is not supported by the Gemini Schema; "
            'use a single type with "nullable": true'
        )
        assert "anyOf" not in node
        assert "$ref" not in node
        assert "$defs" not in node
        assert "additionalProperties" not in node


def test_response_schema_nullable_primary_uses_gemini_representation():
    """primary_material stays nullable via the Gemini-supported representation."""
    from app.gemini_analysis import RESPONSE_SCHEMA

    primary = RESPONSE_SCHEMA["properties"]["primary_material"]
    assert primary["type"] == "string"
    assert primary.get("nullable") is True


def test_response_schema_validates_against_the_gemini_sdk():
    """The exact SDK validation that raised the real ValidationError.

    ``types.Schema.model_validate`` is what the google-genai client runs on the
    configured response_schema before sending the request; if this passes, the
    request is not rejected client-side.  Skipped when google-genai is absent.
    """
    google_genai = pytest.importorskip("google.genai")
    from app.gemini_analysis import RESPONSE_SCHEMA

    validated = google_genai.types.Schema.model_validate(RESPONSE_SCHEMA)
    assert validated.properties["primary_material"].nullable is True
    assert validated.properties["primary_material"].type == google_genai.types.Type.STRING


# ---------------------------------------------------------------------------
# Prompt design: evidence-based, uncertainty-aware, never colour-alone
# ---------------------------------------------------------------------------
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
    assert "[194, 174, 143]" in contents
    assert "[72.0, 2.6, 18.5]" in contents
    # The additional extracted evidence is present.
    assert "HSV" in contents
    assert "brightness" in contents
    assert "saturation" in contents
    assert "flame pixels" in contents
    assert "Fire detection confidence" in contents
    assert "Segmentation confidence" in contents
    assert "bounding box" in contents
    assert "clusters" in contents
    # The full canonical vocabulary is supplied.
    database = _database()
    for entry in database.entries:
        assert entry.name in contents
    # Structured output is requested, and no image/base64 data is sent.
    assert "base64" not in contents.lower()
    assert call["config"].response_mime_type == "application/json"


def test_prompt_frames_the_task_as_evidence_sufficiency_not_colour_matching(monkeypatch):
    """B. The prompt must not ask Gemini to identify the material from RGB/LAB."""
    captured: list[dict] = []

    def capturing_handler(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(text=json.dumps(_valid_payload()))

    _analyze(handler=capturing_handler, monkeypatch=monkeypatch)

    call = captured[0]
    instruction = call["config"].system_instruction
    contents = call["contents"]
    # The uncertainty framing is explicit.
    assert "not necessarily the fuel itself" in instruction
    assert "Do not infer material identity from flame color alone" in instruction
    assert "weak evidence" in instruction
    assert "null" in instruction
    assert "insufficient" in instruction
    # The old colour-alone framing is gone.
    assert "Use the supplied RGB and CIELAB measurements as the primary evidence" not in contents
    assert "primary evidence" not in instruction
    # The user message instructs the model to evaluate distinguishability.
    assert "distinguish among the candidate materials" in instruction
    assert "distinguishes among the candidate materials" in contents
    assert "primary_material to null" in contents


# ---------------------------------------------------------------------------
# Evidence extraction (app/flame_evidence.py)
# ---------------------------------------------------------------------------
def test_extract_flame_evidence_forwards_mean_color_and_derives_hsv():
    evidence = _evidence()
    assert evidence["mean_color"] == MEAN_COLOR
    hsv = evidence["hsv"]
    assert hsv["h"] == pytest.approx(36.5, abs=0.1)
    assert hsv["s"] == pytest.approx(26.3, abs=0.1)
    assert hsv["v"] == pytest.approx(76.1, abs=0.1)
    assert evidence["saturation_0_100"] == hsv["s"]
    assert evidence["brightness_0_255"] == pytest.approx(0.299 * 194 + 0.587 * 174 + 0.114 * 143, abs=0.1)


def test_extract_flame_evidence_collects_region_statistics_and_confidences():
    evidence = _evidence()
    region = evidence["flame_region"]
    assert region["flame_pixel_count"] == SUCCESS_RESULT["flame_analysis"]["flame_pixel_count"]
    assert region["samples_used"] == 2000
    assert region["pixels_sampled"] is True
    assert region["mask_area_ratio"] == pytest.approx(0.1302, abs=1e-4)
    assert region["mask_width_px"] == 96
    assert region["mask_height_px"] == 96
    assert evidence["detection_confidence"] == 0.7899
    assert evidence["segmentation_confidence"] == 0.9578


def test_extract_flame_evidence_collects_bounding_box_geometry():
    evidence = _evidence()
    box = evidence["detection_bounding_box"]
    assert box["width_px"] == 60
    assert box["height_px"] == 60
    assert box["aspect_ratio"] == 1.0


def test_extract_flame_evidence_collects_clusters_and_dominant_colors():
    evidence = _evidence()
    distribution = evidence["color_distribution"]
    assert distribution["algorithms"] == SUCCESS_RESULT["flame_analysis"]["algorithms"]
    kmeans = distribution["per_algorithm"]["kmeans"]
    assert kmeans["cluster_count"] == 2
    assert kmeans["representative_rgb"] == [255, 203, 89]
    assert kmeans["dominant_color"]["rgb"] == [255, 214, 105]
    assert kmeans["clusters"]
    assert all("pixel_share" in cluster for cluster in kmeans["clusters"])


def test_extract_flame_evidence_omits_missing_fields():
    result = copy.deepcopy(SUCCESS_RESULT)
    result["segmentation"]["confidence"] = None
    result["flame_analysis"]["kmeans"] = None
    result["flame_analysis"]["gmm"] = None
    result["flame_analysis"]["bayesian_gmm"] = None
    result["flame_analysis"]["dbscan"] = None
    result["flame_analysis"]["agglomerative"] = None
    result["flame_analysis"]["algorithms"] = []
    result["flame_analysis"]["skipped_reason"] = "fewer than 10 flame pixels"

    evidence = extract_flame_evidence(result)

    assert "segmentation_confidence" not in evidence
    assert evidence["color_distribution"]["per_algorithm"] == {}
    assert "fewer than 10 flame pixels" in str(evidence["color_distribution"]["skipped_reason"])


def test_extract_flame_evidence_handles_empty_or_malformed_result():
    evidence = extract_flame_evidence({})
    assert "mean_color" not in evidence
    assert "flame_region" not in evidence
    assert extract_flame_evidence(None) == {}
    assert extract_flame_evidence("not a dict") == {}


def test_rgb_to_hsv_known_values():
    assert rgb_to_hsv([255, 0, 0]) == {"h": 0.0, "s": 100.0, "v": 100.0}
    assert rgb_to_hsv([0, 255, 0])["h"] == pytest.approx(120.0, abs=0.1)
    assert rgb_to_hsv([0, 0, 255])["h"] == pytest.approx(240.0, abs=0.1)
    grey = rgb_to_hsv([128, 128, 128])
    assert grey["s"] == 0.0


def test_rgb_to_hsv_rejects_bad_input():
    assert rgb_to_hsv(None) is None
    assert rgb_to_hsv([1, 2]) is None
    assert rgb_to_hsv([300, 0, 0]) is None
    assert rgb_to_hsv("red") is None


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
    assert ai["evidence_quality"] == "moderate"
    assert "ai_material_ms" in body["timing"]
    assert body["timing"]["ai_material_ms"] >= 0


def test_ai_material_analysis_reports_uncertain_when_evidence_is_ambiguous(monkeypatch):
    """The API surfaces the uncertain/null outcome; the deterministic result is untouched."""
    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(_ok_handler(_uncertain_payload())),
    )
    with _api_client(monkeypatch) as client:
        body = _analyzed_body(client)

    from tests.test_api import SUCCESS_RESULT

    ai = body["ai_material_analysis"]
    assert ai["available"] is True
    assert ai["primary_material"] is None
    assert ai["uncertain"] is True
    assert ai["overall_confidence_level"] == "low"
    assert ai["evidence_quality"] == "insufficient"
    assert len(ai["matches"]) == 5
    # The deterministic analysis is never overwritten or merged.
    assert body["material_analysis"] == SUCCESS_RESULT["material_analysis"]


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
