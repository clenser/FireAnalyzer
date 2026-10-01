"""Tests for ``POST /material-identification`` (deterministic video material ID).

The endpoint is a thin transport over the *existing* deterministic matcher, so
these tests assert two different things:

* the contract - shape, validation, error format, canonical names, alternatives,
  fire class and extinguishing agents;
* the reuse itself - that the same matcher, the same canonical dataset and the
  same fire-class mapping the image workflow uses are what produced the answer,
  and that no detection or segmentation ran to get it.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import video_material  # noqa: E402
from app.api import create_app  # noqa: E402
from app.color_analysis import RepresentativeColor  # noqa: E402
from app.config import Settings  # noqa: E402
from app.errors import MissingDatabaseError  # noqa: E402
from app.fire_classes import MATERIAL_FIRE_CLASS, MAPPING_SOURCE  # noqa: E402
from app.material_matching import load_material_database, match_material  # noqa: E402
from app.video_material import identify_material  # noqa: E402
from tests.test_api import StubAnalyzer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "flame_dataset.json"

#: An aggregated flame colour of the kind a client computes across video frames:
#: a deep orange flame (RGB) with its CIELAB equivalent.
RGB = [166.0, 80.0, 27.0]
LAB = [43.8, 31.4, 43.0]
BODY = {"rgb": RGB, "lab": LAB}

CANONICAL_MATERIALS = tuple(entry.name for entry in load_material_database(DATASET).entries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class DatabaseStub(StubAnalyzer):
    """Stub analyzer that exposes the canonical database (or deliberately not)."""

    def __init__(self, *, database=..., result=None) -> None:
        super().__init__(result=result)
        self._database = load_material_database(DATASET) if database is ... else database

    @property
    def database(self):
        if self._database is None:
            raise MissingDatabaseError("The material database has not been loaded yet.")
        return self._database


def _client(stub: StubAnalyzer | None, **overrides) -> TestClient:
    overrides.setdefault("gemini_enabled", False)
    settings = Settings.from_env().with_overrides(**overrides)
    factory = (lambda: stub) if stub is not None else (lambda: None)
    return TestClient(create_app(settings=settings, analyzer_factory=factory))


def _identify(stub=None, body=None, **overrides) -> dict:
    with _client(stub if stub is not None else DatabaseStub(), **overrides) as client:
        response = client.post("/material-identification", json=BODY if body is None else body)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Valid requests
# ---------------------------------------------------------------------------
def test_valid_rgb_and_lab_returns_a_deterministic_identification():
    body = _identify()

    assert body["success"] is True
    material = body["material_analysis"]
    assert material["primary_material"] in CANONICAL_MATERIALS
    assert 0.0 <= material["similarity"] <= 1.0
    assert material["score_basis"] == "mean LAB distance to flame_dataset.json reference colours"
    # The whole body must survive json.dumps untouched (no numpy leaking out).
    assert json.loads(json.dumps(body)) == body


def test_the_material_analysis_is_the_same_structure_the_image_workflow_returns():
    """Identical keys and types to ``/analyze``'s ``material_analysis``."""
    body = _identify()
    material = body["material_analysis"]

    assert set(material) == {
        "primary_material",
        "similarity",
        "alternatives",
        "database_notes",
        "score_basis",
    }
    assert isinstance(material["similarity"], float)
    for alternative in material["alternatives"]:
        assert set(alternative) == {"material", "similarity"}
        assert alternative["material"] in CANONICAL_MATERIALS
        assert isinstance(alternative["similarity"], float)


def test_alternative_matches_are_returned_and_bounded_by_the_setting():
    default_alternatives = _identify()["material_analysis"]["alternatives"]
    assert len(default_alternatives) == Settings.from_env().max_alternatives

    two = _identify(max_alternatives=2)["material_analysis"]["alternatives"]
    assert len(two) == 2
    assert [entry["material"] for entry in two] == [
        entry["material"] for entry in default_alternatives[:2]
    ]


def test_alternatives_are_ordered_by_similarity_and_exclude_the_primary():
    material = _identify()["material_analysis"]
    scores = [entry["similarity"] for entry in material["alternatives"]]
    assert scores == sorted(scores, reverse=True)
    assert material["primary_material"] not in {
        entry["material"] for entry in material["alternatives"]
    }


def test_canonical_material_names_come_from_the_dataset():
    """Only names that exist in ``flame_dataset.json`` are ever returned."""
    body = _identify()
    assert body["primary_material"] in CANONICAL_MATERIALS
    for alternative in body["alternatives"]:
        assert alternative["material"] in CANONICAL_MATERIALS


def test_top_level_fields_mirror_the_nested_material_analysis():
    body = _identify()
    material = body["material_analysis"]
    assert body["primary_material"] == material["primary_material"]
    assert body["similarity"] == material["similarity"]
    assert body["alternatives"] == material["alternatives"]


def test_integer_and_float_values_are_both_accepted():
    integral = _identify(body={"rgb": [255, 255, 0], "lab": [97, -20, 90]})
    fractional = _identify(body={"rgb": [255.0, 255.0, 0.0], "lab": [97.0, -20.0, 90.0]})
    assert integral == fractional


# ---------------------------------------------------------------------------
# The deterministic matcher is the existing one
# ---------------------------------------------------------------------------
def test_the_existing_matcher_produces_the_answer(monkeypatch):
    """A spy around the real matcher: one call, with the supplied colour."""
    calls: list[tuple] = []
    real_match_material = match_material

    def spy(colors, database, settings):
        calls.append((colors, database, settings))
        return real_match_material(colors, database, settings)

    monkeypatch.setattr(video_material, "match_material", spy)
    stub = DatabaseStub()
    body = _identify(stub)

    assert len(calls) == 1, "the deterministic matcher must run exactly once"
    colors, database, settings = calls[0]
    assert len(colors) == 1
    assert database is stub.database
    assert settings.dataset_path == Settings.from_env().dataset_path
    # The colour handed to the matcher is the client's colour, unaltered.
    assert colors[0].lab == pytest.approx(np.asarray(LAB, dtype=np.float64))
    assert list(colors[0].rgb) == [166, 80, 27]
    # And the answer is exactly what that call returns.
    assert body["material_analysis"]["primary_material"] in CANONICAL_MATERIALS


def test_the_result_is_identical_to_running_the_matcher_directly():
    """Endpoint output == :func:`match_material` + :func:`app.fire_classes.classify`."""
    database = load_material_database(DATASET)
    settings = Settings.from_env()
    color = RepresentativeColor(
        method=video_material.AGGREGATED_COLOR_METHOD,
        rgb=np.asarray([166, 80, 27], dtype=np.uint8),
        lab=np.asarray(LAB, dtype=np.float64),
    )
    analysis, suppression = match_material([color], database, settings)

    body = _identify()
    assert body["material_analysis"]["primary_material"] == analysis.primary_material
    assert body["material_analysis"]["similarity"] == analysis.similarity
    assert body["material_analysis"]["alternatives"] == [
        {"material": score.material, "similarity": score.similarity}
        for score in analysis.alternatives
    ]
    assert body["material_analysis"]["database_notes"] == analysis.database_notes
    assert body["suppression_information"]["methods"] == suppression.methods


def test_identify_material_is_usable_as_a_library_call():
    database = load_material_database(DATASET)
    body = identify_material(RGB, LAB, database, Settings.from_env())
    assert body["material_analysis"]["primary_material"] in CANONICAL_MATERIALS


# ---------------------------------------------------------------------------
# No YOLO, no segmentation, no LLM
# ---------------------------------------------------------------------------
def test_no_detection_or_segmentation_runs():
    """The analyzer is never asked to analyse anything."""
    stub = DatabaseStub()
    with _client(stub) as client:
        response = client.post("/material-identification", json=BODY)
    assert response.status_code == 200
    assert stub.calls == [], "the deterministic endpoint must not run inference"


def test_it_works_without_models_loaded_or_attached():
    """No YOLO detection, no segmentation - not even a loaded analyzer."""
    unloaded = DatabaseStub()
    unloaded._models_loaded = False
    unloaded._segmentation_available = False

    for stub in (unloaded, None):
        body = _identify(stub)
        assert body["success"] is True
        assert body["material_analysis"]["primary_material"] in CANONICAL_MATERIALS


def test_it_falls_back_to_loading_the_dataset_when_the_analyzer_has_none():
    stub = DatabaseStub(database=None)
    body = _identify(stub)
    assert body["material_analysis"]["primary_material"] in CANONICAL_MATERIALS
    assert stub.calls == []


def test_a_missing_database_is_reported_as_service_unavailable(tmp_path):
    with _client(DatabaseStub(database=None), dataset_path=tmp_path / "absent.json") as client:
        response = client.post("/material-identification", json=BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MISSING_DATABASE"


def test_gemini_is_not_called_by_this_endpoint(monkeypatch):
    """A deterministic endpoint must not reach the AI path at all."""

    def explode(*args, **kwargs):
        raise AssertionError("Gemini must not be called for deterministic identification")

    monkeypatch.setattr("app.video_material.gemini_video_material_analysis", explode)
    monkeypatch.setattr("app.gemini_analysis._create_client", explode)
    body = _identify(**{"gemini_enabled": True, "gemini_api_key": "test-key-do-not-leak"})
    assert body["primary_material"] in CANONICAL_MATERIALS


def test_the_api_key_never_appears_in_the_response():
    body = _identify(**{"gemini_enabled": True, "gemini_api_key": "test-key-do-not-leak"})
    assert "test-key-do-not-leak" not in json.dumps(body)


# ---------------------------------------------------------------------------
# Fire class and extinguishing agents (existing backend mapping)
# ---------------------------------------------------------------------------
def test_fire_class_comes_from_the_existing_material_mapping():
    body = _identify()
    material = body["material_analysis"]["primary_material"]

    assert body["fire_class"]["class"] == MATERIAL_FIRE_CLASS[material]
    assert body["fire_class"]["material"] == material
    assert body["fire_class"]["mapping_source"] == MAPPING_SOURCE
    # `class_` never leaks the Python field name.
    assert "class_" not in body["fire_class"]
    assert body["fire_class"]["confidence"] == body["material_analysis"]["similarity"]


def test_extinguishing_agents_are_copied_from_the_dataset():
    database = load_material_database(DATASET)
    body = _identify()
    material = body["material_analysis"]["primary_material"]

    assert [agent["name"] for agent in body["extinguishing_agents"]] == list(
        database.get(material).extinguishers
    )
    assert body["suppression_information"]["methods"] == [
        agent["name"] for agent in body["extinguishing_agents"]
    ]
    assert body["suppression_information"]["source"] == "flame_dataset.json"
    for agent in body["extinguishing_agents"]:
        assert agent["compound"] is None
        assert agent["fire_class"] == body["fire_class"]["class"]


# ---------------------------------------------------------------------------
# Validation - the project's normal error format
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body",
    [
        {"rgb": [166, 80], "lab": LAB},
        {"rgb": [], "lab": LAB},
        {"rgb": [166, 80, 27, 12], "lab": LAB},
        {"rgb": "166,80,27", "lab": LAB},
        {"rgb": 166, "lab": LAB},
        {"rgb": None, "lab": LAB},
    ],
    ids=["two-rgb", "empty-rgb", "four-rgb", "not-a-list", "scalar-rgb", "null-rgb"],
)
def test_invalid_rgb_length_is_rejected(body):
    with _client(DatabaseStub()) as client:
        response = client.post("/material-identification", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "rgb" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "body",
    [
        {"rgb": RGB, "lab": [43.8, 31.4]},
        {"rgb": RGB, "lab": []},
        {"rgb": RGB, "lab": [43.8, 31.4, 43.0, 12.0]},
        {"rgb": RGB, "lab": None},
    ],
    ids=["two-lab", "empty-lab", "four-lab", "null-lab"],
)
def test_invalid_lab_length_is_rejected(body):
    with _client(DatabaseStub()) as client:
        response = client.post("/material-identification", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "lab" in response.json()["error"]["message"]


def test_a_missing_colour_is_rejected():
    for body in ({"rgb": RGB}, {"lab": LAB}, {}):
        with _client(DatabaseStub()) as client:
            response = client.post("/material-identification", json=body)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    "body",
    [
        {"rgb": [300, 0, 0], "lab": LAB},
        {"rgb": [0, -5, 0], "lab": LAB},
        {"rgb": ["red", 0, 0], "lab": LAB},
        {"rgb": [True, False, 0], "lab": LAB},
        {"rgb": [166, None, 27], "lab": LAB},
        {"rgb": [166, 80, 27], "lab": [120, 31.4, 43.0]},
        {"rgb": [166, 80, 27], "lab": [-1, 31.4, 43.0]},
        {"rgb": [166, 80, 27], "lab": [43.8, 200, 43.0]},
        {"rgb": [166, 80, 27], "lab": [43.8, 31.4, -500]},
        {"rgb": [166, 80, 27], "lab": ["lightness", 31.4, 43.0]},
    ],
    ids=[
        "rgb-above-255",
        "rgb-negative",
        "rgb-non-numeric",
        "rgb-boolean",
        "rgb-null-channel",
        "lab-lightness-above-100",
        "lab-lightness-negative",
        "lab-a-above-127",
        "lab-b-below-128",
        "lab-non-numeric",
    ],
)
def test_out_of_range_and_non_numeric_values_are_rejected(body):
    with _client(DatabaseStub()) as client:
        response = client.post("/material-identification", json=body)
    assert response.status_code == 422, body
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_fields_are_rejected():
    """No room for an image, a base64 blob or any other undocumented field."""
    with _client(DatabaseStub()) as client:
        response = client.post(
            "/material-identification",
            json={**BODY, "image_base64": "iVBORw0KGgoAAAANSUhEUg"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "image_base64" in response.json()["error"]["message"]


def test_a_non_json_body_is_rejected():
    with _client(DatabaseStub()) as client:
        response = client.post(
            "/material-identification",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_validation_failures_use_the_project_error_shape():
    with _client(DatabaseStub()) as client:
        response = client.post("/material-identification", json={"rgb": [1, 2], "lab": LAB})
    body = response.json()
    assert set(body) == {"success", "error"}
    assert body["success"] is False
    assert set(body["error"]) == {"code", "message"}
    assert "Traceback" not in response.text


def test_a_malformed_request_never_reaches_the_matcher(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the matcher must not run on an invalid request")

    monkeypatch.setattr(video_material, "match_material", explode)
    with _client(DatabaseStub()) as client:
        response = client.post("/material-identification", json={"rgb": [1, 2], "lab": LAB})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------
def test_openapi_documents_the_endpoint():
    with _client(DatabaseStub()) as client:
        schema = client.get("/openapi.json").json()

    assert "/material-identification" in schema["paths"]
    operation = schema["paths"]["/material-identification"]["post"]
    assert operation["summary"]
    assert "deterministic" in operation["description"].lower()
    assert set(operation["responses"]) >= {"200", "422", "503"}

    body = schema["components"]["schemas"]["MaterialIdentificationRequest"]
    assert set(body["required"]) == {"rgb", "lab"}
    assert body["additionalProperties"] is False


def test_the_image_endpoint_is_still_documented_alongside_it():
    with _client(DatabaseStub()) as client:
        schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) >= {
        "/",
        "/health",
        "/activity",
        "/analyze",
        "/material-identification",
        "/video-material-analysis",
    }