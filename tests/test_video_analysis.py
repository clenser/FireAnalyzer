"""Tests for ``POST /video-material-analysis`` (ONE consolidated Gemini analysis).

The whole point of this endpoint is that a video produces **one** Gemini request
and **one** consolidated answer, so most of these tests count Gemini calls and
inspect what that single prompt actually contained:

* exactly one ``generate_content`` call per video, whatever the frame count;
* every analysed frame's evidence present in that one prompt;
* the mandated consolidation instruction present;
* no image, mask or base64 anywhere in the request;
* the image workflow's behaviour (and its single call) untouched.

Gemini is always mocked - the real API is never called.

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

from app import video_material  # noqa: E402
from app.api import create_app  # noqa: E402
from app.config import Settings  # noqa: E402
from app.gemini_analysis import (  # noqa: E402
    SYSTEM_INSTRUCTION,
    VIDEO_CONSOLIDATION_INSTRUCTION,
    VIDEO_SYSTEM_INSTRUCTION,
)
from app.material_matching import load_material_database  # noqa: E402
from app.video_material import (  # noqa: E402
    MAX_VIDEO_FRAMES,
    VIDEO_DISPLAY_THRESHOLD_PERCENT,
    FrameEvidence,
    build_video_evidence,
    frame_evidence,
    video_material_analysis,
)
from tests.test_api import SUCCESS_RESULT, StubAnalyzer, _png_bytes  # noqa: E402
from tests.test_gemini_analysis import (  # noqa: E402
    CANONICAL_MATERIALS,
    _StubClient,
    _StubModels,
    _ok_handler,
    _uncertain_payload,
    _valid_payload,
)
from tests.test_material_identification import DatabaseStub  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "flame_dataset.json"

#: Five sampled frames of one fire event, exactly as a client collects them from
#: ``/analyze`` responses.  Each carries a different measured colour.
FRAMES = [
    {
        "frame_index": 0,
        "timestamp_seconds": 0.0,
        "rgb": [255, 180, 60],
        "lab": [70.0, 25.0, 60.0],
        "detection_confidence": 0.91,
        "flame_area_ratio": 0.18,
        "segmentation_confidence": 0.95,
    },
    {
        "frame_index": 1,
        "timestamp_seconds": 4.2,
        "rgb": [166, 80, 27],
        "lab": [43.8, 31.4, 43.0],
        "detection_confidence": 0.78,
        "flame_area_ratio": 0.14,
        "segmentation_confidence": 0.91,
    },
    {
        "frame_index": 2,
        "timestamp_seconds": 8.9,
        "rgb": [255, 214, 105],
        "lab": [86.2, 2.1, 71.4],
        "detection_confidence": 0.64,
        "flame_area_ratio": 0.09,
        "segmentation_confidence": 0.88,
    },
    {
        "frame_index": 3,
        "timestamp_seconds": 13.6,
        "rgb": [200, 95, 30],
        "lab": [48.0, 33.0, 52.0],
        "detection_confidence": 0.83,
        "flame_area_ratio": 0.21,
        "segmentation_confidence": 0.93,
    },
    {
        "frame_index": 4,
        "timestamp_seconds": 18.1,
        "rgb": [255, 160, 40],
        "lab": [63.4, 28.7, 66.2],
        "detection_confidence": 0.72,
        "flame_area_ratio": 0.12,
        "segmentation_confidence": 0.9,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _RecordingModels(_StubModels):
    """Records every Gemini request so the tests can count them."""

    @property
    def call_count(self) -> int:
        return len(self.calls)


class _RecordingClient(_StubClient):
    """A Gemini client whose ``models`` attribute is the shared recorder."""

    def __init__(self, models) -> None:
        self.models = models


def _install(monkeypatch, handler) -> list[_RecordingModels]:
    """Point ``_create_client`` at a recording stub; returns the recorder.

    ``_create_client`` is called once per analysis, so every client shares one
    recorder: the tests can then assert on the *sequence* of requests as well as
    their count.
    """
    models = _RecordingModels(handler)
    monkeypatch.setattr(
        "app.gemini_analysis._create_client", lambda _settings: _RecordingClient(models)
    )
    return [models]


def _total_calls(recorders: list[_RecordingModels]) -> int:
    return sum(recorder.call_count for recorder in recorders)


def _settings(**overrides) -> Settings:
    overrides.setdefault("gemini_enabled", True)
    overrides.setdefault("gemini_api_key", "test-key-do-not-leak")
    return Settings.from_env().with_overrides(**overrides)


def _client(monkeypatch, handler=None, **overrides) -> TestClient:
    """A test client with Gemini enabled and a stubbed analyzer.

    Passing ``handler`` installs a recording Gemini stub; tests that need to count
    calls install it themselves so they keep the recorder.
    """
    if handler is not None:
        _install(monkeypatch, handler)
    application = create_app(
        settings=_settings(**overrides),
        analyzer_factory=lambda: StubAnalyzer(result=copy.deepcopy(SUCCESS_RESULT)),
    )
    return TestClient(application)


def _post(client: TestClient, frames=None) -> dict:
    response = client.post(
        "/video-material-analysis",
        json={"frames": FRAMES if frames is None else frames},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _database():
    return load_material_database(DATASET)


# ---------------------------------------------------------------------------
# The consolidated happy path
# ---------------------------------------------------------------------------
def test_valid_frame_evidence_returns_one_consolidated_result(monkeypatch):
    with _client(monkeypatch, _ok_handler(_valid_payload())) as client:
        body = _post(client)

    assert body["available"] is True
    assert body["primary_material"] == CANONICAL_MATERIALS[0]
    assert len(body["matches"]) == 5
    assert body["overall_confidence_level"] == "medium"
    assert body["uncertain"] is False
    assert body["evidence_quality"] == "moderate"
    assert body["reasoning_summary"]
    assert body["frames_supplied"] == len(FRAMES)
    assert body["frames_analyzed"] == len(FRAMES)
    assert json.loads(json.dumps(body)) == body


def test_exactly_one_gemini_call_for_the_whole_collection(monkeypatch):
    """The core guarantee: N frames in, one request out - never one per frame."""
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        _post(client)

    assert recorders, "Gemini was never called"
    assert _total_calls(recorders) == 1, "expected exactly 1 Gemini call for 5 frames"


@pytest.mark.parametrize("count", [1, 2, 5, 20])
def test_one_gemini_call_regardless_of_the_frame_count(monkeypatch, count):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    frames = [dict(FRAMES[index % len(FRAMES)], frame_index=index) for index in range(count)]
    with _client(monkeypatch) as client:
        body = _post(client, frames)

    assert _total_calls(recorders) == 1, count
    assert body["frames_analyzed"] == count


def test_gemini_receives_every_analysed_frame(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        _post(client)

    contents = recorders[0].calls[0]["contents"]
    for frame in FRAMES:
        assert f"frame {frame['frame_index']}" in contents
        assert str(frame["rgb"]) in contents
        assert str(frame["lab"]) in contents
    assert contents.count("- frame ") == len(FRAMES)


def test_gemini_receives_the_per_frame_measurements_and_derived_values(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        _post(client)

    contents = recorders[0].calls[0]["contents"]
    assert "HSV" in contents
    assert "brightness" in contents
    assert "saturation" in contents
    assert "detection confidence" in contents
    assert "segmentation confidence" in contents
    assert "flame area" in contents
    assert "t=4.2s" in contents
    # Cross-frame statistics: what lets the model judge consistency.
    assert "Observations analysed: 5 frames of one fire event" in contents
    assert "Mean LAB across the frames" in contents
    assert "Per-channel LAB standard deviation across the frames" in contents
    assert "Mean flame area across the frames" in contents
    assert "Observed time span: 18.1s" in contents


def test_the_prompt_demands_one_consolidated_assessment(monkeypatch):
    """The mandated instruction must be in the request, verbatim."""
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        _post(client)

    call = recorders[0].calls[0]
    assert VIDEO_CONSOLIDATION_INSTRUCTION in call["config"].system_instruction
    assert VIDEO_CONSOLIDATION_INSTRUCTION in call["contents"]
    assert "multiple samples of the same fire event" in call["contents"]
    assert (
        "Do not independently classify each frame and average confidence percentages"
        in call["contents"]
    )
    assert (
        "Confidence must reflect the consistency, quality, and amount of evidence"
        in call["contents"]
    )
    # ... and the model is told not to fall back on a single frame's answer.
    assert "Never return one assessment per observation" in call["config"].system_instruction


def test_the_response_carries_no_per_frame_results(monkeypatch):
    with _client(monkeypatch, _ok_handler(_valid_payload())) as client:
        body = _post(client)

    # One primary material, one ranked list, one confidence level - for the video.
    assert isinstance(body["primary_material"], str)
    assert len(body["matches"]) == 5
    assert {match["rank"] for match in body["matches"]} == {1, 2, 3, 4, 5}
    assert "frames" not in body
    assert "per_frame" not in json.dumps(body)


def test_the_canonical_vocabulary_is_offered_to_gemini(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        _post(client)

    contents = recorders[0].calls[0]["contents"]
    assert "Canonical material vocabulary (18 categories)" in contents
    for material in CANONICAL_MATERIALS:
        assert material in contents


# ---------------------------------------------------------------------------
# Structured output / canonical material / confidence validation
# ---------------------------------------------------------------------------
def test_structured_json_is_requested_and_the_payload_is_typed(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        body = _post(client)

    call = recorders[0].calls[0]
    assert call["config"].response_mime_type == "application/json"
    assert call["model"] == Settings.from_env().gemini_model
    assert call["config"].response_schema == __import__(
        "app.gemini_analysis", fromlist=["RESPONSE_SCHEMA"]
    ).RESPONSE_SCHEMA
    for match in body["matches"]:
        assert set(match) == {"rank", "material", "confidence_percent", "reason"}
        assert isinstance(match["rank"], int)
        assert isinstance(match["material"], str)
        assert isinstance(match["confidence_percent"], float)
        assert isinstance(match["reason"], str) and match["reason"]


def test_returned_materials_are_canonical_dataset_names(monkeypatch):
    with _client(monkeypatch, _ok_handler(_valid_payload())) as client:
        body = _post(client)

    canonical = {entry.name for entry in _database().entries}
    assert body["primary_material"] in canonical
    assert {match["material"] for match in body["matches"]} <= canonical


@pytest.mark.parametrize("material", ["Unobtainium", "wood", "Wood Material"])
def test_a_non_canonical_material_is_rejected(monkeypatch, material):
    payload = _valid_payload()
    payload["matches"][1]["material"] = material
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)

    assert body["available"] is False
    assert "canonical vocabulary" in body["error"]


def test_an_unknown_primary_material_is_rejected(monkeypatch):
    payload = _valid_payload()
    payload["primary_material"] = "Volcanic Ash"
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False


def test_duplicate_materials_are_rejected(monkeypatch):
    payload = _valid_payload()
    payload["matches"][3]["material"] = payload["matches"][0]["material"]
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False
    assert "duplicate" in body["error"]


@pytest.mark.parametrize("bad", [-5.0, 150.0, "high", None])
def test_an_out_of_range_confidence_is_rejected(monkeypatch, bad):
    payload = _valid_payload()
    payload["matches"][0]["confidence_percent"] = bad
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False
    assert "confidence_percent" in body["error"]


def test_a_wrong_candidate_count_is_rejected(monkeypatch):
    payload = _valid_payload()
    payload["matches"] = payload["matches"][:3]
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False
    assert "exactly 5" in body["error"]


def test_broken_ranks_are_rejected(monkeypatch):
    payload = _valid_payload()
    payload["matches"][0]["rank"] = 3
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False
    assert "ranks" in body["error"]


def test_rank_one_must_be_the_primary_material(monkeypatch):
    payload = _valid_payload()
    payload["matches"][0]["material"] = CANONICAL_MATERIALS[1]
    payload["matches"][1]["material"] = CANONICAL_MATERIALS[0]
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False
    assert "rank 1" in body["error"]


def test_an_uncertain_consolidated_result_is_accepted(monkeypatch):
    """The video path keeps the same uncertainty contract as the image path."""
    with _client(monkeypatch, _ok_handler(_uncertain_payload())) as client:
        body = _post(client)

    assert body["available"] is True
    assert body["primary_material"] is None
    assert body["uncertain"] is True
    assert body["overall_confidence_level"] == "low"
    assert body["evidence_quality"] == "insufficient"
    assert len(body["matches"]) == 5


def test_a_null_primary_claiming_certainty_is_rejected(monkeypatch):
    payload = _uncertain_payload(uncertain=False)
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False
    assert "null primary_material" in body["error"]


def test_an_invalid_evidence_quality_is_rejected(monkeypatch):
    payload = _valid_payload(evidence_quality="excellent")
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["available"] is False
    assert "evidence_quality" in body["error"]


@pytest.mark.parametrize(
    "text",
    ["not json {", "", "[]", "null", '{"primary_material": "Wood Materials"}'],
    ids=["garbage", "empty", "list", "null", "incomplete"],
)
def test_a_malformed_gemini_response_is_handled(monkeypatch, text):
    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(lambda **kwargs: SimpleNamespace(text=text)),
    )
    with _client(monkeypatch) as client:
        response = client.post("/video-material-analysis", json={"frames": FRAMES})

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["error"]
    # The frame counts still describe what was submitted.
    assert body["frames_analyzed"] == len(FRAMES)


def test_gemini_returning_several_results_instead_of_one_is_rejected(monkeypatch):
    """A list of per-frame assessments is not a consolidated answer."""
    monkeypatch.setattr(
        "app.gemini_analysis._create_client",
        lambda _s: _StubClient(
            lambda **kwargs: SimpleNamespace(
                text=json.dumps([_valid_payload(), _valid_payload()])
            )
        ),
    )
    with _client(monkeypatch) as client:
        body = _post(client)
    assert body["available"] is False
    assert "not a JSON object" in body["error"]


# ---------------------------------------------------------------------------
# Confidence rule: never inflated, never hidden
# ---------------------------------------------------------------------------
def test_a_low_confidence_result_is_still_returned_unchanged(monkeypatch):
    below = 38.5
    payload = _valid_payload()
    payload["matches"][0]["confidence_percent"] = below
    payload["overall_confidence_level"] = "low"
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)

    assert body["available"] is True
    assert body["matches"][0]["confidence_percent"] == below
    assert below < body["display_threshold_percent"]


def test_the_display_threshold_is_reported_and_not_applied_to_the_result(monkeypatch):
    payload = _valid_payload()
    payload["matches"][0]["confidence_percent"] = 12.0
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)

    assert body["display_threshold_percent"] == VIDEO_DISPLAY_THRESHOLD_PERCENT == 45.0
    assert body["matches"][0]["confidence_percent"] == 12.0
    assert body["primary_material"] is not None  # not suppressed by the backend


def test_a_high_confidence_result_is_not_altered(monkeypatch):
    payload = _valid_payload()
    payload["matches"][0]["confidence_percent"] = 88.0
    with _client(monkeypatch, _ok_handler(payload)) as client:
        body = _post(client)
    assert body["matches"][0]["confidence_percent"] == 88.0


# ---------------------------------------------------------------------------
# Gemini unavailable / broken
# ---------------------------------------------------------------------------
def test_gemini_disabled_is_reported_as_unavailable(monkeypatch):
    def explode(_settings):  # pragma: no cover - must never run
        raise AssertionError("Gemini must not be called while disabled")

    monkeypatch.setattr("app.gemini_analysis._create_client", explode)
    with _client(monkeypatch, gemini_enabled=False) as client:
        body = _post(client)

    assert body["available"] is False
    assert "disabled" in body["error"]
    assert body["frames_analyzed"] == len(FRAMES)


def test_a_missing_api_key_is_reported_as_unavailable(monkeypatch):
    with _client(monkeypatch, gemini_api_key=None) as client:
        body = _post(client)
    assert body["available"] is False
    assert "GEMINI_API_KEY" in body["error"]


def test_a_gemini_api_failure_is_reported_as_unavailable(monkeypatch):
    def exploding(**kwargs):
        raise RuntimeError("quota exhausted")

    with _client(monkeypatch, exploding) as client:
        body = _post(client)
    assert body["available"] is False
    assert "Gemini API request failed" in body["error"]


def test_a_slow_gemini_call_times_out(monkeypatch):
    def slow(**kwargs):
        time.sleep(0.5)
        return SimpleNamespace(text=json.dumps(_valid_payload()))

    with _client(monkeypatch, slow, gemini_timeout_s=0.05) as client:
        body = _post(client)
    assert body["available"] is False
    assert "timed out" in body["error"]


def test_a_missing_database_is_reported_as_unavailable(tmp_path, monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    application = create_app(
        settings=_settings(dataset_path=tmp_path / "absent.json"),
        analyzer_factory=lambda: DatabaseStub(database=None),
    )
    with TestClient(application) as client:
        body = _post(client)

    assert body["available"] is False
    assert "material database" in body["error"]
    assert _total_calls(recorders) == 0


def test_an_unexpected_failure_never_fails_the_request(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("internal detail / secret path C:\\models")

    monkeypatch.setattr(video_material, "gemini_video_material_analysis", explode)
    with _client(monkeypatch) as client:
        body = _post(client)

    assert body["available"] is False
    assert "secret" not in json.dumps(body)
    assert body["frames_analyzed"] == len(FRAMES)


# ---------------------------------------------------------------------------
# Malformed / missing evidence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "frames",
    [
        [],
        [{"frame_index": 0, "rgb": [1, 2, 3]}],
        [{"frame_index": 0, "lab": [1, 2, 3]}],
        [{"rgb": [1, 2, 3], "lab": [1, 2, 3]}],
        [{"frame_index": 0, "rgb": [1, 2], "lab": [1, 2, 3]}],
        [{"frame_index": 0, "rgb": [1, 2, 3], "lab": [1, 2]}],
        [{"frame_index": 0, "rgb": [1, 2, 3, 4], "lab": [1, 2, 3]}],
        [{"frame_index": 0, "rgb": ["a", "b", "c"], "lab": [1, 2, 3]}],
        [{"frame_index": 0, "rgb": [300, 0, 0], "lab": [1, 2, 3]}],
        [{"frame_index": 0, "rgb": [0, 0, 0], "lab": [150, 0, 0]}],
        [{"frame_index": 0, "rgb": [0, 0, 0], "lab": [1, 2, 3], "detection_confidence": 1.4}],
        [{"frame_index": -1, "rgb": [0, 0, 0], "lab": [1, 2, 3]}],
        [{"frame_index": 0, "rgb": [True, 0, 0], "lab": [1, 2, 3]}],
        [{"frame_index": 0, "rgb": [0, 0, 0], "lab": [1, 2, 3], "timestamp_seconds": -5.0}],
        [{"frame_index": 0, "rgb": [0, 0, 0], "lab": [1, 2, 3], "flame_area_ratio": 12.0}],
        [{"frame_index": 0, "rgb": [0, 0, 0], "lab": [1, 2, 3], "image_base64": "iVBORw0K"}],
        [{"frame_index": 0, "rgb": [0, 0, 0], "lab": [1, 2, 3], "mask": "iVBORw0K"}],
        [{"frame_index": 0, "rgb": [0, 0, 0], "lab": [1, 2, 3], "filename": "frame3.png"}],
        ["frame 0 was orange"],
        [None],
        "frame",
    ],
    ids=[
        "no-frames",
        "missing-lab",
        "missing-rgb",
        "missing-index",
        "short-rgb",
        "short-lab",
        "long-rgb",
        "non-numeric-rgb",
        "rgb-out-of-range",
        "lab-out-of-range",
        "confidence-out-of-range",
        "negative-index",
        "boolean-channel",
        "negative-timestamp",
        "ratio-out-of-range",
        "base64-image",
        "mask",
        "filename",
        "not-an-object",
        "null-frame",
        "frames-not-a-list",
    ],
)
def test_malformed_frame_evidence_is_rejected(monkeypatch, frames):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        response = client.post("/video-material-analysis", json={"frames": frames})

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    # Rejected before any Gemini spend.
    assert _total_calls(recorders) == 0


def test_images_and_base64_are_never_accepted(monkeypatch):
    """The endpoint is numbers only - there is no image input path at all."""
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        for extra in ("image_base64", "image", "frame_image", "mask_base64", "b64"):
            response = client.post(
                "/video-material-analysis",
                json={"frames": [{**FRAMES[0], extra: "iVBORw0KGgo="}]},
            )
            assert response.status_code == 422, extra
            assert response.json()["error"]["code"] == "VALIDATION_ERROR", extra
    assert _total_calls(recorders) == 0


def test_more_frames_than_the_cap_is_rejected(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    frames = [dict(FRAMES[0], frame_index=index) for index in range(MAX_VIDEO_FRAMES + 1)]
    with _client(monkeypatch) as client:
        response = client.post("/video-material-analysis", json={"frames": frames})
    assert response.status_code == 422
    assert _total_calls(recorders) == 0


def test_the_cap_limit_itself_is_accepted(monkeypatch):
    frames = [dict(FRAMES[0], frame_index=index) for index in range(MAX_VIDEO_FRAMES)]
    with _client(monkeypatch, _ok_handler(_valid_payload())) as client:
        body = _post(client, frames)
    assert body["frames_analyzed"] == MAX_VIDEO_FRAMES


def test_a_frame_without_optional_evidence_still_works(monkeypatch):
    """Absent timestamps/confidences are omitted, never invented."""
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    minimal = [{"frame_index": 3, "rgb": [166, 80, 27], "lab": [43.8, 31.4, 43.0]}]
    with _client(monkeypatch) as client:
        body = _post(client, minimal)

    assert body["available"] is True
    contents = recorders[0].calls[0]["contents"]
    assert "t=" not in contents
    assert "detection confidence" not in contents
    assert "segmentation confidence" not in contents
    assert "flame area" not in contents
    # The measured colour is still there, and still one consolidated answer.
    assert "[166, 80, 27]" in contents


def test_no_usable_evidence_never_reaches_gemini(monkeypatch):
    """The library entry point refuses to spend a call on empty evidence."""

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("Gemini must not be called without frame evidence")

    monkeypatch.setattr(video_material, "gemini_video_material_analysis", explode)

    payload = video_material_analysis([], _settings())
    assert payload["available"] is False
    assert payload["frames_analyzed"] == 0
    assert payload["frames_supplied"] == 0


def test_video_material_analysis_is_usable_as_a_library_call(monkeypatch):
    _install(monkeypatch, _ok_handler(_valid_payload()))
    payload = video_material_analysis(
        [FrameEvidence(**frame) for frame in FRAMES], _settings(), _database()
    )
    assert payload["available"] is True
    assert payload["primary_material"] == CANONICAL_MATERIALS[0]
    assert payload["frames_analyzed"] == len(FRAMES)


# ---------------------------------------------------------------------------
# Evidence normalisation (library level)
# ---------------------------------------------------------------------------
def test_frame_evidence_reuses_the_image_evidence_vocabulary():
    evidence = frame_evidence(FrameEvidence(**FRAMES[1]))

    assert evidence["frame_index"] == 1
    assert evidence["timestamp_seconds"] == 4.2
    assert evidence["mean_color"] == {"rgb": [166, 80, 27], "lab": [43.8, 31.4, 43.0]}
    assert set(evidence["hsv"]) == {"h", "s", "v"}
    assert evidence["saturation_0_100"] == evidence["hsv"]["s"]
    assert evidence["brightness_0_255"] == pytest.approx(
        0.299 * 166 + 0.587 * 80 + 0.114 * 27, abs=0.05
    )
    assert evidence["detection_confidence"] == 0.78
    assert evidence["segmentation_confidence"] == 0.91
    assert evidence["flame_region"]["mask_area_ratio"] == 0.14


def test_frame_evidence_omits_what_was_not_supplied():
    evidence = frame_evidence(
        FrameEvidence(frame_index=0, rgb=[10, 20, 30], lab=[5.0, 1.0, 2.0])
    )
    assert "timestamp_seconds" not in evidence
    assert "detection_confidence" not in evidence
    assert "segmentation_confidence" not in evidence
    assert "flame_region" not in evidence


def test_build_video_evidence_aggregates_the_collection():
    evidence = build_video_evidence([FrameEvidence(**frame) for frame in FRAMES])

    assert evidence["frame_count"] == len(FRAMES)
    assert len(evidence["frames"]) == len(FRAMES)
    collection = evidence["collection"]
    assert collection["frame_count"] == len(FRAMES)
    assert collection["time_span_seconds"] == 18.1
    assert len(collection["mean_rgb"]) == 3
    assert len(collection["mean_lab"]) == 3
    assert len(collection["lab_std_dev"]) == 3
    # The mean is of the measured colours, and the spread is non-zero because the
    # frames genuinely differ.
    assert collection["mean_lab"][0] == pytest.approx(62.28, abs=0.01)
    assert any(value > 0 for value in collection["lab_std_dev"])
    assert collection["mean_flame_area_ratio"] == pytest.approx(0.148, abs=1e-4)


def test_build_video_evidence_is_empty_without_frames():
    assert build_video_evidence([]) == {}


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
def test_the_api_key_is_never_exposed(monkeypatch):
    with _client(monkeypatch, _ok_handler(_valid_payload())) as client:
        response = client.post("/video-material-analysis", json={"frames": FRAMES})

    assert response.status_code == 200
    assert "test-key-do-not-leak" not in response.text
    assert "GEMINI_API_KEY" not in response.text


def test_the_prompt_contains_no_image_data(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        _post(client)

    call = recorders[0].calls[0]
    # The frames arrive as one text prompt; there are no inline data parts.
    assert isinstance(call["contents"], str)
    lowered = call["contents"].lower()
    for forbidden in ("base64", "data:image", "b64", ".png", ".jpg", "jpeg"):
        assert forbidden not in lowered, forbidden


# ---------------------------------------------------------------------------
# The image workflow must be untouched
# ---------------------------------------------------------------------------
def test_the_image_workflow_still_makes_one_call_per_image(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        response = client.post(
            "/analyze", files={"image": ("f.png", _png_bytes(), "image/png")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ai_material_analysis"]["available"] is True
    assert body["ai_material_analysis"]["primary_material"] == CANONICAL_MATERIALS[0]
    assert len(recorders[0].calls) == 1
    # The image prompt is unchanged: single-frame wording, no video instruction.
    call = recorders[0].calls[0]
    assert VIDEO_CONSOLIDATION_INSTRUCTION not in call["contents"]
    assert call["config"].system_instruction == SYSTEM_INSTRUCTION


def test_the_image_and_video_prompts_are_distinct():
    assert VIDEO_SYSTEM_INSTRUCTION != SYSTEM_INSTRUCTION
    assert VIDEO_CONSOLIDATION_INSTRUCTION not in SYSTEM_INSTRUCTION
    assert "Do not infer material identity from flame color alone" in VIDEO_SYSTEM_INSTRUCTION
    assert "Return structured JSON only." in VIDEO_SYSTEM_INSTRUCTION


def test_an_image_analysis_and_a_video_analysis_do_not_share_a_call(monkeypatch):
    recorders = _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        client.post("/analyze", files={"image": ("f.png", _png_bytes(), "image/png")})
        _post(client)

    calls = recorders[0].calls
    assert len(calls) == 2, "one call for the image and one for the whole video"
    assert [call["config"].system_instruction for call in calls] == [
        SYSTEM_INSTRUCTION,
        VIDEO_SYSTEM_INSTRUCTION,
    ]


def test_the_video_endpoint_does_not_disturb_analyze(monkeypatch):
    """Analysing a video leaves the image pipeline exactly as it was."""
    _install(monkeypatch, _ok_handler(_valid_payload()))
    with _client(monkeypatch) as client:
        _post(client)
        response = client.post(
            "/analyze", files={"image": ("f.png", _png_bytes(), "image/png")}
        )

    body = response.json()
    expected = {
        key: value
        for key, value in SUCCESS_RESULT.items()
        if key not in {"timing", "ai_material_analysis"}
    }
    assert {key: value for key, value in body.items() if key != "timing"} == (
        expected | {"ai_material_analysis": body["ai_material_analysis"]}
    )
    assert body["ai_material_analysis"]["available"] is True


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------
def test_openapi_documents_the_video_endpoint(monkeypatch):
    with _client(monkeypatch) as client:
        schema = client.get("/openapi.json").json()

    operation = schema["paths"]["/video-material-analysis"]["post"]
    assert operation["summary"]
    assert "ONE Gemini request" in operation["description"]
    assert set(operation["responses"]) >= {"200", "422"}

    body = schema["components"]["schemas"]["VideoMaterialAnalysisRequest"]
    assert body["required"] == ["frames"]
    assert body["additionalProperties"] is False

    frame = schema["components"]["schemas"]["FrameEvidence"]
    assert set(frame["required"]) == {"frame_index", "rgb", "lab"}
    assert frame["additionalProperties"] is False


def test_openapi_documents_the_consolidation_guarantee(monkeypatch):
    with _client(monkeypatch) as client:
        schema = client.get("/openapi.json").json()

    description = schema["info"]["description"]
    assert "/video-material-analysis" in description
    assert "never called per frame" in description
    assert "GEMINI_API_KEY" in description