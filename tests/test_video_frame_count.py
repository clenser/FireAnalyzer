"""Tests for the user-controlled `frame_count` video-analysis parameter.

Covers both the pure clamping logic (`resolve_frame_count`) and the
`/analyze-video` HTTP contract, with a stub analyzer that fabricates
per-frame results (no YOLO weights, no network calls) so the suite runs in
milliseconds.

Run with::

    python -m pytest tests/test_video_frame_count.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzer import ProcessedFrame  # noqa: E402
from app.api import create_app  # noqa: E402
from app.config import Settings  # noqa: E402
from app.material_matching import load_material_database  # noqa: E402
from app.video_analysis import (  # noqa: E402
    MAX_FRAMES_LIMIT,
    MIN_FRAMES_LIMIT,
    resolve_frame_count,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "flame_dataset.json"


# ---------------------------------------------------------------------------
# resolve_frame_count: pure clamping logic
# ---------------------------------------------------------------------------
def test_resolve_frame_count_omitted_uses_server_default():
    settings = Settings.from_env().with_overrides(video_max_frames=12)
    assert resolve_frame_count(None, settings) == 12


def test_resolve_frame_count_within_range_is_used_verbatim():
    settings = Settings.from_env().with_overrides(video_max_frames=12)
    assert resolve_frame_count(20, settings) == 20


def test_resolve_frame_count_below_minimum_is_clamped_up():
    settings = Settings.from_env().with_overrides(video_max_frames=12)
    assert resolve_frame_count(1, settings) == MIN_FRAMES_LIMIT


def test_resolve_frame_count_above_maximum_is_clamped_down():
    settings = Settings.from_env().with_overrides(video_max_frames=12)
    assert resolve_frame_count(10_000, settings) == MAX_FRAMES_LIMIT


# ---------------------------------------------------------------------------
# Stub analyzer: fabricates realistic per-frame results without ML weights.
# ---------------------------------------------------------------------------
class StubVideoAnalyzer:
    """Flame presence is decided by mean pixel brightness, so test videos can
    control it deterministically.  Material fusion is faked but keeps the
    same shape `app.analyzer.FlameAnalyzer.finalize` produces, specifically
    the `supporting_evidence.{deterministic,vision}_leader` fields that
    `app.video_analysis._evidence_summary` reads.
    """

    def __init__(self, database, *, force_uncertain: bool = False) -> None:
        self.database = database
        self.force_uncertain = force_uncertain
        self._models_loaded = True
        self._segmentation_available = True
        self._device = "cpu"
        self.frame_calls: list[np.ndarray] = []

    @property
    def device(self) -> str:
        return self._device

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    @property
    def segmentation_available(self) -> bool:
        return self._segmentation_available

    def process_frame(self, image_bgr: np.ndarray) -> ProcessedFrame:
        self.frame_calls.append(image_bgr)
        has_flame = float(image_bgr[..., 2].mean()) > 150.0
        if not has_flame:
            return ProcessedFrame(
                False,
                {
                    "success": False,
                    "error": {"code": "NO_FIRE_DETECTED", "message": "No fire detected."},
                    "detection_count": 0,
                    "mask_count": 0,
                },
            )
        mask = np.full(image_bgr.shape[:2], 255, dtype=np.uint8)
        deterministic = {
            "available": True,
            "leader": "wood",
            "reliability": 0.9,
            "evidence_quality": "strong",
            "ranking": [
                {"material": "wood", "similarity": 0.9, "lab_distance": 5.0, "support": 0.8},
                {"material": "plastic", "similarity": 0.4, "lab_distance": 40.0, "support": 0.2},
            ],
        }
        result = {"success": True, "detection_count": 1, "mask_count": 1}
        return ProcessedFrame(True, result, image=image_bgr, mask=mask, deterministic=deterministic)

    def finalize(self, frame: ProcessedFrame, vision=None, extra_timing=None):
        if not frame.success:
            return frame.result
        det = frame.deterministic or {}
        vision = vision or {}
        candidates = vision.get("candidates") or []
        vision_leader = candidates[0]["material"] if candidates else None
        final_material = None if self.force_uncertain else det.get("leader")
        result = dict(frame.result)
        result.update(
            {
                "vision_evidence": vision,
                "vision_provider": vision.get("vision_provider", "none"),
                "final_material": final_material,
                "confidence": 0.0 if self.force_uncertain else 0.82,
                "confidence_percent": 0.0 if self.force_uncertain else 82.0,
                "confidence_level": "low" if self.force_uncertain else "high",
                "uncertain": self.force_uncertain,
                "uncertainty_reasons": ["forced uncertain for the test"] if self.force_uncertain else [],
                "leading_candidates": [] if self.force_uncertain else [final_material],
                "candidate_materials": [
                    {"material": row["material"], "score": row["support"]} for row in det.get("ranking", [])
                ],
                "supporting_evidence": {
                    "deterministic_leader": det.get("leader"),
                    "vision_leader": vision_leader,
                },
                "fire_class": {
                    "class": "A",
                    "description": "",
                    "confidence": 0.0 if self.force_uncertain else 0.82,
                    "material": final_material,
                    "basis": "",
                    "mapping_source": "",
                    "notes": "",
                },
                "extinguishing_agents": [],
                "error": None,
            }
        )
        return result


def _write_video(path: str, frame_values: list[int], size: int = 64, fps: float = 10.0) -> None:
    """A tiny raw video where frame *i* is a solid-colour square of brightness
    `frame_values[i]` in the red channel - bright enough (>150) reads as
    "contains a flame" to `StubVideoAnalyzer.process_frame`.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (size, size))
    assert writer.isOpened()
    for value in frame_values:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :, 2] = value  # BGR: red channel
        writer.write(frame)
    writer.release()


@pytest.fixture
def flame_video(tmp_path: Path) -> Path:
    path = tmp_path / "flame.mp4"
    _write_video(str(path), [220] * 30)
    return path


@pytest.fixture
def no_flame_video(tmp_path: Path) -> Path:
    path = tmp_path / "no_flame.mp4"
    _write_video(str(path), [10] * 30)
    return path


def _client(stub: StubVideoAnalyzer, **overrides) -> TestClient:
    overrides.setdefault("gemini_enabled", False)
    overrides.setdefault("vision_enabled", False)  # no network calls in this suite
    # Cache is file-based and persists outside the TestClient; default it off so
    # tests that reuse identical video bytes never see another test's entry.
    # Tests exercising the cache itself pass an explicit per-test cache_dir.
    overrides.setdefault("cache_enabled", "cache_dir" in overrides)
    settings = Settings.from_env().with_overrides(**overrides)
    application = create_app(settings=settings, analyzer_factory=lambda: stub)
    return TestClient(application)


def _post_video(client: TestClient, path: Path, **form) -> "object":
    with path.open("rb") as handle:
        return client.post(
            "/analyze-video",
            files={"video": ("clip.mp4", handle, "application/octet-stream")},
            data=form,
        )


@pytest.fixture
def database():
    return load_material_database(DATASET)


# ---------------------------------------------------------------------------
# 1) frame_count validation and defaulting
# ---------------------------------------------------------------------------
def test_omitted_frame_count_preserves_default_behaviour(flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub, video_max_frames=12) as client:
        response = _post_video(client, flame_video)
    assert response.status_code == 200
    body = response.json()
    assert body["video"]["frames_requested"] == 12


def test_valid_small_frame_count_is_honoured(flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub) as client:
        response = _post_video(client, flame_video, frame_count=MIN_FRAMES_LIMIT)
    assert response.status_code == 200
    assert response.json()["video"]["frames_requested"] == MIN_FRAMES_LIMIT


def test_valid_normal_frame_count_is_honoured(flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub) as client:
        response = _post_video(client, flame_video, frame_count=10)
    assert response.status_code == 200
    assert response.json()["video"]["frames_requested"] == 10


def test_frame_count_below_minimum_is_rejected(flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub) as client:
        response = _post_video(client, flame_video, frame_count=MIN_FRAMES_LIMIT - 1)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_frame_count_above_maximum_is_rejected(flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub) as client:
        response = _post_video(client, flame_video, frame_count=MAX_FRAMES_LIMIT + 1)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_frame_count_must_be_an_integer(flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub) as client:
        response = _post_video(client, flame_video, frame_count="not-a-number")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# 2) vision budget stays independent of frame_count
# ---------------------------------------------------------------------------
def test_large_frame_count_does_not_blow_the_vision_budget(tmp_path, database):
    # Needs more raw frames than MAX_FRAMES_LIMIT, or sampling is capped by the
    # video's own length rather than by frame_count.
    path = tmp_path / "long_flame.mp4"
    _write_video(str(path), [220] * (MAX_FRAMES_LIMIT + 10))
    stub = StubVideoAnalyzer(database)
    with _client(stub, vision_enabled=True, video_vision_frames=2) as client:
        response = _post_video(client, path, frame_count=MAX_FRAMES_LIMIT)
    assert response.status_code == 200
    body = response.json()
    assert body["video"]["frames_requested"] == MAX_FRAMES_LIMIT
    # The vision budget (not frame_count) bounds how many frames got vision evidence.
    assert body["vision_frames"]["requested"] <= 2


# ---------------------------------------------------------------------------
# 3) cache identity incorporates the effective frame_count
# ---------------------------------------------------------------------------
def test_same_video_same_frame_count_is_served_from_cache(flame_video, database, tmp_path):
    stub = StubVideoAnalyzer(database)
    with _client(stub, cache_dir=str(tmp_path / "cache")) as client:
        first = _post_video(client, flame_video, frame_count=5)
        assert first.json()["cached"] is False
        second = _post_video(client, flame_video, frame_count=5)
        assert second.json()["cached"] is True
    assert len(stub.frame_calls) == 5  # only the first request ran the pipeline


def test_same_video_different_frame_count_bypasses_cache_and_differs(flame_video, database, tmp_path):
    stub = StubVideoAnalyzer(database)
    with _client(stub, cache_dir=str(tmp_path / "cache")) as client:
        first = _post_video(client, flame_video, frame_count=5)
        second = _post_video(client, flame_video, frame_count=10)
    assert first.json()["cached"] is False
    assert second.json()["cached"] is False
    assert first.json()["video"]["frames_requested"] == 5
    assert second.json()["video"]["frames_requested"] == 10


def test_force_new_analysis_bypasses_cache(flame_video, database, tmp_path):
    stub = StubVideoAnalyzer(database)
    with _client(stub, cache_dir=str(tmp_path / "cache")) as client:
        first = _post_video(client, flame_video, frame_count=5)
        assert first.json()["cached"] is False
        second = _post_video(client, flame_video, frame_count=5, force_new_analysis="true")
    assert second.json()["cached"] is False
    assert len(stub.frame_calls) == 10  # ran the pipeline twice, 5 frames each


# ---------------------------------------------------------------------------
# 4) response content: flame / no-flame / uncertain material
# ---------------------------------------------------------------------------
def test_video_with_flame_reports_detection_summary_and_evidence(flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub) as client:
        response = _post_video(client, flame_video, frame_count=6)
    body = response.json()
    assert body["success"] is True
    assert body["detection_summary"]["sampled_frames"] == 6
    assert body["detection_summary"]["flame_frames"] == 6
    assert body["detection_summary"]["text"] == "Flame seen in 6 of 6 sampled frames"
    assert body["final_material"] == "wood"
    assert body["evidence_summary"]["top_colour_match"]["material"] == "wood"
    assert len(body["frames"]) == 6
    assert body["representative_frames"]


def test_video_without_flame_reports_no_fire_detected(no_flame_video, database):
    stub = StubVideoAnalyzer(database)
    with _client(stub) as client:
        response = _post_video(client, no_flame_video, frame_count=6)
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "NO_FIRE_DETECTED"
    assert body["detection_summary"]["flame_frames"] == 0
    assert body["detection_summary"]["sampled_frames"] == 6
    assert len(body["frames"]) == 6  # every sampled frame's result is kept


def test_uncertain_material_still_exposes_strongest_evidence(flame_video, database):
    stub = StubVideoAnalyzer(database, force_uncertain=True)
    with _client(stub) as client:
        response = _post_video(client, flame_video, frame_count=6)
    body = response.json()
    assert body["success"] is True
    assert body["final_material"] is None
    assert body["uncertain"] is True
    # Even though the fused decision is uncertain, the deterministic colour
    # leader is still visible per frame and aggregated here.
    assert body["evidence_summary"]["top_colour_match"]["material"] == "wood"
    assert body["evidence_summary"]["colour_frames_considered"] == 6
    assert len(body["frames"]) == 6
