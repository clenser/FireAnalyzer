"""Tests for the FastAPI HTTP layer.

The unit tests inject a stub analyzer through ``create_app``'s
``analyzer_factory`` hook, so they run in milliseconds and never touch the
model weights.  One integration test at the bottom exercises the real
``FlameAnalyzer`` with a real photo and is skipped when the weights or the
sample images are unavailable.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import ERROR_STATUS, create_app  # noqa: E402
from app.config import Settings  # noqa: E402
from app.errors import (  # noqa: E402
    AnalyzerUnavailableError,
    ImageTooLargeError,
    InferenceError,
    InvalidImageError,
    MissingUploadError,
    NoFireDetectedError,
)

ROOT = Path(__file__).resolve().parent.parent
DET_MODEL = ROOT / "models" / "OBJ_best.pt"
SEG_MODEL = ROOT / "models" / "SEG_best.pt"

SUCCESS_RESULT = {
    "success": True,
    "fire_detection": {
        "detected": True,
        "confidence": 0.7899,
        "bounding_box": {"x1": 208, "y1": 144, "x2": 530, "y2": 452},
    },
    "segmentation": {
        "available": True,
        "fallback_used": False,
        "flame_pixel_count": 44820,
        "mask_area_ratio": 0.068131,
        "confidence": 0.9578,
    },
    "flame_analysis": {
        "kmeans": {
            "rgb": [255, 231, 141],
            "lab": [92.17, -1.4, 47.2],
            "method": "kmeans",
            "cluster_count": 2,
            "samples_used": 2000,
            "pixels_sampled": True,
        },
        "gmm": {
            "rgb": [255, 221, 98],
            "lab": [89.09, -0.36, 63.31],
            "method": "gmm",
            "cluster_count": 2,
            "samples_used": 2000,
            "pixels_sampled": True,
        },
        "mean_color": {"rgb": [255, 221, 98], "lab": [89.09, -0.36, 63.31]},
        "flame_pixel_count": 44820,
        "samples_used": 2000,
        "pixels_sampled": True,
    },
    "material_analysis": {
        "primary_material": "Natural Fibers",
        "similarity": 0.897,
        "alternatives": [{"material": "Wax Materials", "similarity": 0.8885}],
        "database_notes": "Yellow flame, cotton burns fast, ...",
        "score_basis": "mean LAB distance to flame_dataset.json reference colours",
    },
    "suppression_information": {
        "source": "flame_dataset.json",
        "material": "Natural Fibers",
        "methods": ["Water", "CO2", "Foam"],
        "database_notes": "Yellow flame, cotton burns fast, ...",
    },
    "timing": {
        "total_ms": 83.2,
        "detection_ms": 41.0,
        "segmentation_ms": 22.0,
        "color_ms": 19.0,
        "material_ms": 0.9,
    },
    "error": None,
}


class StubAnalyzer:
    """Stands in for :class:`app.analyzer.FlameAnalyzer`."""

    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self.result = SUCCESS_RESULT if result is None else result
        self.raises = raises
        self.calls: list[np.ndarray] = []
        self._device = "cpu"
        self._models_loaded = True
        self._segmentation_available = True

    @property
    def device(self) -> str:
        return self._device

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    @property
    def segmentation_available(self) -> bool:
        return self._segmentation_available

    def analyze_image(self, image_bgr):
        self.calls.append(image_bgr)
        if self.raises is not None:
            raise self.raises
        return self.result


def _png_bytes(size: int = 96, color=(0, 140, 255)) -> bytes:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[20 : size - 20, 20 : size - 20] = color
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _client(stub: StubAnalyzer, **overrides) -> TestClient:
    settings = Settings.from_env().with_overrides(**overrides) if overrides else Settings.from_env()
    application = create_app(settings=settings, analyzer_factory=lambda: stub)
    return TestClient(application)


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------
def test_root_returns_service_identification():
    with _client(StubAnalyzer()) as client:
        response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "FlameAnalyzer"
    assert body["status"] == "ok"
    json.dumps(body)


def test_health_reports_ready_state():
    with _client(StubAnalyzer()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["segmentation_available"] is True
    assert body["device"] == "cpu"


def test_health_reports_unavailable_before_startup():
    # No context manager -> the lifespan (which loads the analyzer) never runs.
    client = _client(StubAnalyzer())
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["models_loaded"] is False
    assert body["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_analyze_is_unavailable_before_startup():
    client = _client(StubAnalyzer())
    response = client.post("/analyze", files={"image": ("a.png", _png_bytes(), "image/png")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# POST /analyze - happy path
# ---------------------------------------------------------------------------
def test_analyze_accepts_a_valid_image():
    stub = StubAnalyzer()
    with _client(stub) as client:
        response = client.post("/analyze", files={"image": ("fire.png", _png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body == SUCCESS_RESULT
    for section in (
        "fire_detection",
        "segmentation",
        "flame_analysis",
        "material_analysis",
        "suppression_information",
        "timing",
    ):
        assert section in body
    # The decoded image reached the analyzer as a BGR array.
    assert len(stub.calls) == 1
    assert stub.calls[0].shape == (96, 96, 3)
    # json.dumps must work on the response as-is.
    assert json.loads(json.dumps(response.json())) == body


def test_analyze_does_not_leak_paths_or_arrays():
    with _client(StubAnalyzer()) as client:
        response = client.post("/analyze", files={"image": ("fire.png", _png_bytes(), "image/png")})
    text = response.text
    assert "ndarray" not in text
    assert "models" not in response.headers.get("content-type", "")
    assert ".pt" not in text


def test_analyzer_is_created_once_per_app():
    created = []

    def factory():
        created.append(StubAnalyzer())
        return created[-1]

    settings = Settings.from_env()
    application = create_app(settings=settings, analyzer_factory=factory)
    with TestClient(application) as client:
        for _ in range(3):
            client.post("/analyze", files={"image": ("f.png", _png_bytes(), "image/png")})
    assert len(created) == 1
    assert len(created[0].calls) == 3


# ---------------------------------------------------------------------------
# POST /analyze - request errors
# ---------------------------------------------------------------------------
def test_analyze_without_image_returns_422():
    with _client(StubAnalyzer()) as client:
        response = client.post("/analyze")
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "image" in body["error"]["message"]


def test_analyze_with_empty_file_returns_422():
    with _client(StubAnalyzer()) as client:
        response = client.post("/analyze", files={"image": ("empty.png", b"", "image/png")})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MISSING_IMAGE"


def test_validation_error_does_not_echo_the_uploaded_payload():
    payload = b"SENSITIVE-UPLOAD-CONTENT"
    with _client(StubAnalyzer()) as client:
        response = client.post("/analyze", content=payload, headers={"Content-Type": "image/png"})
    assert response.status_code == 422
    assert "SENSITIVE-UPLOAD-CONTENT" not in response.text


def test_analyze_with_invalid_image_returns_400():
    stub = StubAnalyzer()
    with _client(stub) as client:
        response = client.post(
            "/analyze", files={"image": ("not.png", b"this is not an image", "image/png")}
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"
    assert stub.calls == []  # rejected before any inference


def test_analyze_rejects_oversized_upload_by_content_length():
    # Random bytes: incompressible, so the payload really is over the limit.
    # The size check happens on the headers, before the body is parsed.
    payload = os.urandom(2 * 1024 * 1024)
    stub = StubAnalyzer()
    with _client(stub, max_image_mb=1) as client:
        response = client.post("/analyze", files={"image": ("big.png", payload, "image/png")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    assert stub.calls == []


def test_analyze_rejects_oversized_chunked_upload():
    """No Content-Length header: the limit is enforced while reading the body."""
    boundary = "flameboundary"
    stub = StubAnalyzer()

    def body():
        yield (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; filename="big.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        for _ in range(8):
            yield b"\x00" * 200_000
        yield f"\r\n--{boundary}--\r\n".encode()

    with _client(stub, max_image_mb=1) as client:
        response = client.post(
            "/analyze",
            content=body(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"
    assert stub.calls == []


# ---------------------------------------------------------------------------
# POST /analyze - pipeline and server errors
# ---------------------------------------------------------------------------
def test_no_fire_detected_is_a_200_outcome():
    result = {
        "success": False,
        "error": {"code": "NO_FIRE_DETECTED", "message": NoFireDetectedError.message},
    }
    with _client(StubAnalyzer(result=result)) as client:
        response = client.post("/analyze", files={"image": ("f.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json() == result


def test_analysis_error_maps_to_http_status():
    cases = [
        (InvalidImageError(), 400),
        (NoFireDetectedError(), 200),
        (ImageTooLargeError(), 413),
        (MissingUploadError(), 422),
        (AnalyzerUnavailableError(), 503),
        (InferenceError(), 500),
    ]
    for error, expected in cases:
        with _client(StubAnalyzer(raises=error)) as client:
            response = client.post(
                "/analyze", files={"image": ("f.png", _png_bytes(), "image/png")}
            )
        assert response.status_code == expected, error.code
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == error.code
        assert ERROR_STATUS[error.code] == expected


def test_unexpected_exception_becomes_safe_500():
    stub = StubAnalyzer(raises=RuntimeError("internal detail / secret path C:\\models"))
    settings = Settings.from_env()
    application = create_app(settings=settings, analyzer_factory=lambda: stub)
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.post("/analyze", files={"image": ("f.png", _png_bytes(), "image/png")})

    assert response.status_code == 500
    body = response.json()
    assert body == {
        "success": False,
        "error": {"code": "INFERENCE_ERROR", "message": InferenceError.message},
    }
    assert "secret path" not in response.text
    assert "Traceback" not in response.text


class ConcurrencyProbe(StubAnalyzer):
    """Records whether two inferences ever overlap."""

    def __init__(self, delay: float = 0.2) -> None:
        super().__init__()
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def analyze_image(self, image_bgr):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return self.result
        finally:
            with self._guard:
                self.active -= 1


def test_concurrent_requests_are_serialised_without_deadlock():
    """Regression test: inference must be serialised, and must not deadlock.

    A `threading.Lock` acquired on the event loop would block the loop while
    waiting, so the request holding it could never resume to release it.
    """
    probe = ConcurrencyProbe()
    results: list[int] = []
    with _client(probe) as client:
        def send() -> None:
            response = client.post("/analyze", files={"image": ("probe.png", _png_bytes(8), "image/png")})
            results.append(response.status_code)

        threads = [threading.Thread(target=send) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive(), "request thread hung (deadlock)"
    assert results == [200, 200, 200, 200]
    assert probe.max_active == 1, f"inference ran {probe.max_active}x concurrently"


def test_unhandled_route_exception_becomes_safe_500():
    class ExplodingHealth(StubAnalyzer):
        @property
        def device(self) -> str:
            raise RuntimeError("device lookup exploded with C:\\secret")

    settings = Settings.from_env()
    application = create_app(settings=settings, analyzer_factory=lambda: ExplodingHealth())
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/health")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INFERENCE_ERROR"
    assert "secret" not in response.text


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
def test_cors_is_disabled_by_default():
    with _client(StubAnalyzer()) as client:
        response = client.get("/", headers={"Origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_configured_origin():
    with _client(StubAnalyzer(), cors_origins=("http://localhost:5173",)) as client:
        response = client.get("/", headers={"Origin": "http://localhost:5173"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"

        preflight = client.options(
            "/analyze",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert "POST" in preflight.headers["access-control-allow-methods"]


def test_cors_rejects_unlisted_origin():
    with _client(StubAnalyzer(), cors_origins=("http://localhost:5173",)) as client:
        response = client.get("/", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_wildcard_never_allows_credentials():
    with _client(StubAnalyzer(), cors_origins=("*",)) as client:
        response = client.get("/", headers={"Origin": "http://any.example"})
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------
def test_openapi_documents_the_analyze_endpoint():
    with _client(StubAnalyzer()) as client:
        schema = client.get("/openapi.json").json()
        docs = client.get("/docs")
    assert docs.status_code == 200
    assert set(schema["paths"]) >= {"/", "/health", "/analyze"}
    operation = schema["paths"]["/analyze"]["post"]
    assert operation["summary"]
    assert operation["description"]
    assert set(operation["responses"]) >= {"200", "400", "413", "422", "500", "503"}
    body_schema = schema["components"]["schemas"]["Body_analyze_analyze_post"]
    assert body_schema["required"] == ["image"]
    assert body_schema["properties"]["image"]["type"] == "string"
    assert body_schema["properties"]["image"]["contentMediaType"] == "application/octet-stream"


# ---------------------------------------------------------------------------
# Integration with the real models (skipped without weights / sample photos)
# ---------------------------------------------------------------------------
def _sample_image() -> Path | None:
    for name in ("5.png", "1.png", "2.png", "3.png", "4.png", "7.jpeg", "8.png", "9.jpg", "fire_1.jpg"):
        candidate = ROOT / name
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(
    not (DET_MODEL.is_file() and SEG_MODEL.is_file()) or _sample_image() is None,
    reason="custom YOLO weights or sample photos are unavailable",
)
def test_api_end_to_end_with_real_models():
    payload = _sample_image().read_bytes()
    settings = Settings.from_env()
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health").json()["models_loaded"] is True
        response = client.post("/analyze", files={"image": ("sample.png", payload, "image/png")})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    for section in (
        "fire_detection",
        "segmentation",
        "flame_analysis",
        "material_analysis",
        "suppression_information",
        "timing",
    ):
        assert section in body, section
    assert body["fire_detection"]["detected"] is True
    assert body["fire_detection"]["bounding_box"] is not None
    assert body["flame_analysis"]["mean_color"]["rgb"] is not None
    assert body["material_analysis"]["primary_material"] is not None
    assert body["suppression_information"]["source"] == "flame_dataset.json"
    assert body["timing"]["total_ms"] > 0
    json.dumps(body)
