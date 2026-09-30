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
from app.fire_classes import MATERIAL_FIRE_CLASS, MAPPING_SOURCE  # noqa: E402
from app.mask import MASK_ENCODING, encode_mask_base64, mask_from_base64  # noqa: E402
from app.material_matching import load_material_database  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DET_MODEL = ROOT / "models" / "OBJ_best.pt"
SEG_MODEL = ROOT / "models" / "SEG_best.pt"
DATASET = ROOT / "data" / "flame_dataset.json"
RETAINED_METHOD_KEYS = ("kmeans", "gmm", "bayesian_gmm", "dbscan", "agglomerative")

#: The stub result mirrors a real 96x96 run.  The mask is a real base64 PNG of an
#: L-shaped region, so the HTTP tests can decode it and check the pixel count -
#: it is not a placeholder string.  The mask (1200 px) is deliberately a
#: different shape from the detection box (60x60 = 3600 px).
_STUB_MASK_W = _STUB_MASK_H = 96
_STUB_MASK = np.zeros((_STUB_MASK_H, _STUB_MASK_W), dtype=np.uint8)
_STUB_MASK[20:60, 20:60] = 255
_STUB_MASK[40:60, 40:60] = 0
_STUB_MASK_COUNT = int(np.count_nonzero(_STUB_MASK))
_STUB_MASK_RATIO = round(_STUB_MASK_COUNT / float(_STUB_MASK.size), 6)
_STUB_MASK_B64 = encode_mask_base64(_STUB_MASK, compression=6)

_CLUSTER = {
    "rgb": [255, 203, 89],
    "lab": [84.7, 7.62, 62.79],
    "method": "kmeans",
    "cluster_count": 2,
    "samples_used": 2000,
    "pixels_sampled": True,
    "dominant_cluster": 0,
    "dominant_color": {"rgb": [255, 214, 105], "lab": [86.24, 2.1, 71.4]},
    "centroids": [
        {
            "index": 0,
            "size": 1502,
            "weight": 0.751,
            "rgb": [255, 214, 105],
            "lab": [86.24, 2.1, 71.4],
        },
        {
            "index": 1,
            "size": 498,
            "weight": 0.249,
            "rgb": [255, 165, 0],
            "lab": [72.1, 23.4, 78.9],
        },
    ],
    "noise_count": 0,
    "representative": "unweighted mean of the K-Means centroids",
    "fallback": False,
}


def _clustering(method: str, **overrides) -> dict:
    entry = {**_CLUSTER, "method": method}
    entry.update(overrides)
    return entry


SUCCESS_RESULT = {
    "success": True,
    "fire_detection": {
        "detected": True,
        "confidence": 0.7899,
        "bounding_box": {"x1": 10, "y1": 10, "x2": 70, "y2": 70},
        "bbox": {"x1": 10, "y1": 10, "x2": 70, "y2": 70},
    },
    "segmentation": {
        "available": True,
        "fallback_used": False,
        "flame_pixel_count": _STUB_MASK_COUNT,
        "mask_area_ratio": _STUB_MASK_RATIO,
        "confidence": 0.9578,
        "mask_width": _STUB_MASK_W,
        "mask_height": _STUB_MASK_H,
        "mask_encoding": "png_base64",
        "mask": _STUB_MASK_B64,
        "bbox_fallback_reason": None,
        "fallback": False,
    },
    "flame_analysis": {
        "kmeans": _clustering("kmeans"),
        "gmm": _clustering("gmm"),
        "bayesian_gmm": _clustering("bayesian_gmm"),
        "dbscan": _clustering("dbscan", cluster_count=1, noise_count=16),
        "agglomerative": _clustering("agglomerative"),
        "mean_color": {"rgb": [255, 221, 98], "lab": [89.09, -0.36, 63.31]},
        "flame_pixel_count": _STUB_MASK_COUNT,
        "samples_used": 2000,
        "pixels_sampled": True,
        "n_clusters": 2,
        "algorithms": ["K-Means", "GMM", "Bayesian GMM", "DBSCAN", "Agglomerative"],
        "skipped_reason": None,
    },
    "material_analysis": {
        "primary_material": "Natural Fibers",
        "similarity": 0.9268,
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
    "fire_class": {
        "class": "Class A",
        "description": "Ordinary combustibles",
        "confidence": 0.9268,
        "material": "Natural Fibers",
        "basis": "material 'Natural Fibers' -> Class A via app/fire_classes.py",
        "mapping_source": "app/fire_classes.py (documented material -> fire class mapping)",
        "notes": "Derived from the matched material; not predicted by the detection model.",
    },
    "extinguishing_agents": [
        {
            "name": "Water",
            "compound": None,
            "type": "cooling",
            "source": "flame_dataset.json",
            "fire_class": "Class A",
            "compound_basis": "name verbatim from flame_dataset.json; no compound asserted",
        },
        {
            "name": "CO2",
            "compound": None,
            "type": "oxygen_displacement",
            "source": "flame_dataset.json",
            "fire_class": "Class A",
            "compound_basis": "name verbatim from flame_dataset.json; no compound asserted",
        },
        {
            "name": "Foam",
            "compound": None,
            "type": "blanketing",
            "source": "flame_dataset.json",
            "fire_class": "Class A",
            "compound_basis": "name verbatim from flame_dataset.json; no compound asserted",
        },
    ],
    "timing": {
        "total_ms": 83.2,
        "detection_ms": 41.0,
        "segmentation_ms": 22.0,
        "color_ms": 19.0,
        "material_ms": 0.9,
    },
    "ai_material_analysis": {
        "available": False,
        "error": "Gemini material analysis is disabled (GEMINI_ENABLED is not enabled).",
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
    # Gemini stays off unless a test explicitly enables it, so the shared
    # fixture is deterministic regardless of the ambient environment.
    overrides.setdefault("gemini_enabled", False)
    settings = Settings.from_env().with_overrides(**overrides)
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


# ---------------------------------------------------------------------------
# POST /analyze - the new fields
# ---------------------------------------------------------------------------
def _analyzed(client: TestClient) -> dict:
    response = client.post("/analyze", files={"image": ("fire.png", _png_bytes(), "image/png")})
    assert response.status_code == 200, response.text
    return response.json()


def test_response_carries_the_segmentation_mask():
    with _client(StubAnalyzer()) as client:
        body = _analyzed(client)

    segmentation = body["segmentation"]
    assert segmentation["mask_encoding"] == MASK_ENCODING
    assert isinstance(segmentation["mask"], str) and segmentation["mask"]
    assert segmentation["mask_width"] == _STUB_MASK_W
    assert segmentation["mask_height"] == _STUB_MASK_H
    assert segmentation["available"] is True


def test_the_mask_decodes_over_http_and_matches_the_reported_count():
    """The client can recover the exact pixels the backend measured."""
    with _client(StubAnalyzer()) as client:
        body = _analyzed(client)

    segmentation = body["segmentation"]
    decoded = mask_from_base64(segmentation["mask"])

    assert decoded is not None, "the API shipped an undecodable mask"
    assert decoded.shape == (segmentation["mask_height"], segmentation["mask_width"])
    assert int(decoded.sum()) == segmentation["flame_pixel_count"]
    assert segmentation["mask_area_ratio"] == pytest.approx(
        int(decoded.sum()) / decoded.size, abs=1e-6
    )
    # The mask is an L, not the detection rectangle: the corner is cut out.
    assert decoded[20:60, 20:60].any() and not decoded[40:60, 40:60].any()


def test_bounding_box_and_mask_are_separate_in_the_response():
    with _client(StubAnalyzer()) as client:
        body = _analyzed(client)

    box = body["fire_detection"]["bounding_box"]
    assert body["fire_detection"]["bbox"] == box  # short alias preserved
    box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    assert box_area == 3600
    assert body["segmentation"]["flame_pixel_count"] == 1200
    assert body["segmentation"]["flame_pixel_count"] != box_area


def test_response_carries_all_five_retained_algorithms():
    with _client(StubAnalyzer()) as client:
        body = _analyzed(client)

    flame = body["flame_analysis"]
    for method in RETAINED_METHOD_KEYS:
        entry = flame[method]
        assert entry is not None, f"{method} missing from the response"
        assert entry["method"] == method
        assert entry["centroids"]
        assert entry["representative"]
    assert flame["algorithms"] == [
        "K-Means",
        "GMM",
        "Bayesian GMM",
        "DBSCAN",
        "Agglomerative",
    ]
    assert "meanshift" not in json.dumps(body).lower()


def test_response_carries_the_fire_class():
    with _client(StubAnalyzer()) as client:
        body = _analyzed(client)

    fire_class = body["fire_class"]
    material = body["material_analysis"]["primary_material"]
    assert fire_class["class"] == MATERIAL_FIRE_CLASS[material]
    assert fire_class["material"] == material
    assert fire_class["mapping_source"] == MAPPING_SOURCE
    assert "not predicted" in fire_class["notes"]


def test_response_carries_extinguishing_agents_from_the_dataset():
    database = load_material_database(DATASET)
    with _client(StubAnalyzer()) as client:
        body = _analyzed(client)

    agents = body["extinguishing_agents"]
    material = body["material_analysis"]["primary_material"]
    assert [agent["name"] for agent in agents] == list(database.get(material).extinguishers)
    assert [agent["name"] for agent in agents] == body["suppression_information"]["methods"]
    for agent in agents:
        assert agent["compound"] is None
        assert agent["source"] == "flame_dataset.json"
        assert agent["fire_class"] == body["fire_class"]["class"]


def test_response_has_no_giant_uncompressed_pixel_array():
    """A 96x96 mask must travel as a small PNG, not a list of numbers."""
    with _client(StubAnalyzer()) as client:
        response = client.post(
            "/analyze", files={"image": ("fire.png", _png_bytes(), "image/png")}
        )

    body = response.json()
    # Structurally: no long array of numbers anywhere in the payload.
    assert not [
        value
        for section in body.values()
        if isinstance(section, dict)
        for value in section.values()
        if isinstance(value, list) and len(value) > 32
    ]
    assert isinstance(body["segmentation"]["mask"], str)
    # The mask travels as one short base64 string, not 9216 JSON numbers.
    raw_mask_as_json = _STUB_MASK.size * 4
    assert len(body["segmentation"]["mask"]) < raw_mask_as_json
    assert len(response.content) < raw_mask_as_json


def test_mask_can_be_omitted_from_the_response():
    result = json.loads(json.dumps(SUCCESS_RESULT))
    result["segmentation"]["mask"] = None
    result["segmentation"]["mask_encoding"] = None
    with _client(StubAnalyzer(result=result)) as client:
        body = _analyzed(client)

    assert body["segmentation"]["mask"] is None
    # The measurements survive, so a client can still report the flame area.
    assert body["segmentation"]["flame_pixel_count"] == _STUB_MASK_COUNT


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
        "fire_class",
        "extinguishing_agents",
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


def test_api_mask_survives_the_real_http_round_trip():
    """What a browser receives: a decodable mask matching the reported count."""
    payload = _sample_image().read_bytes()
    settings = Settings.from_env()
    with TestClient(create_app(settings=settings)) as client:
        response = client.post("/analyze", files={"image": ("sample.png", payload, "image/png")})

    assert response.status_code == 200, response.text
    body = response.json()
    segmentation = body["segmentation"]

    decoded = mask_from_base64(segmentation["mask"])
    assert decoded is not None
    assert decoded.shape == (segmentation["mask_height"], segmentation["mask_width"])
    assert int(decoded.sum()) == segmentation["flame_pixel_count"]
    assert segmentation["flame_pixel_count"] > 0

    for method in RETAINED_METHOD_KEYS:
        assert body["flame_analysis"][method] is not None

    material = body["material_analysis"]["primary_material"]
    assert body["fire_class"]["class"] == MATERIAL_FIRE_CLASS[material]
    assert body["extinguishing_agents"]


# ---------------------------------------------------------------------------
# CLI / API parity
# ---------------------------------------------------------------------------
def test_cli_prints_the_same_payload_the_api_returns(monkeypatch, capsys, tmp_path):
    """The CLI is a thin wrapper: identical analysis, identical JSON."""
    from app import cli

    monkeypatch.setattr(cli, "FlameAnalyzer", lambda settings=None, **kwargs: StubAnalyzer())
    image = tmp_path / "fire.png"
    image.write_bytes(_png_bytes())

    exit_code = cli.main([str(image)])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == SUCCESS_RESULT
    assert printed["segmentation"]["mask"] == _STUB_MASK_B64
    assert printed["fire_class"]["class"] == "Class A"
    assert len(printed["extinguishing_agents"]) == 3


def test_cli_no_mask_reaches_the_analyzer(monkeypatch, capsys, tmp_path):
    """`--no-mask` is a config override; the CLI must thread it through."""
    from app import cli

    seen: list[Settings] = []

    def factory(settings=None, **kwargs):
        seen.append(settings)
        return StubAnalyzer()

    monkeypatch.setattr(cli, "FlameAnalyzer", factory)
    image = tmp_path / "fire.png"
    image.write_bytes(_png_bytes())

    assert cli.main([str(image), "--no-mask"]) == 0
    printed = json.loads(capsys.readouterr().out)

    assert len(seen) == 1
    assert seen[0].emit_mask is False
    # The stub ignores settings, so the printed payload is the analyzer's own.
    assert printed == SUCCESS_RESULT


def test_cli_k_selection_and_clusters_reach_the_analyzer(monkeypatch, capsys, tmp_path):
    from app import cli

    seen: list[Settings] = []

    def factory(settings=None, **kwargs):
        seen.append(settings)
        return StubAnalyzer()

    monkeypatch.setattr(cli, "FlameAnalyzer", factory)
    image = tmp_path / "fire.png"
    image.write_bytes(_png_bytes())

    assert cli.main([str(image), "--k-selection", "fixed", "--clusters", "3"]) == 0
    capsys.readouterr()

    assert seen[0].k_selection == "fixed"
    assert seen[0].n_clusters == 3


def test_cli_reports_a_missing_image(monkeypatch, capsys, tmp_path):
    from app import cli

    assert cli.main([str(tmp_path / "absent.png")]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["success"] is False
    assert printed["error"]["code"] == "INVALID_IMAGE"
