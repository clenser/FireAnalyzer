"""FastAPI HTTP layer for the FlameAnalyzer backend.

This module is intentionally thin.  It performs transport concerns only -
request validation, size limits, decoding uploaded bytes, calling the existing
:func:`app.imaging.decode_image_bytes` helper, invoking the existing
``FlameAnalyzer`` and mapping the existing error codes onto HTTP statuses::

    HTTP request
        -> upload validation / size limit
        -> app.imaging.decode_image_bytes(...)
        -> FlameAnalyzer.analyze_image(...)
        -> existing schemas (already JSON-safe)
        -> JSON response

No detection, segmentation, clustering or material-matching logic lives here.

Run locally with::

    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .analyzer import FlameAnalyzer
from .config import Settings
from .errors import (
    AnalysisError,
    AnalyzerUnavailableError,
    ImageTooLargeError,
    InferenceError,
    InvalidImageError,
    MissingDatabaseError,
    MissingModelError,
    MissingUploadError,
    NoFireDetectedError,
    NoFlamePixelsError,
    ValidationError,
    error_payload,
)
from .imaging import decode_image_bytes

__all__ = ["app", "create_app", "SERVICE_NAME", "ERROR_STATUS"]

logger = logging.getLogger(__name__)

SERVICE_NAME = "FlameAnalyzer"
BYTES_PER_MB = 1024 * 1024
#: multipart/form-data adds a small envelope around the file itself.
_ENVELOPE_ALLOWANCE = 64 * 1024
_READ_CHUNK = 512 * 1024

#: Mapping from the existing error codes in :mod:`app.errors` to HTTP statuses.
#:
#: ``NO_FIRE_DETECTED`` / ``NO_FLAME_PIXELS`` are *valid analysis outcomes* for a
#: perfectly good image, so they return 200 with ``success: false`` in the body.
ERROR_STATUS: dict[str, int] = {
    MissingUploadError.code: 422,
    ValidationError.code: 422,
    InvalidImageError.code: 400,
    ImageTooLargeError.code: 413,
    NoFireDetectedError.code: 200,
    NoFlamePixelsError.code: 200,
    MissingModelError.code: 503,
    MissingDatabaseError.code: 503,
    AnalyzerUnavailableError.code: 503,
    InferenceError.code: 500,
}


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
def _error_response(error: AnalysisError) -> JSONResponse:
    """Render an :class:`AnalysisError` as a JSON response with a mapped status."""
    return JSONResponse(
        status_code=ERROR_STATUS.get(error.code, 500),
        content=error_payload(error.code, error.message),
    )


def _result_response(result: dict[str, Any]) -> JSONResponse:
    """Render a pipeline result, mapping a failure code onto its HTTP status."""
    if result.get("success") is True:
        return JSONResponse(status_code=200, content=result)
    code = ((result.get("error") or {}).get("code")) or InferenceError.code
    return JSONResponse(status_code=ERROR_STATUS.get(code, 500), content=result)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class MaxUploadSizeMiddleware:
    """Reject oversized requests using only the request headers.

    Runs before the body (and therefore the multipart parser) is touched, so a
    multi-gigabyte upload is refused without being buffered in memory.  The
    endpoint additionally caps the number of bytes it reads, which covers
    chunked uploads that carry no ``Content-Length``.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.limit = max_bytes + _ENVELOPE_ALLOWANCE

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        content_length: str | None = None
        for key, value in scope.get("headers", ()):
            if key == b"content-length":
                content_length = value.decode("latin-1")
                break

        if content_length and content_length.isdigit() and int(content_length) > self.limit:
            logger.warning(
                "Rejected request with Content-Length=%s (limit=%s)", content_length, self.limit
            )
            response = _error_response(ImageTooLargeError())
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(
    settings: Settings | None = None,
    analyzer_factory: Callable[[], FlameAnalyzer] | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    settings:
        Configuration to use.  Defaults to :meth:`Settings.from_env`.
    analyzer_factory:
        Factory used to build the analyzer during start-up.  Defaults to a
        single ``FlameAnalyzer(settings)``; tests inject a stub here so the unit
        tests never touch the model weights.
    """
    settings = settings or Settings.from_env()
    factory = analyzer_factory or (lambda: FlameAnalyzer(settings))

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Loading FlameAnalyzer (max_image_mb=%s, cors_origins=%s)",
            settings.max_image_mb,
            ",".join(settings.cors_origins) or "none",
        )
        analyzer = factory()  # models are loaded exactly once, here
        application.state.analyzer = analyzer
        logger.info("FlameAnalyzer ready (%s)", type(analyzer).__name__)
        try:
            yield
        finally:
            application.state.analyzer = None
            logger.info("FlameAnalyzer released")

    application = FastAPI(
        title=f"{SERVICE_NAME} API",
        version=__version__,
        summary="Flame detection, colour analysis and material matching.",
        description=(
            "Headless inference API for the FlameAnalyzer pipeline: YOLO fire "
            "detection, YOLO flame segmentation, K-Means/GMM flame colour "
            "analysis in CIELAB and deterministic material matching against "
            "`flame_dataset.json`.\n\n"
            "No UI, no LLM calls, no tunnels. Models are loaded once per worker "
            "process and reused for every request."
        ),
        lifespan=lifespan,
    )

    # One analyzer per worker process, plus the lock that serialises access to
    # it (Ultralytics predictors are not safe to share across threads).
    application.state.analyzer = None
    application.state.settings = settings
    application.state.inference_lock = threading.Lock()

    application.add_middleware(
        MaxUploadSizeMiddleware, max_bytes=settings.max_image_mb * BYTES_PER_MB
    )

    if settings.cors_origins:
        # A wildcard origin is allowed, but never together with credentials:
        # that combination is rejected by browsers and is unsafe anyway.
        allow_credentials = "*" not in settings.cors_origins
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for origins: %s", ", ".join(settings.cors_origins))

    # -- helpers -----------------------------------------------------------
    def _require_analyzer(request: Request) -> FlameAnalyzer:
        analyzer = getattr(request.app.state, "analyzer", None)
        if analyzer is None:
            raise AnalyzerUnavailableError()
        return analyzer

    async def _read_limited(upload: UploadFile, limit: int) -> bytes:
        """Read at most ``limit`` bytes, never buffering more than that."""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(_READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ImageTooLargeError()
            chunks.append(chunk)
        return b"".join(chunks)

    # -- routes ------------------------------------------------------------
    @application.get("/", tags=["meta"], summary="Service identification")
    async def root() -> dict[str, str]:
        """Minimal service identification. No inference is performed."""
        return {"service": SERVICE_NAME, "status": "ok", "version": __version__}

    @application.get("/health", tags=["meta"], summary="Readiness probe")
    async def health(request: Request) -> JSONResponse:
        """Report readiness without running inference.

        Returns 200 once the models are loaded, 503 while start-up is in
        progress or if a required artifact is missing.
        """
        analyzer = getattr(request.app.state, "analyzer", None)
        if analyzer is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "models_loaded": False,
                    **error_payload(AnalyzerUnavailableError.code, AnalyzerUnavailableError.message),
                },
            )
        ready = bool(analyzer.models_loaded)
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "degraded",
                "models_loaded": ready,
                "segmentation_available": bool(analyzer.segmentation_available),
                "device": analyzer.device,
            },
        )

    @application.post(
        "/analyze",
        tags=["analysis"],
        summary="Analyse an image for fire, flame colour and likely material",
        responses={
            200: {
                "description": (
                    "Analysis finished. Inspect `success`: it is `false` (with an "
                    "`error.code` of `NO_FIRE_DETECTED` or `NO_FLAME_PIXELS`) when "
                    "the image was valid but contained no usable flame."
                ),
                "content": {"application/json": {"example": _SUCCESS_EXAMPLE}},
            },
            400: {
                "description": "The upload could not be decoded as an image.",
                "content": {"application/json": {"example": _INVALID_IMAGE_EXAMPLE}},
            },
            413: {
                "description": "The upload exceeded the configured maximum size.",
                "content": {"application/json": {"example": _TOO_LARGE_EXAMPLE}},
            },
            422: {
                "description": (
                    "The `image` upload field was missing or the multipart body "
                    "could not be parsed."
                ),
                "content": {"application/json": {"example": _VALIDATION_EXAMPLE}},
            },
            500: {
                "description": "Unexpected internal analysis failure.",
                "content": {"application/json": {"example": _INFERENCE_ERROR_EXAMPLE}},
            },
            503: {
                "description": "Models are not loaded or the service is not ready.",
                "content": {"application/json": {"example": _UNAVAILABLE_EXAMPLE}},
            },
        },
    )
    async def analyze(request: Request, image: UploadFile = File(description="Image file to analyse")) -> JSONResponse:
        """Run the existing FlameAnalyzer pipeline on an uploaded image.

        Accepts `multipart/form-data` with a single `image` field. The bytes are
        decoded in memory (no temporary files) and the response is exactly the
        structure produced by ``FlameAnalyzer.analyze_image``: no NumPy arrays,
        no masks, no model objects, no filesystem paths.
        """
        started = time.perf_counter()

        limit = max(1, int(settings.max_image_mb)) * BYTES_PER_MB
        logger.info(
            "Analyze request received (content_type=%s, filename=%s)",
            image.content_type,
            image.filename,
        )

        try:
            analyzer = _require_analyzer(request)
        except AnalysisError as error:
            return _error_response(error)

        try:
            content = await _read_limited(image, limit)
        except ImageTooLargeError:
            logger.warning("Rejected oversized upload (limit=%s bytes)", limit)
            return _error_response(
                ImageTooLargeError(
                    f"The uploaded image exceeds the maximum allowed size "
                    f"({settings.max_image_mb} MB)."
                )
            )

        if not content:
            return _error_response(MissingUploadError())

        try:
            # Decoding is CPU-bound, so it is pushed to a worker thread.
            image_bgr = await run_in_threadpool(decode_image_bytes, content)
        except AnalysisError as error:
            return _error_response(error)
        except Exception:  # noqa: BLE001 - never leak a decoder traceback
            logger.exception("Failed to decode the uploaded image")
            return _error_response(InvalidImageError())

        height, width = image_bgr.shape[:2]
        logger.info("Decoded upload %sx%s for analysis", width, height)

        def analyze_locked() -> dict:
            """Run one inference, serialised per worker process.

            The lock is acquired and released on a worker thread, never on the
            event loop: a blocking acquire on the loop would stop the holder
            from ever being resumed to release it, deadlocking the server.
            """
            with request.app.state.inference_lock:
                return analyzer.analyze_image(image_bgr)

        try:
            result = await run_in_threadpool(analyze_locked)
        except AnalysisError as error:
            return _error_response(error)
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected inference failure")
            return _error_response(InferenceError())

        if not isinstance(result, dict):  # pragma: no cover - defensive
            logger.error("FlameAnalyzer returned a non-dict result")
            return _error_response(InferenceError())

        response = _result_response(result)
        logger.info(
            "Analysis complete (success=%s, inference_ms=%.2f, total_ms=%.2f)",
            result.get("success"),
            (result.get("timing") or {}).get("total_ms", 0.0),
            (time.perf_counter() - started) * 1000.0,
        )
        return response

    # -- exception handlers ------------------------------------------------
    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Render FastAPI's own request validation failures in our error format.

        Only the field location and the reason are echoed back - never the
        offending input, which could contain uploaded bytes.
        """
        problems = []
        for detail in exc.errors():
            location = ".".join(str(part) for part in detail.get("loc", ()) if part != "body")
            message = str(detail.get("msg", "invalid value"))
            problems.append(f"{location or 'body'}: {message}")
        message = "Invalid request - " + "; ".join(problems) if problems else ValidationError.message
        logger.info("Rejected malformed request (%s)", message)
        return _error_response(ValidationError(message))

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Convert anything unexpected into a safe 500 (no stack traces)."""
        logger.exception("Unhandled exception while serving %s", request.url.path)
        return _error_response(InferenceError())

    return application


# ---------------------------------------------------------------------------
# OpenAPI examples (captured from a real run of `python main.py 5.png`)
# ---------------------------------------------------------------------------
_SUCCESS_EXAMPLE: dict[str, Any] = {
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
        "alternatives": [
            {"material": "Paper Products(Wood material)", "similarity": 0.8885},
            {"material": "Wax Materials", "similarity": 0.8885},
        ],
        "database_notes": "Yellow flame, cotton burns fast, wool self-extinguishes, ...",
        "score_basis": "mean LAB distance to flame_dataset.json reference colours",
    },
    "suppression_information": {
        "source": "flame_dataset.json",
        "material": "Natural Fibers",
        "methods": ["Water", "CO2", "Foam"],
        "database_notes": "Yellow flame, cotton burns fast, wool self-extinguishes, ...",
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

_MISSING_IMAGE_EXAMPLE: dict[str, Any] = {
    "success": False,
    "error": {
        "code": "MISSING_IMAGE",
        "message": "No image file was provided in the 'image' field.",
    },
}
_VALIDATION_EXAMPLE: dict[str, Any] = {
    "success": False,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid request - image: Field required",
    },
}
_INVALID_IMAGE_EXAMPLE: dict[str, Any] = {
    "success": False,
    "error": {
        "code": "INVALID_IMAGE",
        "message": "The uploaded data is not a decodable image.",
    },
}
_TOO_LARGE_EXAMPLE: dict[str, Any] = {
    "success": False,
    "error": {
        "code": "IMAGE_TOO_LARGE",
        "message": "The uploaded image exceeds the maximum allowed size (10 MB).",
    },
}
_INFERENCE_ERROR_EXAMPLE: dict[str, Any] = {
    "success": False,
    "error": {
        "code": "INFERENCE_ERROR",
        "message": "An unexpected error occurred during inference.",
    },
}
_UNAVAILABLE_EXAMPLE: dict[str, Any] = {
    "success": False,
    "error": {
        "code": "SERVICE_UNAVAILABLE",
        "message": "The analysis service is not ready to accept requests.",
    },
}

#: Module level application used by ``uvicorn app.api:app``.
app = create_app()
