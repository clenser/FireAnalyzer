"""FastAPI HTTP layer for the FlameAnalyzer backend.

Transport concerns only - upload validation, size limits, decoding, the
48-hour result cache and error mapping.  The analysis itself lives in
:class:`app.analyzer.FlameAnalyzer` (detections, masks, colour evidence),
:mod:`app.vision_providers` (Groq / Gemini vision evidence),
:mod:`app.material_fusion` (the Python material decision) and
:mod:`app.video_analysis` (server-side video analysis)::

    POST /analyze        image -> SHA-256 cache -> <=3 detections -> <=3 masks
                         -> merged mask -> RGB/LAB -> vision evidence
                         -> Python fusion -> fire class -> cache
    POST /analyze-video  video -> SHA-256 cache -> frames -> per-frame pipeline
                         -> Python majority vote -> cache
    POST /material-identification   one client-measured colour -> Python fusion
    POST /video-material-analysis   client frame colours -> Python majority vote
    POST /activity       EC2 inactivity watchdog heartbeat (frontend only)

No LLM ever produces a final material or a final video conclusion.

Run locally with::

    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .analysis_cache import AnalysisCache, sha256_hex
from .analyzer import FlameAnalyzer
from .config import Settings
from .flame_evidence import extract_flame_evidence
from .errors import (
    AnalysisError,
    AnalyzerUnavailableError,
    DetectionFailedError,
    FrameExtractionError,
    ImageTooLargeError,
    InferenceError,
    InvalidImageError,
    InvalidVideoError,
    MissingDatabaseError,
    MissingModelError,
    MissingUploadError,
    NoFireDetectedError,
    NoFlamePixelsError,
    UnsupportedMediaError,
    ValidationError,
    VideoTooLargeError,
    error_payload,
)
from .gemini_analysis import gemini_material_analysis
from .imaging import decode_image_bytes
from .material_matching import MaterialDatabase, load_material_database
from .video_analysis import analyze_video_file
from .video_material import (
    MaterialIdentificationRequest,
    VideoMaterialAnalysisRequest,
    identify_material,
    video_material_analysis,
)
from .vision_providers import collect_vision_evidence, unavailable_vision

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
    DetectionFailedError.code: 500,
    InvalidVideoError.code: 400,
    FrameExtractionError.code: 400,
    VideoTooLargeError.code: 413,
    UnsupportedMediaError.code: 415,
}

#: Container extensions accepted by ``POST /analyze-video``.
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".3gp", ".wmv", ".flv")
_VIDEO_PATH = "/analyze-video"
#: Outcomes worth caching: completed analyses, including "no flame found".
_CACHEABLE_ERRORS = (NoFireDetectedError.code, NoFlamePixelsError.code)


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


def _query_flag(request: Request, name: str) -> bool:
    """Boolean query parameter (``1/true/yes/on``); the form field is preferred."""
    value = request.query_params.get(name)
    return bool(value) and value.strip().lower() in {"1", "true", "yes", "on"}


def _is_video_upload(upload: UploadFile) -> bool:
    content_type = (upload.content_type or "").lower()
    suffix = Path(upload.filename or "").suffix.lower()
    if content_type.startswith("video/"):
        return True
    return suffix in _VIDEO_EXTENSIONS and content_type in ("", "application/octet-stream")


def _cacheable(result: dict[str, Any]) -> bool:
    if result.get("success") is True:
        return True
    return ((result.get("error") or {}).get("code")) in _CACHEABLE_ERRORS


async def _ai_material_payload(request: Request, result: dict[str, Any]) -> dict[str, Any]:
    """The existing numerical Gemini analysis, as separate secondary evidence.

    It receives only numbers extracted from the deterministic result (never the
    image), never influences the fused material, and any failure becomes
    ``{"available": false, ...}``.
    """
    settings = request.app.state.settings
    evidence = extract_flame_evidence(result)
    if not evidence.get("mean_color"):
        return {
            "available": False,
            "role": "secondary_evidence_only",
            "error": "No flame colour evidence was extracted for the AI analysis.",
        }

    analyzer = getattr(request.app.state, "analyzer", None)
    database = None
    if analyzer is not None:
        try:
            database = analyzer.database
        except Exception:  # noqa: BLE001 - a stub/unloaded analyzer simply has no database
            database = None

    started = time.perf_counter()
    try:
        payload = await run_in_threadpool(gemini_material_analysis, evidence, settings, database)
    except Exception:  # noqa: BLE001 - the AI analysis must never fail the request
        logger.exception("AI material analysis failed unexpectedly")
        payload = {"available": False, "error": "The AI material analysis could not be completed."}
    payload["role"] = "secondary_evidence_only"
    if payload.get("available") is True:
        payload["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    return payload


# ---------------------------------------------------------------------------
# EC2 inactivity heartbeat
# ---------------------------------------------------------------------------
def _touch_activity_file(path: Path) -> None:
    """Record a frontend activity heartbeat for the EC2 inactivity watchdog.

    Writes the current Unix time into the file and refreshes its modification
    time, creating the parent directory when needed.  Only ``POST /activity``
    calls this - analysis requests never reset the watchdog.  Best-effort: a
    failure is logged and swallowed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        path.write_text(f"{int(now)}\n", encoding="utf-8")
        os.utime(path, (now, now))
    except OSError:
        logger.warning("Could not update the activity file %s", path)


# ---------------------------------------------------------------------------
# Material database access (no models required)
# ---------------------------------------------------------------------------
def _material_database(request: Request) -> MaterialDatabase:
    """The canonical material database, with or without a loaded analyzer.

    The deterministic matcher only needs ``flame_dataset.json``, so
    ``/material-identification`` works without any YOLO model loaded.  The
    analyzer's in-memory copy is preferred - it is the one ``/analyze`` matched
    against - and the same file is loaded directly when no analyzer is attached
    yet.  Raises :class:`~app.errors.MissingDatabaseError` (mapped to 503) when
    the dataset itself is unavailable.
    """
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is not None:
        try:
            return analyzer.database
        except Exception:  # noqa: BLE001 - an analyzer without a database simply has none
            logger.info("Analyzer exposes no material database; loading the dataset directly")
    return load_material_database(request.app.state.settings.dataset_path)


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

    def __init__(self, app: Any, max_bytes: int, max_video_bytes: int | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.limit = max_bytes + _ENVELOPE_ALLOWANCE
        self.video_limit = (max_video_bytes or max_bytes) + _ENVELOPE_ALLOWANCE

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        content_length: str | None = None
        for key, value in scope.get("headers", ()):
            if key == b"content-length":
                content_length = value.decode("latin-1")
                break

        is_video = scope.get("path") == _VIDEO_PATH
        limit = self.video_limit if is_video else self.limit
        if content_length and content_length.isdigit() and int(content_length) > limit:
            logger.warning(
                "Rejected request with Content-Length=%s (limit=%s)", content_length, limit
            )
            response = _error_response(VideoTooLargeError() if is_video else ImageTooLargeError())
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
        summary="Fire detection, flame segmentation, fire-class and agent analysis.",
        description=(
            "Headless inference API for the FlameAnalyzer pipeline.\n\n"
            "`POST /analyze`: up to 3 YOLO fire detections, one segmentation mask per "
            "detection, the masks merged into one flame mask (boxes are never merged), "
            "CIELAB/RGB colour evidence from the merged mask, vision evidence from Groq "
            "(Gemini vision as fallback) and a **Python** fusion engine that makes the "
            "final material decision (`final_material`, `confidence`, `uncertain`). "
            "The fire class and extinguishing agents come from the deterministic "
            "mapping in `app/fire_classes.py`. Vision models are evidence only.\n\n"
            "`POST /analyze-video`: the same per-frame pipeline over evenly spaced "
            "frames and a Python majority/consistency vote; returns up to 3 "
            "representative frames plus every analysed frame. No LLM produces the "
            "video conclusion.\n\n"
            "Results are cached for 48 hours by the SHA-256 of the uploaded file "
            "(`cached: true`); send `force_new_analysis=true` to re-run and replace "
            "the entry.\n\n"
            "The secondary numerical Gemini analysis (`ai_material_analysis`) remains "
            "separate evidence and never decides the material. API keys are read "
            "from the server environment only and are never returned or logged."
        ),
        lifespan=lifespan,
    )

    # One analyzer per worker process, plus the lock that serialises access to
    # it (Ultralytics predictors are not safe to share across threads).
    application.state.analyzer = None
    application.state.settings = settings
    application.state.inference_lock = threading.Lock()

    application.state.cache = AnalysisCache(settings.cache_dir, enabled=settings.cache_enabled)

    application.add_middleware(
        MaxUploadSizeMiddleware,
        max_bytes=settings.max_image_mb * BYTES_PER_MB,
        max_video_bytes=settings.max_video_mb * BYTES_PER_MB,
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
        "/activity",
        tags=["meta"],
        summary="EC2 inactivity watchdog heartbeat",
        responses={
            200: {
                "description": "Heartbeat accepted; the activity file was touched.",
                "content": {"application/json": {"example": {"status": "active"}}},
            },
        },
    )
    async def activity_post(request: Request) -> dict[str, str]:
        """Heartbeat for the EC2 inactivity watchdog.

        Updates ``settings.activity_file`` (default
        ``/var/lib/flame-analyzer/last-activity``), the only timestamp the
        watchdog uses.  The frontend calls this while the user is actively using
        the application; analysis requests never update it, so the shutdown
        countdown runs from the last frontend activity.  No authentication, no
        body, no expensive processing.  The update is best-effort: even if the file cannot be
        written the endpoint still returns 200 so a transient filesystem
        problem never turns into a frontend-visible error.
        """
        _touch_activity_file(Path(request.app.state.settings.activity_file))
        return {"status": "active"}

    @application.get(
        "/activity",
        tags=["meta"],
        summary="Activity endpoint test",
        responses={
            200: {
                "description": "The activity endpoint is reachable.",
                "content": {"application/json": {"example": {"status": "active"}}},
            },
        },
    )
    async def activity_get() -> dict[str, str]:
        """Lightweight liveness check for the activity endpoint. Testing only."""
        return {"status": "active"}

    @application.post(
        "/analyze",
        tags=["analysis"],
        summary="Analyse an image for fire, flame region, fire class and agents",
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
    async def analyze(
        request: Request,
        image: UploadFile = File(description="Image file to analyse"),
        force_new_analysis: bool = Form(
            False,
            description="Ignore any cached result, re-run the full analysis and replace the cache entry.",
        ),
    ) -> JSONResponse:
        """Run the full image pipeline on an uploaded image.

        Accepts `multipart/form-data` with an `image` field and an optional
        `force_new_analysis` boolean (also accepted as a query parameter).

        Pipeline: SHA-256 of the uploaded bytes -> 48-hour cache lookup (skipped
        when `force_new_analysis=true`) -> up to 3 YOLO detections -> one mask per
        detection -> merged mask -> RGB/LAB evidence -> vision evidence (Groq,
        Gemini fallback) -> Python fusion -> deterministic fire class -> cache.

        `cached` reports whether the result came from the cache.  The detection
        boxes (`detections[]`) are never merged; `segmentation.mask` is the merged
        flame mask used for every measurement.
        """
        started = time.perf_counter()
        force = force_new_analysis or _query_flag(request, "force_new_analysis")

        limit = max(1, int(settings.max_image_mb)) * BYTES_PER_MB
        logger.info(
            "Analyze request received (content_type=%s, force_new_analysis=%s)",
            image.content_type,
            force,
        )
        if _is_video_upload(image):
            return _error_response(
                UnsupportedMediaError("Videos are not accepted here; use POST /analyze-video.")
            )

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

        cache: AnalysisCache = request.app.state.cache
        digest = sha256_hex(content)
        if not force:
            cached = await run_in_threadpool(cache.get, "image", digest)
            if cached is not None:
                cached["cached"] = True
                logger.info("Analysis served from cache")
                return _result_response(cached)

        try:
            analyzer = _require_analyzer(request)
        except AnalysisError as error:
            return _error_response(error)

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

        try:
            result = await _run_image_pipeline(request, analyzer, image_bgr)
        except AnalysisError as error:
            return _error_response(error)
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected inference failure")
            return _error_response(InferenceError())

        if not isinstance(result, dict):  # pragma: no cover - defensive
            logger.error("FlameAnalyzer returned a non-dict result")
            return _error_response(InferenceError())

        result["analysis_type"] = "image"
        result["cached"] = False
        if _cacheable(result):
            await run_in_threadpool(cache.set, "image", digest, result)

        response = _result_response(result)
        logger.info(
            "Analysis complete (success=%s, inference_ms=%.2f, total_ms=%.2f)",
            result.get("success"),
            (result.get("timing") or {}).get("total_ms", 0.0),
            (time.perf_counter() - started) * 1000.0,
        )
        return response

    async def _run_image_pipeline(request: Request, analyzer: Any, image_bgr: Any) -> dict[str, Any]:
        """process_frame (locked) -> vision + numerical Gemini (concurrent) -> fusion."""

        def process_locked():
            """Run one inference, serialised per worker process.

            The lock is acquired and released on a worker thread, never on the
            event loop: a blocking acquire on the loop would stop the holder
            from ever being resumed to release it, deadlocking the server.
            """
            with request.app.state.inference_lock:
                if not hasattr(analyzer, "process_frame"):
                    return analyzer.analyze_image(image_bgr)
                return analyzer.process_frame(image_bgr)

        frame = await run_in_threadpool(process_locked)
        if isinstance(frame, dict):  # analyzer without the frame API
            if frame.get("success") is True:
                frame["ai_material_analysis"] = await _ai_material_payload(request, frame)
            return frame
        if not frame.success:
            return frame.result

        async def vision_task() -> tuple[dict[str, Any], float]:
            stage = time.perf_counter()
            try:
                block = await run_in_threadpool(
                    collect_vision_evidence, frame.image, frame.mask, settings, analyzer.database
                )
            except Exception:  # noqa: BLE001 - vision never breaks the analysis
                logger.exception("Vision evidence failed unexpectedly")
                block = unavailable_vision("vision evidence failed unexpectedly")
            return block, round((time.perf_counter() - stage) * 1000.0, 2)

        (vision, vision_ms), ai_payload = await asyncio.gather(
            vision_task(), _ai_material_payload(request, frame.result)
        )
        frame.image, frame.mask = None, None
        result = await run_in_threadpool(
            analyzer.finalize, frame, vision, {"vision_ms": vision_ms}
        )
        result["ai_material_analysis"] = ai_payload
        return result

    @application.post(
        "/analyze-video",
        tags=["analysis"],
        summary="Analyse a video: per-frame detection, masks, evidence and Python voting",
        responses={
            200: {
                "description": (
                    "Video analysed. `success: false` with `NO_FIRE_DETECTED` means the "
                    "video was valid but no sampled frame contained a usable flame."
                )
            },
            400: {"description": "The upload is not a decodable video (`INVALID_VIDEO`, `FRAME_EXTRACTION_FAILED`)."},
            413: {"description": "The upload exceeded `FLAME_MAX_VIDEO_MB` (`VIDEO_TOO_LARGE`)."},
            415: {"description": "The upload is not a video (`UNSUPPORTED_MEDIA`)."},
            422: {"description": "The `video` field is missing (`VALIDATION_ERROR`)."},
            503: {"description": "Models are not loaded."},
        },
    )
    async def analyze_video(
        request: Request,
        video: UploadFile = File(description="Video file to analyse"),
        force_new_analysis: bool = Form(
            False,
            description="Ignore any cached result, re-run the full analysis and replace the cache entry.",
        ),
    ) -> JSONResponse:
        """Full server-side video analysis.

        The cache key is the SHA-256 of the **original video file**.  Evenly
        spaced frames (`FLAME_VIDEO_MAX_FRAMES`) each go through the image
        pipeline (<=3 detections, <=3 masks, merged mask, RGB/LAB evidence,
        vision evidence for up to `FLAME_VIDEO_VISION_FRAMES` frames, Python
        fusion).  The video material and fire class come from a Python
        majority/consistency vote over the frame decisions - never from an LLM.
        Returns `representative_frames` (up to 3) and every frame in `frames`.
        """
        started = time.perf_counter()
        force = force_new_analysis or _query_flag(request, "force_new_analysis")
        if not _is_video_upload(video):
            return _error_response(UnsupportedMediaError("The uploaded file is not a supported video."))

        try:
            analyzer = _require_analyzer(request)
        except AnalysisError as error:
            return _error_response(error)

        limit = max(1, int(settings.max_video_mb)) * BYTES_PER_MB
        suffix = Path(video.filename or "").suffix.lower()
        suffix = suffix if suffix in _VIDEO_EXTENSIONS else ".mp4"
        tmp_path: str | None = None
        try:
            try:
                tmp_path, digest, size = await _spool_upload(video, limit, suffix)
            except VideoTooLargeError:
                return _error_response(
                    VideoTooLargeError(
                        f"The uploaded video exceeds the maximum allowed size ({settings.max_video_mb} MB)."
                    )
                )
            if size == 0:
                return _error_response(MissingUploadError("No video file was provided in the 'video' field."))

            cache: AnalysisCache = request.app.state.cache
            if not force:
                cached = await run_in_threadpool(cache.get, "video", digest)
                if cached is not None:
                    cached["cached"] = True
                    logger.info("Video analysis served from cache")
                    return _result_response(cached)

            try:
                result = await run_in_threadpool(
                    analyze_video_file,
                    tmp_path,
                    analyzer,
                    settings,
                    request.app.state.inference_lock,
                )
            except AnalysisError as error:
                return _error_response(error)
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected video analysis failure")
                return _error_response(InferenceError())

            result["cached"] = False
            if _cacheable(result):
                await run_in_threadpool(cache.set, "video", digest, result)
            logger.info(
                "Video analysis complete (success=%s, frames=%s, total_ms=%.2f)",
                result.get("success"),
                (result.get("video") or {}).get("frames_sampled"),
                (time.perf_counter() - started) * 1000.0,
            )
            return _result_response(result)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning("Could not delete a temporary video file")

    async def _spool_upload(upload: UploadFile, limit: int, suffix: str) -> tuple[str, str, int]:
        """Stream an upload to a temp file while hashing it; never buffers it all."""
        hasher = hashlib.sha256()
        total = 0
        fd, path = tempfile.mkstemp(prefix="flame-video-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                while True:
                    chunk = await upload.read(_READ_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise VideoTooLargeError()
                    hasher.update(chunk)
                    handle.write(chunk)
        except BaseException:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return path, hasher.hexdigest(), total

    @application.post(
        "/material-identification",
        tags=["analysis"],
        summary="Identify the material from an aggregated flame colour (deterministic)",
        responses={
            200: {
                "description": (
                    "The same deterministic material identification the image "
                    "workflow returns, plus the fire class and extinguishing "
                    "agents derived from it through the existing mapping."
                ),
                "content": {
                    "application/json": {"example": _MATERIAL_IDENTIFICATION_EXAMPLE}
                },
            },
            422: {
                "description": (
                    "The body was malformed: `rgb` and `lab` must each be exactly "
                    "three numeric values in range."
                ),
                "content": {"application/json": {"example": _VALIDATION_EXAMPLE}},
            },
            500: {"description": "Unexpected internal failure."},
            503: {
                "description": "`flame_dataset.json` could not be loaded.",
                "content": {"application/json": {"example": _UNAVAILABLE_EXAMPLE}},
            },
        },
    )
    async def material_identification(
        request: Request, payload: MaterialIdentificationRequest
    ) -> JSONResponse:
        """Run the existing deterministic material matcher on supplied evidence.

        Body: `{"rgb": [R, G, B], "lab": [L, a, b]}` - one aggregated flame colour,
        which is what a client gets after averaging its analysed video frames.
        RGB channels are 0-255; LAB is in the `scikit-image` convention the
        dataset uses (L 0-100, a/b -128..127).

        The colour is ranked by the same LAB matcher ``/analyze`` uses and decided
        by the same Python fusion engine (colour evidence only - there is no image
        for vision evidence here), so the result is `uncertain` whenever colour
        cannot separate the leading materials.  The fire class and agents come
        from `app/fire_classes.py`.
        """
        started = time.perf_counter()
        try:
            database = _material_database(request)
            body = identify_material(payload.rgb, payload.lab, database, settings)
        except AnalysisError as error:
            return _error_response(error)
        except Exception:  # noqa: BLE001 - never leak an internal traceback
            logger.exception("Material identification failed unexpectedly")
            return _error_response(InferenceError())

        logger.info(
            "Deterministic material identification complete (material=%s, similarity=%s, total_ms=%.2f)",
            body.get("primary_material"),
            body.get("similarity"),
            (time.perf_counter() - started) * 1000.0,
        )
        return JSONResponse(status_code=200, content=body)

    @application.post(
        "/video-material-analysis",
        tags=["analysis"],
        summary="Deterministic video material vote over client-measured frame colours",
        responses={
            422: {
                "description": (
                    "The body was malformed: `frames` must be a non-empty list of "
                    "frames, each with exactly three RGB and three LAB numbers. "
                    "Unknown fields - including any image or base64 payload - are "
                    "rejected."
                ),
                "content": {"application/json": {"example": _VALIDATION_EXAMPLE}},
            },
            503: {"description": "`flame_dataset.json` could not be loaded."},
        },
    )
    async def video_material_analysis_endpoint(
        request: Request, payload: VideoMaterialAnalysisRequest
    ) -> JSONResponse:
        """Decide every supplied frame with the Python fusion engine, then vote.

        No LLM is called: the former single-Gemini consolidated answer was
        removed.  Legacy fields (`available`, `primary_material`, `matches`,
        `overall_confidence_level`, `frames_supplied`, `frames_analyzed`,
        `display_threshold_percent`) are still returned, now derived from the
        deterministic vote.  For full server-side analysis use `/analyze-video`.
        """
        started = time.perf_counter()
        try:
            database = _material_database(request)
            body = await run_in_threadpool(
                video_material_analysis, payload.frames, settings, database
            )
        except AnalysisError as error:
            return _error_response(error)
        except Exception:  # noqa: BLE001 - never leak an internal traceback
            logger.exception("Video material analysis failed unexpectedly")
            return _error_response(InferenceError())

        logger.info(
            "Video material vote complete (frames=%s, material=%s, total_ms=%.2f)",
            body.get("frames_analyzed"),
            body.get("final_material"),
            (time.perf_counter() - started) * 1000.0,
        )
        return JSONResponse(status_code=200, content=body)

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
#: One clustering entry, trimmed to a single centroid so the docs stay readable.
#: The live response carries `centroids` for every cluster of every algorithm.
_CLUSTER_EXAMPLE: dict[str, Any] = {
    "rgb": [255, 203, 89],
    "lab": [84.7, 7.62, 62.79],
    "method": "kmeans",
    "cluster_count": 4,
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
        {"index": 1, "size": 498, "weight": 0.249, "rgb": [255, 165, 0], "lab": [72.1, 23.4, 78.9]},
    ],
    "noise_count": 0,
    "representative": "unweighted mean of the K-Means centroids",
    "fallback": False,
}

_SUCCESS_EXAMPLE: dict[str, Any] = {
    "success": True,
    "fire_detection": {
        "detected": True,
        "confidence": 0.7899,
        "bounding_box": {"x1": 208, "y1": 144, "x2": 530, "y2": 452},
        "bbox": {"x1": 208, "y1": 144, "x2": 530, "y2": 452},
    },
    "segmentation": {
        "available": True,
        "fallback_used": False,
        "flame_pixel_count": 44820,
        "mask_area_ratio": 0.068131,
        "confidence": 0.9578,
        "mask_width": 966,
        "mask_height": 681,
        "mask_encoding": "png_base64",
        "mask": "<base64 PNG, 966x681, one byte per pixel scaled to 0 or 255>",
        "bbox_fallback_reason": None,
        "fallback": False,
    },
    "flame_analysis": {
        "kmeans": _CLUSTER_EXAMPLE,
        "gmm": {**_CLUSTER_EXAMPLE, "method": "gmm"},
        "bayesian_gmm": {**_CLUSTER_EXAMPLE, "method": "bayesian_gmm"},
        "dbscan": {
            **_CLUSTER_EXAMPLE,
            "method": "dbscan",
            "cluster_count": 1,
            "noise_count": 16,
            "representative": "unweighted mean of the DBSCAN cluster means (noise excluded)",
        },
        "agglomerative": {
            **_CLUSTER_EXAMPLE,
            "method": "agglomerative",
            "representative": "unweighted mean of the Ward agglomerative cluster means",
        },
        "mean_color": {"rgb": [255, 221, 98], "lab": [89.09, -0.36, 63.31]},
        "flame_pixel_count": 44820,
        "samples_used": 2000,
        "pixels_sampled": True,
        "n_clusters": 4,
        "algorithms": ["K-Means", "GMM", "Bayesian GMM", "DBSCAN", "Agglomerative"],
        "skipped_reason": None,
    },
    "material_analysis": {
        "primary_material": "Natural Fibers",
        "similarity": 0.9268,
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
    "fire_class": {
        "class": "Class A",
        "description": "Ordinary combustibles",
        "confidence": 0.9268,
        "material": "Natural Fibers",
        "basis": "material 'Natural Fibers' -> Class A via app/fire_classes.py ...",
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
            "compound_basis": "name verbatim from flame_dataset.json; the dataset "
            "records no chemical identity, so no compound formula is asserted",
        },
        {
            "name": "CO2",
            "compound": None,
            "type": "oxygen_displacement",
            "source": "flame_dataset.json",
            "fire_class": "Class A",
            "compound_basis": "name verbatim from flame_dataset.json; ...",
        },
    ],
    "timing": {
        "total_ms": 300.4,
        "detection_ms": 28.0,
        "segmentation_ms": 41.0,
        "color_ms": 227.0,
        "material_ms": 1.4,
    },
    "ai_material_analysis": {
        "available": False,
        "error": "Gemini material analysis is disabled (GEMINI_ENABLED is not enabled).",
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

#: `POST /material-identification` - the same deterministic structures
#: `/analyze` returns, for a flame colour the client aggregated itself.  The
#: canonical name, the similarity, the alternatives and the fire class all come
#: from `flame_dataset.json` and `app/fire_classes.py`.
_MATERIAL_IDENTIFICATION_EXAMPLE: dict[str, Any] = {
    "success": True,
    "material_analysis": {
        "primary_material": "Natural Fibers",
        "similarity": 0.9268,
        "alternatives": [
            {"material": "Paper Products(Wood material)", "similarity": 0.8885},
            {"material": "Wax Materials", "similarity": 0.8885},
            {"material": "Wood Materials", "similarity": 0.8851},
        ],
        "database_notes": "Yellow flame, cotton burns fast, wool self-extinguishes, ...",
        "score_basis": "mean LAB distance to flame_dataset.json reference colours",
    },
    "primary_material": "Natural Fibers",
    "similarity": 0.9268,
    "alternatives": [
        {"material": "Paper Products(Wood material)", "similarity": 0.8885},
        {"material": "Wax Materials", "similarity": 0.8885},
        {"material": "Wood Materials", "similarity": 0.8851},
    ],
    "suppression_information": {
        "source": "flame_dataset.json",
        "material": "Natural Fibers",
        "methods": ["Water", "CO2", "Foam"],
        "database_notes": "Yellow flame, cotton burns fast, wool self-extinguishes, ...",
    },
    "fire_class": {
        "class": "Class A",
        "description": "Ordinary combustibles",
        "confidence": 0.9268,
        "material": "Natural Fibers",
        "basis": "material 'Natural Fibers' -> Class A via app/fire_classes.py ...",
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
            "compound_basis": "name verbatim from flame_dataset.json; ...",
        },
        {
            "name": "CO2",
            "compound": None,
            "type": "oxygen_displacement",
            "source": "flame_dataset.json",
            "fire_class": "Class A",
            "compound_basis": "name verbatim from flame_dataset.json; ...",
        },
    ],
}

#: Module level application used by ``uvicorn app.api:app``.
app = create_app()
