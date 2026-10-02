"""Error types for the inference backend.

Every failure that an API client may legitimately trigger is represented by an
:class:`AnalysisError` carrying a stable machine-readable ``code`` and a short,
human readable ``message``.  Tracebacks are logged internally and never leak
into the response payload.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "AnalysisError",
    "InvalidImageError",
    "MissingUploadError",
    "ImageTooLargeError",
    "NoFireDetectedError",
    "NoFlamePixelsError",
    "MissingModelError",
    "MissingDatabaseError",
    "AnalyzerUnavailableError",
    "ValidationError",
    "InferenceError",
    "DetectionFailedError",
    "InvalidVideoError",
    "VideoTooLargeError",
    "UnsupportedMediaError",
    "FrameExtractionError",
    "error_payload",
    "ERROR_CODES",
]

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Base class for all recoverable, client-visible analysis failures."""

    code: str = "INFERENCE_ERROR"
    message: str = "The analysis could not be completed."

    def __init__(self, message: str | None = None, code: str | None = None) -> None:
        self.message = message or self.__class__.message
        if code is not None:
            self.code = code
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible error body."""
        return {"code": self.code, "message": self.message}


class InvalidImageError(AnalysisError):
    """The supplied bytes/array are not a decodable 3-channel image."""

    code = "INVALID_IMAGE"
    message = "The supplied data could not be decoded as a valid image."


class MissingUploadError(AnalysisError):
    """The request did not contain the expected ``image`` upload field."""

    code = "MISSING_IMAGE"
    message = "No image file was provided in the 'image' field."


class ImageTooLargeError(AnalysisError):
    """The upload exceeded the configured maximum size."""

    code = "IMAGE_TOO_LARGE"
    message = "The uploaded image exceeds the maximum allowed size."


class NoFireDetectedError(AnalysisError):
    """The detection model found no fire region above the confidence threshold."""

    code = "NO_FIRE_DETECTED"
    message = "No fire region was detected in the supplied image."


class NoFlamePixelsError(AnalysisError):
    """A fire region was found but the mask contained no usable pixels."""

    code = "NO_FLAME_PIXELS"
    message = "A fire region was detected but no flame pixels could be extracted."


class MissingModelError(AnalysisError):
    """A required model weight file is missing or unreadable."""

    code = "MISSING_MODEL"
    message = "A required detection/segmentation model could not be loaded."


class MissingDatabaseError(AnalysisError):
    """The material database is missing or malformed."""

    code = "MISSING_DATABASE"
    message = "The material database could not be loaded."


class AnalyzerUnavailableError(AnalysisError):
    """The analyzer has not finished initialising (models not loaded yet)."""

    code = "SERVICE_UNAVAILABLE"
    message = "The analysis service is not ready to accept requests."


class ValidationError(AnalysisError):
    """The request itself was malformed (e.g. the ``image`` field is absent)."""

    code = "VALIDATION_ERROR"
    message = "The request is missing or has an invalid 'image' field."


class InferenceError(AnalysisError):
    """Any unexpected failure inside the ML pipeline."""

    code = "INFERENCE_ERROR"
    message = "An unexpected error occurred during inference."


class DetectionFailedError(AnalysisError):
    """The detection model raised while processing a valid image."""

    code = "DETECTION_FAILED"
    message = "Fire detection failed for the supplied image."


class InvalidVideoError(AnalysisError):
    """The upload is empty, corrupt or not a decodable video."""

    code = "INVALID_VIDEO"
    message = "The uploaded data could not be decoded as a valid video."


class VideoTooLargeError(AnalysisError):
    """The video upload exceeded the configured maximum size."""

    code = "VIDEO_TOO_LARGE"
    message = "The uploaded video exceeds the maximum allowed size."


class UnsupportedMediaError(AnalysisError):
    """The upload's media type is not supported by this endpoint."""

    code = "UNSUPPORTED_MEDIA"
    message = "The uploaded file type is not supported by this endpoint."


class FrameExtractionError(AnalysisError):
    """The video opened but no frame could be extracted from it."""

    code = "FRAME_EXTRACTION_FAILED"
    message = "No frames could be extracted from the uploaded video."


#: Every error code this backend is able to emit, for API documentation.
ERROR_CODES: tuple[str, ...] = (
    MissingUploadError.code,
    InvalidImageError.code,
    ImageTooLargeError.code,
    NoFireDetectedError.code,
    NoFlamePixelsError.code,
    MissingModelError.code,
    MissingDatabaseError.code,
    AnalyzerUnavailableError.code,
    ValidationError.code,
    InferenceError.code,
    DetectionFailedError.code,
    InvalidVideoError.code,
    VideoTooLargeError.code,
    UnsupportedMediaError.code,
    FrameExtractionError.code,
)


def error_payload(code: str, message: str) -> dict[str, Any]:
    """Build the canonical ``{"success": False, "error": {...}}`` response."""
    return {"success": False, "error": {"code": code, "message": message}}
