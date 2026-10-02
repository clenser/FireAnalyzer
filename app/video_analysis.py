"""Server-side video analysis: per-frame pipeline + deterministic aggregation.

::

    original video file (already hashed/cached by the HTTP layer)
      -> evenly spaced frame extraction (``FLAME_VIDEO_MAX_FRAMES``)
      -> every frame: FlameAnalyzer.process_frame
           (<=3 detections -> <=3 masks -> merged mask -> RGB/LAB -> LAB ranking)
      -> vision evidence (Groq, Gemini fallback) for up to
         ``FLAME_VIDEO_VISION_FRAMES`` evenly spaced frames with a flame
      -> every frame: Python fusion -> frame material + fire class
      -> Python majority/consistency vote (app.material_fusion)
      -> consolidated result + representative frames + every frame result

No LLM produces the video conclusion: vision models only contribute per-frame
evidence, and the final material and fire class come from Python.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from .analyzer import FlameAnalyzer, ProcessedFrame
from .config import Settings
from .errors import FrameExtractionError, InvalidVideoError
from .material_fusion import aggregate_video_frames, video_fire_class
from .vision_providers import safe_flame_crop, unavailable_vision, vision_evidence_for_crop

__all__ = ["MAX_FRAMES_LIMIT", "REPRESENTATIVE_FRAMES", "analyze_video_file"]

logger = logging.getLogger(__name__)

MAX_FRAMES_LIMIT = 60
REPRESENTATIVE_FRAMES = 3
_MAX_COUNTED_FRAMES = 200_000
_THUMBNAIL_MAX_SIDE = 480
_THUMBNAIL_QUALITY = 75
_VISION_WORKERS = 3


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _evenly_spaced(total: int, count: int) -> list[int]:
    """``count`` distinct indices spread over ``range(total)``, ascending."""
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [total // 2]
    step = (total - 1) / float(count - 1)
    return sorted({int(round(i * step)) for i in range(count)})


def _thumbnail(image: np.ndarray) -> dict[str, Any] | None:
    longest = max(image.shape[:2])
    if longest > _THUMBNAIL_MAX_SIDE:
        scale = _THUMBNAIL_MAX_SIDE / float(longest)
        image = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), _THUMBNAIL_QUALITY])
    if not ok:
        return None
    return {
        "encoding": "jpeg_base64",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "data": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }


def _extract_frames(path: str, max_frames: int) -> tuple[list[tuple[int, float, np.ndarray]], dict[str, Any]]:
    """Decode up to ``max_frames`` evenly spaced frames from a video file."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise InvalidVideoError()
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0 or fps > 1000:
            fps = 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if total <= 0:
            # Container without a reliable frame count: count by grabbing.
            total = 0
            while total < _MAX_COUNTED_FRAMES and capture.grab():
                total += 1
            capture.release()
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise InvalidVideoError()
        if total <= 0:
            raise FrameExtractionError()

        targets = _evenly_spaced(total, max_frames)
        wanted = set(targets)
        last = targets[-1]
        frames: list[tuple[int, float, np.ndarray]] = []
        index = 0
        while index <= last:
            if not capture.grab():
                break
            if index in wanted:
                ok, frame = capture.retrieve()
                if ok and frame is not None and frame.size:
                    timestamp = round(index / fps, 3) if fps else None
                    frames.append((index, timestamp, frame))
            index += 1
    finally:
        capture.release()

    if not frames:
        raise FrameExtractionError()
    info = {
        "frame_count": total,
        "fps": round(fps, 3) if fps else None,
        "duration_seconds": round(total / fps, 3) if fps else None,
        "width": width,
        "height": height,
        "frames_requested": len(targets),
        "frames_sampled": len(frames),
    }
    return frames, info


def _representatives(results: list[dict[str, Any]], final_material: str | None) -> list[int]:
    """Up to three frame positions: best-confidence frame in each third of the timeline.

    Frames that agree with the consolidated material are preferred; ties go to
    the earlier frame, so the choice is deterministic.
    """
    successful = [i for i, r in enumerate(results) if r.get("success")]
    agreeing = [i for i in successful if final_material and results[i].get("final_material") == final_material]
    pool = agreeing or successful
    if not pool:
        return []
    if len(pool) <= REPRESENTATIVE_FRAMES:
        return pool
    chosen: list[int] = []
    for part in range(REPRESENTATIVE_FRAMES):
        start = part * len(pool) // REPRESENTATIVE_FRAMES
        end = (part + 1) * len(pool) // REPRESENTATIVE_FRAMES
        segment = pool[start:end]
        best = max(segment, key=lambda i: (float(results[i].get("confidence") or 0.0), -i))
        chosen.append(best)
    return chosen


def analyze_video_file(
    path: str,
    analyzer: FlameAnalyzer,
    settings: Settings,
    lock: threading.Lock,
) -> dict[str, Any]:
    """Analyse a video file on disk.  Raises only for invalid/corrupt input."""
    started = time.perf_counter()
    max_frames = max(1, min(MAX_FRAMES_LIMIT, int(settings.video_max_frames)))

    stage = time.perf_counter()
    frames, info = _extract_frames(path, max_frames)
    extraction_ms = _elapsed_ms(stage)

    # 1) Deterministic per-frame processing (serialised on the inference lock).
    stage = time.perf_counter()
    processed: list[tuple[int, float | None, ProcessedFrame, dict[str, Any] | None, Any]] = []
    for index, timestamp, image in frames:
        with lock:
            frame = analyzer.process_frame(image)
        crop = safe_flame_crop(frame.image, frame.mask) if frame.success else None
        try:
            thumb = _thumbnail(image)
        except Exception:  # noqa: BLE001 - a thumbnail is cosmetic
            thumb = None
        frame.image, frame.mask = None, None  # keep memory bounded
        processed.append((index, timestamp, frame, thumb, crop))
    frames.clear()
    processing_ms = _elapsed_ms(stage)

    # 2) Vision evidence for evenly spaced frames that contain a flame.
    stage = time.perf_counter()
    with_flame = [pos for pos, item in enumerate(processed) if item[2].success]
    budget = max(0, int(settings.video_vision_frames)) if settings.vision_enabled else 0
    selected = [with_flame[i] for i in _evenly_spaced(len(with_flame), budget)]
    vision: dict[int, dict[str, Any]] = {}
    if selected:
        database = analyzer.database
        with ThreadPoolExecutor(max_workers=_VISION_WORKERS) as pool:
            futures = {
                pos: pool.submit(vision_evidence_for_crop, processed[pos][4], settings, database)
                for pos in selected
            }
            for pos, future in futures.items():
                try:
                    vision[pos] = future.result()
                except Exception:  # noqa: BLE001 - vision never breaks the video
                    logger.exception("Vision evidence failed for a video frame")
                    vision[pos] = unavailable_vision("vision evidence failed unexpectedly")
    vision_ms = _elapsed_ms(stage)

    # 3) Python fusion per frame.
    stage = time.perf_counter()
    results: list[dict[str, Any]] = []
    for pos, (index, timestamp, frame, thumb, _crop) in enumerate(processed):
        if frame.success:
            block = vision.get(pos) or unavailable_vision(
                "frame not selected for vision evidence (FLAME_VIDEO_VISION_FRAMES budget)"
            )
            try:
                result = analyzer.finalize(frame, block)
            except Exception:  # noqa: BLE001
                logger.exception("Frame fusion failed")
                result = {
                    "success": False,
                    "error": {"code": "INFERENCE_ERROR", "message": "Frame analysis failed."},
                }
        else:
            result = dict(frame.result)
        entry = {"frame_index": index, "timestamp_seconds": timestamp, **result, "thumbnail": thumb}
        results.append(entry)
    successful = [r for r in results if r.get("success")]
    aggregate = aggregate_video_frames(successful, len(results), analyzer.database)
    fire_class, agents = video_fire_class(analyzer.database, aggregate, results)
    fusion_ms = _elapsed_ms(stage)

    rep_positions = _representatives(results, aggregate["final_material"])
    providers = [r.get("vision_provider") for r in successful if r.get("vision_provider") not in (None, "none")]
    provider_counts = {name: providers.count(name) for name in ("groq", "gemini") if providers.count(name)}
    vision_provider = (
        "none" if not provider_counts else next(iter(provider_counts)) if len(provider_counts) == 1 else "mixed"
    )

    payload: dict[str, Any] = {
        "success": bool(successful),
        "analysis_type": "video",
        "video": info,
        "detection_count": sum(int(r.get("detection_count") or 0) for r in results),
        "mask_count": sum(int(r.get("mask_count") or 0) for r in results),
        "frames_with_flame": len(successful),
        "vision_provider": vision_provider,
        "vision_frames": {
            "requested": len(selected),
            "by_provider": provider_counts,
        },
        "final_material": aggregate["final_material"],
        "confidence": aggregate["confidence"],
        "confidence_percent": aggregate["confidence_percent"],
        "confidence_level": aggregate["confidence_level"],
        "uncertain": aggregate["uncertain"],
        "uncertainty_reasons": aggregate["uncertainty_reasons"],
        "leading_candidates": aggregate["leading_candidates"],
        "candidate_materials": aggregate["votes"],
        "consolidated": aggregate,
        "fire_class": fire_class,
        "extinguishing_agents": agents,
        "representative_frame_indices": [results[pos]["frame_index"] for pos in rep_positions],
        "representative_frames": [results[pos] for pos in rep_positions],
        "frames": results,
        "timing": {
            "frame_extraction_ms": extraction_ms,
            "frame_processing_ms": processing_ms,
            "vision_ms": vision_ms,
            "fusion_ms": fusion_ms,
            "total_ms": _elapsed_ms(started),
        },
        "error": None,
    }
    if not successful:
        codes = [(r.get("error") or {}).get("code") for r in results]
        if any(code in ("NO_FIRE_DETECTED", "NO_FLAME_PIXELS") for code in codes):
            payload["error"] = {
                "code": "NO_FIRE_DETECTED",
                "message": "No analysable flame was found in any sampled frame of the video.",
            }
        else:
            payload["error"] = {
                "code": next((code for code in codes if code), "INFERENCE_ERROR"),
                "message": "None of the sampled frames could be analysed.",
            }
    return payload
