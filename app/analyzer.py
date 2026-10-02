"""``FlameAnalyzer`` - the reusable, side-effect-free inference backend.

The class owns two long-lived YOLO models and the material database, and
exposes a single entry point::

    analyzer = FlameAnalyzer()
    result = analyzer.analyze_image(image_bgr)   # -> JSON-serialisable dict

Design constraints
------------------
* No GUI, no web server, no tunnels, no network calls, no disk I/O, no LLM.
* Models are loaded exactly once and kept in memory (CUDA is used when
  available).
* Public results contain only JSON-native types.
* A single instance is intended for sequential use.  Ultralytics predictors
  hold mutable state, so concurrent calls on the *same* instance are not
  supported; create one instance per worker/container instead.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import torch
from ultralytics import YOLO

from .color_analysis import (
    METHOD_LABELS,
    FlameColorResult,
    MethodResult,
    RepresentativeColor,
    analyze_flame_colors,
    representative_to_schema,
)
from .config import Settings
from .detection import detect_fires
from .errors import (
    AnalysisError,
    DetectionFailedError,
    InferenceError,
    MissingDatabaseError,
    MissingModelError,
    NoFireDetectedError,
    NoFlamePixelsError,
    error_payload,
)
from .fire_classes import classify_decision
from .imaging import validate_image
from .material_fusion import deterministic_evidence, fuse_material
from .material_matching import MaterialDatabase, load_material_database, rank_materials
from .schemas import (
    AnalysisResult,
    ClusteringResult,
    ClusterCentroid,
    FireDetection,
    FlameAnalysis,
    FlameColor,
    SegmentationSummary,
    jsonable,
)
from .segmentation import segment_detections
from .vision_providers import unavailable_vision

__all__ = ["FlameAnalyzer", "ProcessedFrame"]

logger = logging.getLogger(__name__)

_TIMING_DECIMALS = 2


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, _TIMING_DECIMALS)


def _centroid_schema(result: MethodResult, centroid) -> ClusterCentroid:
    """Convert one internal cluster centroid into its schema entry."""
    return ClusterCentroid(
        index=centroid.index,
        size=centroid.size,
        weight=centroid.weight,
        rgb=[int(v) for v in centroid.rgb()],
        lab=[round(float(v), 2) for v in centroid.lab],
    )


def _clustering_schema(
    result: MethodResult | None,
    method: str,
    colors: FlameColorResult,
) -> ClusteringResult | None:
    """Build the schema entry for one clustering method, if it succeeded."""
    if result is None or result.color is None:
        return None
    dominant = result.dominant
    return ClusteringResult(
        rgb=result.color.rgb_list,
        lab=result.color.lab_list,
        method=method,
        cluster_count=result.cluster_count,
        samples_used=colors.samples_used,
        pixels_sampled=colors.pixels_sampled,
        dominant_cluster=result.dominant_index,
        dominant_color=(
            FlameColor(
                rgb=[int(v) for v in dominant.rgb()],
                lab=[round(float(v), 2) for v in dominant.lab],
            )
            if dominant is not None
            else FlameColor()
        ),
        centroids=[_centroid_schema(result, item) for item in result.centroids],
        noise_count=result.noise_count,
        representative=result.representative,
        fallback=result.fallback,
    )


@dataclass
class ProcessedFrame:
    """Output of :meth:`FlameAnalyzer.process_frame` (one image or video frame).

    ``image`` and ``mask`` (the merged flame mask) are kept in memory only so
    the vision evidence can be built from the same flame region; they are never
    serialised.
    """

    success: bool
    result: dict[str, Any]
    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    deterministic: dict[str, Any] | None = None
    timing: dict[str, float] = field(default_factory=dict)


def _detections_payload(
    detections: list[FireDetection], segmentation: SegmentationSummary | None
) -> list[dict[str, Any]]:
    """Every detection with its own box and the mask that was derived for it."""
    masks = {item.get("detection_index"): item for item in (segmentation.detection_masks if segmentation else [])}
    payload: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        box = jsonable(dataclasses.asdict(detection.bounding_box)) if detection.bounding_box else None
        mask_info = masks.get(index, {})
        payload.append(
            {
                "index": index,
                "confidence": detection.confidence,
                "bounding_box": box,
                "bbox": box,
                "mask_source": mask_info.get("mask_source", "none"),
                "segmentation_confidence": mask_info.get("segmentation_confidence"),
                "mask_pixel_count": int(mask_info.get("mask_pixel_count", 0)),
            }
        )
    return payload


def _notes(database: MaterialDatabase, material: str | None) -> str | None:
    entry = database.get(material) if material else None
    return entry.notes if entry else None


def _legacy_material_analysis(
    decision: dict[str, Any], deterministic: dict[str, Any], database: MaterialDatabase
) -> dict[str, Any]:
    """The pre-fusion ``material_analysis`` shape, now carrying the fused decision.

    ``primary_material`` is the Python-fused final material (``None`` when
    uncertain); ``similarity`` stays the LAB similarity of that material so the
    field keeps its original meaning.
    """
    similarity = {row["material"]: row["similarity"] for row in deterministic.get("ranking", [])}
    final = decision["final_material"]
    return {
        "primary_material": final,
        "similarity": similarity.get(final, 0.0) if final else 0.0,
        "alternatives": [
            {"material": row["material"], "similarity": similarity.get(row["material"], 0.0)}
            for row in decision["candidate_materials"]
            if row["material"] != final
        ][:3],
        "database_notes": _notes(database, final),
        "score_basis": "Python fusion of CIELAB colour matching and vision evidence (app/material_fusion.py)",
        "uncertain": decision["uncertain"],
    }


class FlameAnalyzer:
    """Loads the models once and runs the detection -> segmentation -> colour
    -> material pipeline.

    Parameters
    ----------
    settings:
        Configuration to use.  Defaults to :meth:`Settings.from_env`.
    autoload:
        Load models and database during construction.  Set to ``False`` to
        defer loading (useful in tests) and call :meth:`load` explicitly.
    """

    def __init__(self, settings: Settings | None = None, *, autoload: bool = True) -> None:
        self.settings = settings or Settings.from_env()
        self._detection_model: YOLO | None = None
        self._segmentation_model: YOLO | None = None
        self._database: MaterialDatabase | None = None
        if autoload:
            self.load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> "FlameAnalyzer":
        """Load both models and the database (idempotent)."""
        self.load_models()
        self.load_database()
        return self

    def load_models(self) -> None:
        """Load ``OBJ_best.pt`` and ``SEG_best.pt`` into memory, once.

        The detection model is mandatory; the segmentation model is optional and
        degrades to the bounding-box mask when absent.
        """
        if self._detection_model is not None:
            return

        det_path = self.settings.detection_model_path
        if not det_path.is_file():
            raise MissingModelError(f"Detection model not found: {det_path}")
        self._detection_model = self._load_yolo(det_path, "detection")

        seg_path = self.settings.segmentation_model_path
        if seg_path.is_file():
            try:
                self._segmentation_model = self._load_yolo(seg_path, "segmentation")
            except AnalysisError:
                logger.exception(
                    "Segmentation model could not be loaded; bounding-box masks will be used"
                )
                self._segmentation_model = None
        else:
            logger.warning(
                "Segmentation model not found at %s; bounding-box masks will be used", seg_path
            )

    def _load_yolo(self, path: Any, label: str) -> YOLO:
        try:
            model = YOLO(str(path))
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean API error
            raise MissingModelError(f"{label.capitalize()} model could not be loaded: {exc}") from exc
        logger.info("Loaded %s model %s", label, path.name)
        return model

    def load_database(self) -> None:
        """Load ``flame_dataset.json`` into memory, once."""
        if self._database is not None:
            return
        self._database = load_material_database(self.settings.dataset_path)

    @property
    def database(self) -> MaterialDatabase:
        """The loaded material database."""
        if self._database is None:
            raise MissingDatabaseError("The material database has not been loaded yet.")
        return self._database

    @property
    def models_loaded(self) -> bool:
        return self._detection_model is not None

    @property
    def segmentation_available(self) -> bool:
        return self._segmentation_model is not None

    @property
    def device(self) -> str:
        """Device the models run on.

        Ultralytics resolves the device on the first prediction (it is not
        known at load time), so this reports the real predictor device when
        available and otherwise falls back to what will be selected.
        """
        predictor = getattr(self._detection_model, "predictor", None)
        resolved = getattr(predictor, "device", None) if predictor is not None else None
        if resolved:
            return str(resolved)
        if self.settings.device:
            return self.settings.device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def process_frame(self, image_bgr: Any) -> ProcessedFrame:
        """Detection -> per-detection masks -> merged mask -> colours -> LAB ranking.

        Shared by image analysis and video-frame analysis.  Holds no network
        I/O, so it is the only part that needs the per-process inference lock.
        Never raises for expected failures: ``ProcessedFrame.success`` is
        ``False`` and ``result`` is a structured error payload.
        """
        started = time.perf_counter()
        timing: dict[str, float] = {}
        detections: list[FireDetection] = []
        segmentation: SegmentationSummary | None = None
        try:
            self._ensure_ready()
            image = validate_image(image_bgr)

            stage = time.perf_counter()
            try:
                detections = detect_fires(self._detection_model, image, self.settings)
            except Exception as exc:  # noqa: BLE001 - surfaced as a clean error
                logger.exception("Fire detection failed")
                raise DetectionFailedError() from exc
            timing["detection_ms"] = _elapsed_ms(stage)
            if not detections:
                raise NoFireDetectedError()

            stage = time.perf_counter()
            boxes = [list(d.bounding_box.as_tuple()) for d in detections if d.bounding_box]
            mask, segmentation, _per_detection = segment_detections(
                self._segmentation_model, image, boxes, self.settings
            )
            timing["segmentation_ms"] = _elapsed_ms(stage)
            if segmentation.flame_pixel_count == 0:
                raise NoFlamePixelsError()

            stage = time.perf_counter()
            colors = analyze_flame_colors(image, mask, self.settings)
            timing["color_ms"] = _elapsed_ms(stage)

            stage = time.perf_counter()
            ranking = rank_materials(colors.representative_colors(), self.database, self.settings)
            best = detections[0]
            deterministic = deterministic_evidence(
                ranking,
                self.settings,
                flame_pixel_count=segmentation.flame_pixel_count,
                mask_area_ratio=segmentation.mask_area_ratio,
                real_mask=bool(segmentation.available),
                detection_confidence=best.confidence,
                detection_count=len(detections),
                mask_count=segmentation.mask_count,
                mean_rgb=colors.mean.rgb_list,
                mean_lab=colors.mean.lab_list,
            )
            timing["material_ms"] = _elapsed_ms(stage)

            core = AnalysisResult(
                success=True,
                fire_detection=best,
                segmentation=segmentation,
                flame_analysis=self._flame_analysis_schema(colors),
            ).to_dict()
            result: dict[str, Any] = {
                "success": True,
                "detection_count": len(detections),
                "mask_count": segmentation.mask_count,
                "fire_detection": core["fire_detection"],
                "detections": _detections_payload(detections, segmentation),
                "segmentation": core["segmentation"],
                "flame_analysis": core["flame_analysis"],
                "deterministic_evidence": deterministic,
            }
            timing["processing_ms"] = _elapsed_ms(started)
            return ProcessedFrame(True, result, image, mask, deterministic, timing)
        except AnalysisError as exc:
            logger.info("Analysis rejected (%s): %s", exc.code, exc.message)
            payload = error_payload(exc.code, exc.message)
        except Exception:  # noqa: BLE001 - never surface a stack trace to clients
            logger.exception("Unexpected inference failure")
            payload = error_payload(InferenceError.code, InferenceError.message)

        payload["detection_count"] = len(detections)
        payload["mask_count"] = segmentation.mask_count if segmentation is not None else 0
        payload["detections"] = _detections_payload(detections, segmentation)
        timing["processing_ms"] = _elapsed_ms(started)
        payload["timing"] = dict(timing)
        return ProcessedFrame(False, payload, None, None, None, timing)

    def finalize(
        self,
        frame: ProcessedFrame,
        vision: dict[str, Any] | None = None,
        extra_timing: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Python fusion of the evidence, then the deterministic fire class.

        ``vision`` is the block from
        :func:`app.vision_providers.collect_vision_evidence` (or ``None`` when
        no vision evidence was requested).  The returned dict is the complete,
        JSON-ready analysis result.
        """
        if not frame.success or frame.deterministic is None:
            return frame.result
        stage = time.perf_counter()
        vision_block = vision if vision is not None else unavailable_vision("vision evidence was not requested")
        decision = fuse_material(frame.deterministic, vision_block, self.database)
        fire_class, agents = classify_decision(
            self.database,
            decision["final_material"],
            decision["leading_candidates"],
            decision["confidence"],
        )
        fusion_ms = _elapsed_ms(stage)

        result = dict(frame.result)
        result["vision_evidence"] = vision_block
        result["vision_provider"] = vision_block.get("vision_provider", "none")
        result.update(decision)
        result["material_analysis"] = _legacy_material_analysis(decision, frame.deterministic, self.database)
        result["suppression_information"] = {
            "source": self.database.source,
            "material": decision["final_material"],
            "methods": [agent.name for agent in agents],
            "database_notes": _notes(self.database, decision["final_material"]),
        }
        fire_payload = jsonable(dataclasses.asdict(fire_class))
        fire_payload["class"] = fire_payload.pop("class_")
        result["fire_class"] = fire_payload
        result["extinguishing_agents"] = [jsonable(dataclasses.asdict(agent)) for agent in agents]
        timing = dict(frame.timing)
        timing.update(extra_timing or {})
        timing["fusion_ms"] = fusion_ms
        timing["total_ms"] = round(sum(v for k, v in timing.items() if k != "processing_ms" and k.endswith("_ms")), 2)
        result["timing"] = timing
        result["error"] = None
        return result

    def analyze_image(self, image_bgr: Any) -> dict[str, Any]:
        """Run the full pipeline on an OpenCV BGR image array (no network calls).

        Deterministic evidence only; the HTTP layer adds vision evidence through
        :meth:`process_frame` + :meth:`finalize`.  On failure the result is
        ``{"success": False, "error": {"code": ..., "message": ...}, ...}`` -
        this method does not raise for expected errors.
        """
        frame = self.process_frame(image_bgr)
        try:
            return self.finalize(frame)
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected failure while fusing the material evidence")
            return error_payload(InferenceError.code, InferenceError.message)

    def _ensure_ready(self) -> None:
        if self._detection_model is None:
            raise MissingModelError("The detection model has not been loaded.")
        if self._database is None:
            raise MissingDatabaseError()

    def _flame_analysis_schema(self, colors: FlameColorResult) -> FlameAnalysis:
        return FlameAnalysis(
            kmeans=_clustering_schema(colors.get("kmeans"), "kmeans", colors),
            gmm=_clustering_schema(colors.get("gmm"), "gmm", colors),
            bayesian_gmm=_clustering_schema(colors.get("bayesian_gmm"), "bayesian_gmm", colors),
            dbscan=_clustering_schema(colors.get("dbscan"), "dbscan", colors),
            agglomerative=_clustering_schema(colors.get("agglomerative"), "agglomerative", colors),
            mean_color=representative_to_schema(colors.mean),
            flame_pixel_count=colors.pixel_count,
            samples_used=colors.samples_used,
            pixels_sampled=colors.pixels_sampled,
            n_clusters=colors.n_clusters,
            algorithms=[
                METHOD_LABELS[name] for name in colors.algorithms
            ],
            skipped_reason=(
                None
                if colors.algorithms
                else (
                    f"fewer than {self.settings.min_flame_pixels} flame pixels; "
                    "only the mean flame colour is reported"
                )
            ),
        )
