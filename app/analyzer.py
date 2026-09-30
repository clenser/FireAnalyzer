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

import logging
import time
from typing import Any

import torch
from ultralytics import YOLO

from .color_analysis import (
    FlameColorResult,
    RepresentativeColor,
    analyze_flame_colors,
    representative_to_schema,
)
from .config import Settings
from .detection import boxes_from_detection, detect_fire
from .errors import (
    AnalysisError,
    InferenceError,
    MissingDatabaseError,
    MissingModelError,
    NoFireDetectedError,
    NoFlamePixelsError,
    error_payload,
)
from .imaging import validate_image
from .material_matching import MaterialDatabase, load_material_database, match_material
from .schemas import (
    AnalysisResult,
    AnalysisTiming,
    ClusteringResult,
    FlameAnalysis,
)
from .segmentation import segment_fire

__all__ = ["FlameAnalyzer"]

logger = logging.getLogger(__name__)

_TIMING_DECIMALS = 2


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, _TIMING_DECIMALS)


def _clustering_schema(
    color: RepresentativeColor | None,
    method: str,
    colors: FlameColorResult,
    cluster_count: int,
) -> ClusteringResult | None:
    """Build the schema entry for one clustering method, if it succeeded."""
    if color is None:
        return None
    return ClusteringResult(
        rgb=color.rgb_list,
        lab=color.lab_list,
        method=method,
        cluster_count=cluster_count,
        samples_used=colors.samples_used,
        pixels_sampled=colors.pixels_sampled,
    )


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
    def analyze_image(self, image_bgr: Any) -> dict[str, Any]:
        """Run the full pipeline on an OpenCV BGR image array.

        Parameters
        ----------
        image_bgr:
            Image as a ``numpy.ndarray`` (grayscale, BGR or BGRA are all
            normalised internally).  Bytes should be decoded first with
            :func:`app.imaging.decode_image_bytes`.

        Returns
        -------
        dict
            A ``json.dumps``-ready result.  On failure it is
            ``{"success": False, "error": {"code": ..., "message": ...}}`` -
            this method does not raise for expected errors.
        """
        started = time.perf_counter()
        try:
            self._ensure_ready()
            image = validate_image(image_bgr)

            stage = time.perf_counter()
            detection = detect_fire(self._detection_model, image, self.settings)
            detection_ms = _elapsed_ms(stage)

            if not detection.detected:
                raise NoFireDetectedError()

            stage = time.perf_counter()
            mask, segmentation = segment_fire(
                self._segmentation_model, image, boxes_from_detection(detection), self.settings
            )
            segmentation_ms = _elapsed_ms(stage)

            if segmentation.flame_pixel_count == 0:
                raise NoFlamePixelsError()

            stage = time.perf_counter()
            colors = analyze_flame_colors(image, mask, self.settings)
            color_ms = _elapsed_ms(stage)

            stage = time.perf_counter()
            representatives = [c for c in (colors.kmeans, colors.gmm, colors.mean) if c is not None]
            material_analysis, suppression = match_material(
                representatives, self.database, self.settings
            )
            material_ms = _elapsed_ms(stage)

            return AnalysisResult(
                success=True,
                fire_detection=detection,
                segmentation=segmentation,
                flame_analysis=self._flame_analysis_schema(colors),
                material_analysis=material_analysis,
                suppression_information=suppression,
                timing=AnalysisTiming(
                    total_ms=_elapsed_ms(started),
                    detection_ms=detection_ms,
                    segmentation_ms=segmentation_ms,
                    color_ms=color_ms,
                    material_ms=material_ms,
                ),
            ).to_dict()
        except AnalysisError as exc:
            logger.info("Analysis rejected (%s): %s", exc.code, exc.message)
            return error_payload(exc.code, exc.message)
        except Exception:  # noqa: BLE001 - never surface a stack trace to clients
            logger.exception("Unexpected inference failure")
            return error_payload(InferenceError.code, InferenceError.message)

    def _ensure_ready(self) -> None:
        if self._detection_model is None:
            raise MissingModelError("The detection model has not been loaded.")
        if self._database is None:
            raise MissingDatabaseError()

    def _flame_analysis_schema(self, colors: FlameColorResult) -> FlameAnalysis:
        cluster_count = self.settings.n_clusters
        return FlameAnalysis(
            kmeans=_clustering_schema(colors.kmeans, "kmeans", colors, cluster_count),
            gmm=_clustering_schema(colors.gmm, "gmm", colors, cluster_count),
            mean_color=representative_to_schema(colors.mean),
            flame_pixel_count=colors.pixel_count,
            samples_used=colors.samples_used,
            pixels_sampled=colors.pixels_sampled,
        )
