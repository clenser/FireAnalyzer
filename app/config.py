"""Central configuration for the FlameAnalyzer inference backend.

Every tunable inference parameter lives here so that nothing is hardcoded in
several places.  Values may be overridden with environment variables, which is
what a container/Modal deployment needs.

Example
-------
>>> settings = Settings.from_env()
>>> settings.imgsz
640
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "Settings",
    "PROJECT_ROOT",
    "MODELS_DIR",
    "DATA_DIR",
    "DEFAULT_DETECTION_MODEL",
    "DEFAULT_SEGMENTATION_MODEL",
    "DEFAULT_DATASET",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
DATA_DIR: Path = PROJECT_ROOT / "data"

DEFAULT_DETECTION_MODEL: Path = MODELS_DIR / "OBJ_best.pt"
DEFAULT_SEGMENTATION_MODEL: Path = MODELS_DIR / "SEG_best.pt"
DEFAULT_DATASET: Path = DATA_DIR / "flame_dataset.json"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple of values."""
    raw = os.environ.get(name)
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable inference configuration.

    Attributes
    ----------
    detection_model_path, segmentation_model_path, dataset_path:
        Filesystem locations of the custom YOLO weights and the material
        database.  Loaded once at start-up.
    imgsz:
        Square inference size (pixels) used by both YOLO models.
    detection_conf, segmentation_conf:
        Confidence thresholds (0-1).  A box/mask below this is discarded.
    detection_class_id:
        Class index that represents fire in ``OBJ_best.pt``.
    segmentation_retina_masks:
        Ask Ultralytics to upscale masks to the original image resolution
        (handles the letterbox padding correctly).  Disable to reproduce the
        older, cheaper "resize the low-res mask" behaviour.
    n_clusters:
        Fixed cluster count for K-Means / GMM.  Kept small on purpose: the
        backend optimises for latency, not for clustering experiments.
    max_flame_pixels:
        Flame pixels are subsampled to at most this many rows before
        clustering, keeping latency bounded for large masks.
    min_flame_pixels:
        Below this count the mean colour is used instead of clustering.
    max_alternatives:
        Number of secondary materials returned next to the primary match.
    similarity_distance_scale:
        LAB distance (Euclidean) that corresponds to a similarity of 0.0.
        ``similarity = max(0, 1 - distance / scale)``.  This is a heuristic
        score, *not* a calibrated probability.
    device:
        ``None`` lets Ultralytics pick CUDA when it is available, otherwise
        CPU.  Never forces CPU on a GPU box.
    max_image_mb:
        Maximum accepted upload size in mebibytes.  Enforced by the HTTP layer
        before any inference happens.  Ignored by the CLI.
    cors_origins:
        Browser origins allowed to call the API.  Empty (the default) means CORS
        is disabled entirely, which is the right setting for a server-to-server
        deployment.  ``("*",)`` allows any origin *without* credentials.
    """

    # --- artifacts -----------------------------------------------------
    detection_model_path: Path = DEFAULT_DETECTION_MODEL
    segmentation_model_path: Path = DEFAULT_SEGMENTATION_MODEL
    dataset_path: Path = DEFAULT_DATASET

    # --- detection -----------------------------------------------------
    imgsz: int = 640
    detection_conf: float = 0.4
    detection_class_id: int = 0

    # --- segmentation --------------------------------------------------
    segmentation_conf: float = 0.4
    segmentation_retina_masks: bool = True
    fallback_to_bbox_mask: bool = True

    # --- colour analysis -----------------------------------------------
    n_clusters: int = 2
    max_flame_pixels: int = 2000
    min_flame_pixels: int = 10
    kmeans_n_init: int = 10
    gmm_max_iter: int = 100
    random_seed: int = 42

    # --- material matching ---------------------------------------------
    max_alternatives: int = 3
    similarity_distance_scale: float = 200.0

    # --- runtime -------------------------------------------------------
    device: str | None = None
    ultralytics_verbose: bool = False

    # --- HTTP API ------------------------------------------------------
    max_image_mb: int = 10
    cors_origins: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` instance from environment variables."""
        # FLAME_MODEL_DIR is a convenience for deployments that mount both
        # weights into one directory.  The explicit per-model variables win.
        model_dir = _env_path("FLAME_MODEL_DIR", MODELS_DIR)
        return cls(
            detection_model_path=_env_path("FLAME_DET_MODEL", model_dir / DEFAULT_DETECTION_MODEL.name),
            segmentation_model_path=_env_path(
                "FLAME_SEG_MODEL", model_dir / DEFAULT_SEGMENTATION_MODEL.name
            ),
            dataset_path=_env_path("FLAME_DATASET", DEFAULT_DATASET),
            imgsz=_env_int("FLAME_IMGSZ", 640),
            detection_conf=_env_float("FLAME_DET_CONF", 0.4),
            detection_class_id=_env_int("FLAME_DET_CLASS", 0),
            segmentation_conf=_env_float("FLAME_SEG_CONF", 0.4),
            segmentation_retina_masks=_env_bool("FLAME_SEG_RETINA", True),
            fallback_to_bbox_mask=_env_bool("FLAME_BBOX_FALLBACK", True),
            n_clusters=_env_int("FLAME_N_CLUSTERS", 2),
            max_flame_pixels=_env_int("FLAME_MAX_PIXELS", 2000),
            min_flame_pixels=_env_int("FLAME_MIN_PIXELS", 10),
            kmeans_n_init=_env_int("FLAME_KMEANS_NINIT", 10),
            gmm_max_iter=_env_int("FLAME_GMM_MAX_ITER", 100),
            random_seed=_env_int("FLAME_SEED", 42),
            max_alternatives=_env_int("FLAME_MAX_ALTERNATIVES", 3),
            similarity_distance_scale=_env_float("FLAME_SIM_SCALE", 200.0),
            device=os.environ.get("FLAME_DEVICE") or None,
            ultralytics_verbose=_env_bool("FLAME_YOLO_VERBOSE", False),
            max_image_mb=_env_int("FLAME_MAX_IMAGE_MB", 10),
            cors_origins=_env_list("FLAME_CORS_ORIGINS"),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)  # type: ignore[arg-type]
