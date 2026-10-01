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
        Cluster count used when ``k_selection`` is ``"fixed"``.  Also the
        fallback when silhouette selection finds no usable score, and the
        lower bound for every algorithm.
    k_selection:
        ``"silhouette"`` (default, restored from the original pipeline) searches
        ``k`` in ``[n_clusters, k_max]`` with a silhouette score and caps the
        winner at ``k_cap``.  ``"fixed"`` always uses ``n_clusters``.
    k_max, k_cap:
        Upper search bound and hard cap for silhouette selection, mirroring the
        original implementation's ``max_k=6`` / ``min(best_k, 4)``.
    silhouette_sample_size:
        Number of samples used for the silhouette score (``sample_size`` in
        scikit-learn).  Bounds the cost of the k search on large masks.
    max_flame_pixels:
        Flame pixels are subsampled to at most this many rows before
        clustering, keeping latency bounded for large masks.
    min_flame_pixels:
        Below this count the mean colour is used instead of clustering.
    bayesian_gmm_max_iter:
        Iteration cap for the variational Bayesian GMM.
    dbscan_eps:
        DBSCAN neighbourhood radius **in standardised (StandardScaler) units**,
        not CIELAB units.  This is the original value.
    dbscan_min_samples, dbscan_min_samples_divisor:
        ``min_samples = max(dbscan_min_samples, n // dbscan_min_samples_divisor)``,
        the original adaptive rule.
    emit_mask:
        Encode the segmentation mask into the response as a base64 PNG.  Turn
        off to keep the payload small when only the statistics are needed.
    mask_png_compression:
        zlib level passed to the PNG encoder (0-9).  A binary mask compresses
        extremely well, so this has little effect on the encoded size.
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
    activity_file:
        Path touched by ``POST /activity`` for the EC2 inactivity watchdog.
        Overridable with ``FLAME_ACTIVITY_FILE`` for testing.
    gemini_enabled:
        Opt in to the *secondary* Gemini material analysis (see
        :mod:`app.gemini_analysis`).  Defaults to ``False``: the deterministic
        FlameAnalyzer pipeline is unchanged unless this is explicitly enabled.
    gemini_api_key:
        API key for the Gemini API.  Read from ``GEMINI_API_KEY`` only - it is
        never hardcoded, never logged and never sent to the frontend.  When
        ``gemini_enabled`` is true and this is missing, the AI analysis reports
        ``{"available": false, ...}`` and the deterministic result is returned
        untouched.
    gemini_model:
        Gemini model used for the secondary analysis.  Defaults to
        ``gemini-3.5-flash-lite``.
    gemini_timeout_s:
        Hard timeout for one Gemini request.  A slow or hung request is abandoned
        after this many seconds and reported as an unavailable AI analysis; it
        never blocks the ``/analyze`` response indefinitely.
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
    k_selection: str = "silhouette"
    k_max: int = 6
    k_cap: int = 4
    silhouette_sample_size: int = 300
    max_flame_pixels: int = 2000
    min_flame_pixels: int = 10
    kmeans_n_init: int = 10
    gmm_max_iter: int = 100
    bayesian_gmm_max_iter: int = 50
    #: Variance floor for BayesianGMM. sklearn's default (1e-6) raises on
    #: ill-defined covariances, which happens on tightly separated flame tones;
    #: a small floor keeps the fit usable there. Raise it for noisier data.
    bayesian_gmm_reg_covar: float = 1e-3
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 3
    dbscan_min_samples_divisor: int = 50
    random_seed: int = 42

    # --- segmentation output -------------------------------------------
    emit_mask: bool = True
    mask_png_compression: int = 6

    # --- material matching ---------------------------------------------
    max_alternatives: int = 3
    similarity_distance_scale: float = 200.0

    # --- runtime -------------------------------------------------------
    device: str | None = None
    ultralytics_verbose: bool = False

    # --- HTTP API ------------------------------------------------------
    max_image_mb: int = 10
    cors_origins: tuple[str, ...] = ()
    #: File whose modification time is refreshed on every ``POST /activity``
    #: heartbeat.  The EC2 inactivity watchdog monitors this path.
    activity_file: str = "/var/run/flame-analyzer-last-activity"

    # --- Gemini (secondary AI material analysis) -------------------------
    gemini_enabled: bool = False
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_timeout_s: float = 30.0

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
            k_selection=os.environ.get("FLAME_K_SELECTION", "silhouette").strip().lower() or "silhouette",
            k_max=_env_int("FLAME_K_MAX", 6),
            k_cap=_env_int("FLAME_K_CAP", 4),
            silhouette_sample_size=_env_int("FLAME_SILHOUETTE_SAMPLES", 300),
            max_flame_pixels=_env_int("FLAME_MAX_PIXELS", 2000),
            min_flame_pixels=_env_int("FLAME_MIN_PIXELS", 10),
            kmeans_n_init=_env_int("FLAME_KMEANS_NINIT", 10),
            gmm_max_iter=_env_int("FLAME_GMM_MAX_ITER", 100),
            bayesian_gmm_max_iter=_env_int("FLAME_BGGMM_MAX_ITER", 50),
            bayesian_gmm_reg_covar=_env_float("FLAME_BGGMM_REG_COVAR", 1e-3),
            dbscan_eps=_env_float("FLAME_DBSCAN_EPS", 0.5),
            dbscan_min_samples=_env_int("FLAME_DBSCAN_MIN_SAMPLES", 3),
            dbscan_min_samples_divisor=_env_int("FLAME_DBSCAN_MIN_SAMPLES_DIV", 50),
            random_seed=_env_int("FLAME_SEED", 42),
            emit_mask=_env_bool("FLAME_EMIT_MASK", True),
            mask_png_compression=_env_int("FLAME_MASK_PNG_COMPRESSION", 6),
            max_alternatives=_env_int("FLAME_MAX_ALTERNATIVES", 3),
            similarity_distance_scale=_env_float("FLAME_SIM_SCALE", 200.0),
            device=os.environ.get("FLAME_DEVICE") or None,
            ultralytics_verbose=_env_bool("FLAME_YOLO_VERBOSE", False),
            max_image_mb=_env_int("FLAME_MAX_IMAGE_MB", 10),
            cors_origins=_env_list("FLAME_CORS_ORIGINS"),
            activity_file=os.environ.get("FLAME_ACTIVITY_FILE") or "/var/run/flame-analyzer-last-activity",
            gemini_enabled=_env_bool("GEMINI_ENABLED", False),
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
            gemini_model=os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite",
            gemini_timeout_s=_env_float("GEMINI_TIMEOUT_S", 30.0),
        )

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)  # type: ignore[arg-type]
