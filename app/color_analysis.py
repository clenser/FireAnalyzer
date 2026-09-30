"""Flame colour analysis.

Pipeline
--------
segmentation mask -> flame pixels (RGB) -> LAB (converted **once**) ->
K-Means / GMM representative colour -> material matching.

Design notes
------------
* The BGR -> RGB swap is performed on the *masked subset* of pixels rather than
  on the whole image, which is orders of magnitude cheaper.
* The RGB -> LAB conversion happens exactly once; K-Means and the GMM share the
  same LAB matrix.
* LAB values follow the ``scikit-image`` convention (L* in 0-100, D65), which is
  the convention used by ``flame_dataset.json``.  OpenCV's ``COLOR_BGR2Lab``
  scales L* to 0-255 and must not be mixed with the database.
* Clustering uses a fixed, small ``k`` and a seeded RNG, so results are
  deterministic for identical inputs.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from skimage import color as skcolor
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .config import Settings
from .schemas import FlameColor

__all__ = [
    "RepresentativeColor",
    "FlameColorResult",
    "extract_flame_pixels",
    "subsample",
    "analyze_flame_colors",
    "representative_to_schema",
]

logger = logging.getLogger(__name__)

LAB_DECIMALS = 2


@dataclass(frozen=True)
class RepresentativeColor:
    """A representative flame colour kept as NumPy arrays (internal use)."""

    method: str
    rgb: np.ndarray  # uint8, shape (3,)
    lab: np.ndarray  # float64, shape (3,)

    @property
    def lab_list(self) -> list[float]:
        return [round(float(v), LAB_DECIMALS) for v in self.lab]

    @property
    def rgb_list(self) -> list[int]:
        return [int(v) for v in self.rgb]


@dataclass(frozen=True)
class FlameColorResult:
    """Representative colours for one image."""

    pixel_count: int
    samples_used: int
    pixels_sampled: bool
    kmeans: RepresentativeColor | None
    gmm: RepresentativeColor | None
    mean: RepresentativeColor


def representative_to_schema(color: RepresentativeColor) -> FlameColor:
    """Convert an internal colour into its JSON-ready schema."""
    return FlameColor(rgb=color.rgb_list, lab=color.lab_list)


def extract_flame_pixels(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the masked flame pixels as a ``uint8`` RGB array of shape ``(N, 3)``.

    The channel order is swapped *after* masking, so only the relevant pixels
    are touched.
    """
    binary = mask > 0
    if not binary.any():
        return np.empty((0, 3), dtype=np.uint8)
    return np.ascontiguousarray(image_bgr[binary][:, ::-1])


def subsample(pixels: np.ndarray, max_pixels: int, seed: int) -> tuple[np.ndarray, bool]:
    """Deterministically cap the number of rows used for clustering."""
    total = int(pixels.shape[0])
    if max_pixels <= 0 or total <= max_pixels:
        return pixels, False
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=max_pixels, replace=False))
    return pixels[indices], True


def _to_lab(rgb_pixels: np.ndarray) -> np.ndarray:
    """Convert ``uint8`` RGB pixels to float64 LAB (scikit-image convention)."""
    normalised = rgb_pixels.astype(np.float64) / 255.0
    return skcolor.rgb2lab(normalised)


def _lab_to_color(lab: np.ndarray, method: str) -> RepresentativeColor:
    """Convert a LAB triple into an internal representative colour."""
    rgb = skcolor.lab2rgb(lab.reshape(1, 3)).reshape(3)
    rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return RepresentativeColor(method=method, rgb=rgb8, lab=np.asarray(lab, dtype=np.float64))


def _kmeans_color(lab_samples: np.ndarray, settings: Settings) -> RepresentativeColor | None:
    """Unweighted mean of the K-Means cluster centres (legacy behaviour)."""
    k = settings.n_clusters
    if lab_samples.shape[0] < k:
        return None
    try:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=settings.kmeans_n_init,
            max_iter=100,
            random_state=settings.random_seed,
        )
        # Uniform flame regions legitimately contain fewer distinct colours
        # than k; that is not a failure.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            centers = model.fit(lab_samples).cluster_centers_
        return _lab_to_color(np.mean(centers, axis=0), "kmeans")
    except Exception:  # noqa: BLE001 - never fail the whole request on one algorithm
        logger.exception("K-Means clustering failed; falling back to mean colour")
        return None


def _gmm_color(lab_samples: np.ndarray, settings: Settings) -> RepresentativeColor | None:
    """Mixture-weight weighted mean of the GMM component means."""
    k = settings.n_clusters
    if lab_samples.shape[0] < k:
        return None
    try:
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=1,
            max_iter=settings.gmm_max_iter,
            random_state=settings.random_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(lab_samples)
        lab = np.average(model.means_, axis=0, weights=model.weights_)
        return _lab_to_color(lab, "gmm")
    except Exception:  # noqa: BLE001
        logger.exception("Gaussian Mixture clustering failed; falling back to mean colour")
        return None


def analyze_flame_colors(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    settings: Settings,
) -> FlameColorResult:
    """Compute the representative flame colours for a mask.

    Returns K-Means and GMM colours when enough flame pixels are available and
    always returns a mean colour, which doubles as the low-data fallback.
    """
    pixels = extract_flame_pixels(image_bgr, mask)
    pixel_count = int(pixels.shape[0])
    if pixel_count == 0:
        empty = RepresentativeColor("mean", np.zeros(3, np.uint8), np.zeros(3, np.float64))
        return FlameColorResult(0, 0, False, None, None, empty)

    samples, sampled = subsample(pixels, settings.max_flame_pixels, settings.random_seed)
    lab_samples = _to_lab(samples)  # single conversion, shared by both algorithms
    mean_lab = np.mean(lab_samples, axis=0)
    mean_color = _lab_to_color(mean_lab, "mean")

    kmeans = gmm = None
    if pixel_count >= max(settings.min_flame_pixels, settings.n_clusters):
        kmeans = _kmeans_color(lab_samples, settings)
        gmm = _gmm_color(lab_samples, settings)

    return FlameColorResult(
        pixel_count=pixel_count,
        samples_used=int(samples.shape[0]),
        pixels_sampled=sampled,
        kmeans=kmeans,
        gmm=gmm,
        mean=mean_color,
    )
