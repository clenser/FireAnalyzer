"""Flame colour analysis.

Pipeline
--------
segmentation mask -> flame pixels (RGB) -> LAB (converted **once**) ->
``StandardScaler`` -> cluster count -> K-Means / GMM / Bayesian GMM / DBSCAN /
Agglomerative -> representative flame colour -> material matching.

Restored clustering algorithms
------------------------------
The original desktop implementation clustered flame pixels with six
algorithms.  MeanShift is the only one that was intentionally removed; the
remaining five are all executed here and all appear in the API response:

==================  ==========================================================
``kmeans``          ``sklearn.cluster.KMeans`` (k-means++ initialisation)
``gmm``             ``sklearn.mixture.GaussianMixture``
``bayesian_gmm``    ``sklearn.mixture.BayesianGaussianMixture`` (variational)
``dbscan``          ``sklearn.cluster.DBSCAN`` (noise-aware, needs no ``k``)
``agglomerative``   ``sklearn.cluster.AgglomerativeClustering`` (Ward linkage)
==================  ==========================================================

MeanShift is not imported and not executed anywhere in this package.

Methodology notes
-----------------
* **Feature space** - CIELAB in the ``scikit-image`` convention (L\\* in 0-100,
  D65), which is the convention ``flame_dataset.json`` was written in.  OpenCV's
  ``COLOR_BGR2Lab`` scales L\\* to 0-255 and must not be mixed with the database.
* **StandardScaler** - the original pipeline standardised the LAB matrix before
  clustering and mapped the centroids back with ``inverse_transform``.  That is
  preserved, and it is what makes ``dbscan_eps`` and Ward linkage meaningful:
  on raw L\\* values an ``eps`` of 0.5 is a fraction of one lightness step, so
  every point would collapse into a single meaningless cluster.  Every reported
  colour is mapped back into LAB, so material matching - which lives in LAB
  space - is unaffected by the scaling.
* **Cluster count** - ``k_selection="silhouette"`` (the default, restored from
  the original pipeline) fits K-Means for every ``k`` in ``[n_clusters, k_max]``,
  scores the labelling with a silhouette value computed on at most
  ``silhouette_sample_size`` rows, and caps the winner at ``k_cap``.
  ``k_selection="fixed"`` always uses ``n_clusters``.  Either way ``n_clusters``
  is the floor and the fallback, so ``k`` is always defined.
 * **Representative colour** - derived the way the original derived it, per
   algorithm: mixture weights for GMM and Bayesian GMM, the unweighted mean of
   the K-Means centroids, and the unweighted mean of the per-cluster means for
   DBSCAN and Agglomerative.  DBSCAN noise points are excluded, and an all-noise
   result falls back to the overall mean, as in the original.
 * **Final (mean) colour** - not one global mask average.  The flame mask is
   split by Euclidean Distance Transform depth into INNER (deepest 40% of
   pixels), MIDDLE (next 35%) and OUTER (outermost 25%); RGB and LAB are
   averaged independently per zone and combined as
   ``0.40*inner + 0.35*middle + 0.25*outer``.  The zone breakdown is kept on
   ``FlameColorResult.zones`` for testing.
* **Cluster centroids / dominant cluster** - every algorithm reports its
  per-cluster centroids, sizes and weights plus the dominant cluster, so the
  dominant-cluster extraction is inspectable rather than implicit.
* **Not restored: the "base/middle/tip" split.**  The original divided the LAB
  rows into three vertical thirds and weighted them 0.5/0.3/0.2.  Those rows are
  the boolean-mask pixel listing, which carries no vertical ordering, so the
  split only fragmented the colour statistics.  It is deliberately left out and
  documented here rather than reintroduced.
* **Shared work** - the LAB conversion and the ``StandardScaler`` fit happen
  once and are shared by all five algorithms, so restoring an algorithm costs
  one estimator fit, not a repeated conversion.
* **Determinism** - every algorithm is seeded from ``random_seed``, including the
  silhouette sampling, so identical inputs give identical outputs.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np
from skimage import color as skcolor
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .schemas import FlameColor

__all__ = [
    "CLUSTERING_METHODS",
    "METHOD_LABELS",
    "K_SELECTION_MODES",
    "RepresentativeColor",
    "Centroid",
    "MethodResult",
    "FlameColorResult",
    "extract_flame_pixels",
    "subsample",
    "select_cluster_count",
    "analyze_flame_colors",
    "representative_to_schema",
]

logger = logging.getLogger(__name__)

LAB_DECIMALS = 2
WEIGHT_DECIMALS = 4

#: ``n_init`` used *inside* the k search.  The original used 3 here (it is an
#: exploratory ranking step, not the final fit), and it keeps the search cheap
#: on large masks.  The reported K-Means colour is fitted with
#: ``Settings.kmeans_n_init``.
K_SEARCH_N_INIT = 3

#: Multipliers applied to ``Settings.bayesian_gmm_reg_covar`` when the VB fit
#: hits an ill-defined covariance.  The first entry is always 1 (the configured
#: value), so a healthy fit never escalates.
_REG_COVAR_ESCALATION = (1.0, 1e2, 1e4)

#: The clustering algorithms this backend runs, in report order.  MeanShift is
#: deliberately absent - it is the single algorithm that was removed.
CLUSTERING_METHODS: tuple[str, ...] = (
    "kmeans",
    "gmm",
    "bayesian_gmm",
    "dbscan",
    "agglomerative",
)

#: Human-readable names used by the API consumer and the technical UI.
METHOD_LABELS: dict[str, str] = {
    "kmeans": "K-Means",
    "gmm": "GMM",
    "bayesian_gmm": "Bayesian GMM",
    "dbscan": "DBSCAN",
    "agglomerative": "Agglomerative",
}

#: ``k_selection`` modes accepted by :class:`app.config.Settings`.
K_SELECTION_MODES: tuple[str, ...] = ("silhouette", "fixed")

#: Zone weights for the final flame colour.  The flame mask is split by EDT
#: depth into INNER (deepest 40%), MIDDLE (next 35%) and OUTER (outermost 25%)
#: and the final RGB/LAB is the weighted combination of the three zone means.
#: These are weights for the final value, not material-class probabilities.
ZONE_WEIGHTS: tuple[float, float, float] = (0.40, 0.35, 0.25)


# ---------------------------------------------------------------------------
# Internal value objects
# ---------------------------------------------------------------------------
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
class Centroid:
    """One cluster centroid in LAB space, with the cluster's size and weight."""

    index: int
    size: int
    weight: float
    lab: np.ndarray  # float64, shape (3,)

    def rgb(self) -> np.ndarray:
        """The centroid as a ``uint8`` RGB triple."""
        rgb = skcolor.lab2rgb(self.lab.reshape(1, 3)).reshape(3)
        return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class MethodResult:
    """Outcome of one clustering algorithm, always expressed in LAB space."""

    color: RepresentativeColor
    cluster_count: int
    centroids: tuple[Centroid, ...] = ()
    noise_count: int = 0
    dominant_index: int = 0
    #: How ``color`` was derived, reported so the choice is not a black box.
    representative: str = ""
    #: True when the algorithm fell back instead of producing its own colour.
    fallback: bool = False

    @property
    def dominant(self) -> Centroid | None:
        for centroid in self.centroids:
            if centroid.index == self.dominant_index:
                return centroid
        return self.centroids[0] if self.centroids else None


@dataclass(frozen=True)
class FlameColorResult:
    """Representative colours for one image, one entry per algorithm."""

    pixel_count: int
    samples_used: int
    pixels_sampled: bool
    n_clusters: int
    methods: Mapping[str, MethodResult]
    mean: RepresentativeColor
    #: 40/35/25 EDT zone breakdown behind ``mean`` (internal/test use only).
    zones: dict[str, Any] = field(default_factory=dict)

    def get(self, method: str) -> MethodResult | None:
        """The full :class:`MethodResult` for ``method``, if it ran."""
        return self.methods.get(method)

    def _color(self, method: str) -> RepresentativeColor | None:
        result = self.methods.get(method)
        return result.color if result is not None else None

    @property
    def kmeans(self) -> RepresentativeColor | None:
        return self._color("kmeans")

    @property
    def gmm(self) -> RepresentativeColor | None:
        return self._color("gmm")

    @property
    def bayesian_gmm(self) -> RepresentativeColor | None:
        return self._color("bayesian_gmm")

    @property
    def dbscan(self) -> RepresentativeColor | None:
        return self._color("dbscan")

    @property
    def agglomerative(self) -> RepresentativeColor | None:
        return self._color("agglomerative")

    @property
    def algorithms(self) -> list[str]:
        """Names of the algorithms that produced a colour, in report order."""
        return [name for name in CLUSTERING_METHODS if name in self.methods]

    def representative_colors(self) -> list[RepresentativeColor]:
        """Every representative colour, for material matching.

        The mean colour comes last: it is the low-data fallback, so it acts as
        supporting evidence rather than as an independent vote.
        """
        colors: list[RepresentativeColor] = [
            color
            for color in (self._color(name) for name in CLUSTERING_METHODS)
            if color is not None
        ]
        return colors + [self.mean]


def representative_to_schema(color: RepresentativeColor) -> FlameColor:
    """Convert an internal colour into its JSON-ready schema."""
    return FlameColor(rgb=color.rgb_list, lab=color.lab_list)


# ---------------------------------------------------------------------------
# Pixel extraction
# ---------------------------------------------------------------------------
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


def _to_lab_color(lab: Sequence[float] | np.ndarray, method: str) -> RepresentativeColor | None:
    """Convert one LAB triple into a colour, or ``None`` if it is not finite."""
    values = np.asarray(lab, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(values)):
        return None
    rgb = skcolor.lab2rgb(values.reshape(1, 3)).reshape(3)
    rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return RepresentativeColor(method=method, rgb=rgb8, lab=values)


# ---------------------------------------------------------------------------
# Cluster count selection
# ---------------------------------------------------------------------------
def select_cluster_count(scaled: np.ndarray, settings: Settings) -> int:
    """Choose the cluster count ``k`` shared by the k-based algorithms.

    ``k_selection="fixed"`` returns ``n_clusters`` unchanged.

    ``"silhouette"`` (the default, restored from the original pipeline) fits
    K-Means for every ``k`` in ``[n_clusters, k_max]``, scores the labelling with
    a silhouette value computed on at most ``silhouette_sample_size`` rows, and
    caps the winner at ``k_cap``.  ``n_clusters`` is used whenever the search
    cannot produce a usable score, so the result is always defined.
    """
    total = int(scaled.shape[0])
    floor = max(2, int(settings.n_clusters))
    if settings.k_selection != "silhouette" or total < 4:
        return min(floor, max(2, total))

    upper = min(int(settings.k_max), total - 1)
    if upper < floor:
        return min(floor, max(2, total))

    best_k, best_score = floor, -1.0
    for k in range(floor, upper + 1):
        # silhouette_score needs strictly more samples than clusters.
        sample_size = min(int(settings.silhouette_sample_size), total)
        if sample_size <= k + 1:
            sample_size = total
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = KMeans(
                    n_clusters=k,
                    random_state=settings.random_seed,
                    n_init=K_SEARCH_N_INIT,
                    max_iter=100,
                )
                labels = model.fit_predict(scaled)
            if len(np.unique(labels)) <= 1:
                continue
            score = silhouette_score(
                scaled,
                labels,
                sample_size=None if sample_size >= total else sample_size,
                random_state=settings.random_seed,
            )
        except Exception:  # noqa: BLE001 - one bad candidate must not fail the stage
            logger.debug("Silhouette selection failed for k=%s", k, exc_info=True)
            continue
        if score > best_score:
            best_score, best_k = float(score), k

    return max(1, min(best_k, int(settings.k_cap)))


# ---------------------------------------------------------------------------
# Cluster bookkeeping
# ---------------------------------------------------------------------------
def _centroids_from_labels(
    lab_samples: np.ndarray, labels: np.ndarray, noise_label: int | None = None
) -> tuple[Centroid, ...]:
    """Per-cluster mean of the LAB samples, sorted by descending size."""
    clustered = labels if noise_label is None else labels[labels != noise_label]
    total = int(clustered.size)
    if total == 0:
        return ()

    centroids: list[Centroid] = []
    for label in np.unique(clustered):
        members = lab_samples[labels == label]
        if members.shape[0] == 0:
            continue
        centroids.append(
            Centroid(
                index=int(label),
                size=int(members.shape[0]),
                weight=round(float(members.shape[0]) / total, WEIGHT_DECIMALS),
                lab=np.mean(members, axis=0),
            )
        )
    centroids.sort(key=lambda item: (-item.size, item.index))
    return tuple(centroids)


def _centroids_from_mixture(means_lab: np.ndarray, weights: np.ndarray) -> tuple[Centroid, ...]:
    """Component means of a fitted mixture, treated as cluster centroids."""
    total = float(np.sum(weights))
    centroids = [
        Centroid(
            index=index,
            size=int(round(float(weight) * total)),
            weight=round(float(weight) / total, WEIGHT_DECIMALS) if total else 0.0,
            lab=np.asarray(means_lab[index], dtype=np.float64),
        )
        for index, weight in enumerate(weights)
    ]
    centroids.sort(key=lambda item: (-item.weight, item.index))
    return tuple(centroids)


def _dominant_index(centroids: Sequence[Centroid]) -> int:
    """The dominant cluster: heaviest for mixtures, largest for partitions."""
    return centroids[0].index if centroids else 0


def _mean_of_centroids(centroids: Sequence[Centroid]) -> np.ndarray | None:
    return None if not centroids else np.mean(np.stack([item.lab for item in centroids]), axis=0)


# ---------------------------------------------------------------------------
# The five clustering algorithms
# ---------------------------------------------------------------------------
def _kmeans_result(
    scaled: np.ndarray, lab_samples: np.ndarray, k: int, settings: Settings
) -> MethodResult | None:
    """K-Means (k-means++ init); representative = unweighted mean of the centroids."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=settings.kmeans_n_init,
                max_iter=100,
                random_state=settings.random_seed,
            )
            labels = model.fit_predict(scaled)

        centroids = _centroids_from_labels(lab_samples, labels)
        # The original averaged the K-Means centroids without weighting them.
        color = _to_lab_color(_mean_of_centroids(centroids), "kmeans")
        if color is None:
            return None
        return MethodResult(
            color=color,
            cluster_count=len(centroids),
            centroids=centroids,
            dominant_index=_dominant_index(centroids),
            representative="unweighted mean of the K-Means centroids",
        )
    except Exception:  # noqa: BLE001 - never fail the whole request on one algorithm
        logger.exception("K-Means clustering failed")
        return None


def _gmm_result(
    scaled: np.ndarray, k: int, settings: Settings, scaler: StandardScaler
) -> MethodResult | None:
    """Gaussian Mixture Model; representative = weight-weighted mean of the means."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=1,
                max_iter=settings.gmm_max_iter,
                random_state=settings.random_seed,
            )
            model.fit(scaled)
        return _mixture_result(model, scaler, "gmm", "GMM component means")
    except Exception:  # noqa: BLE001
        logger.exception("Gaussian Mixture clustering failed")
        return None


def _bayesian_gmm_result(
    scaled: np.ndarray, k: int, settings: Settings, scaler: StandardScaler
) -> MethodResult | None:
    """Variational Bayesian GMM; same aggregation as GMM but with VB inference.

    The VB fit inverts a precision matrix per component, so tightly separated or
    near-degenerate colours make the empirical covariance ill defined.  Rather
    than dropping the algorithm, the variance floor is escalated and the fit is
    retried; the floor that worked is reported in ``representative``.
    """
    for factor in _REG_COVAR_ESCALATION:
        reg_covar = settings.bayesian_gmm_reg_covar * factor
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = BayesianGaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    n_init=1,
                    max_iter=settings.bayesian_gmm_max_iter,
                    weight_concentration_prior_type="dirichlet_process",
                    reg_covar=reg_covar,
                    random_state=settings.random_seed,
                )
                model.fit(scaled)
            break
        except Exception:  # noqa: BLE001
            model = None
            logger.debug(
                "Bayesian Gaussian Mixture fit failed with reg_covar=%g, escalating",
                reg_covar,
                exc_info=True,
            )
    else:
        logger.exception("Bayesian Gaussian Mixture clustering failed for every reg_covar")
        return None

    result = _mixture_result(model, scaler, "bayesian_gmm", "variational Bayesian GMM component means")
    if result is not None and factor != 1:
        result = replace(
            result,
            representative=f"{result.representative} (reg_covar raised {factor:g}x to fit)",
        )
    return result


def _mixture_result(
    model, scaler: StandardScaler, method: str, label: str
) -> MethodResult | None:
    """Shared reporting for GMM and Bayesian GMM, mapping means back into LAB."""
    means_lab = scaler.inverse_transform(model.means_)
    weights = np.asarray(model.weights_, dtype=np.float64)
    centroids = _centroids_from_mixture(means_lab, weights)
    color = _to_lab_color(np.average(means_lab, axis=0, weights=weights), method)
    if color is None:
        return None
    return MethodResult(
        color=color,
        cluster_count=len(centroids),
        centroids=centroids,
        dominant_index=_dominant_index(centroids),
        representative=f"mixture-weight weighted mean of the {label}",
    )


def _dbscan_result(
    scaled: np.ndarray, lab_samples: np.ndarray, settings: Settings
) -> MethodResult | None:
    """DBSCAN; noise points are excluded, an all-noise run falls back to the mean."""
    total = int(scaled.shape[0])
    divisor = max(1, int(settings.dbscan_min_samples_divisor))
    min_samples = min(total, max(int(settings.dbscan_min_samples), total // divisor))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = DBSCAN(eps=settings.dbscan_eps, min_samples=min_samples).fit_predict(scaled)

        noise = int(np.count_nonzero(labels == -1))
        centroids = _centroids_from_labels(lab_samples, labels, noise_label=-1)
        if not centroids:
            # Every point was noise; the original fell back to the overall mean.
            color = _to_lab_color(np.mean(lab_samples, axis=0), "dbscan")
            if color is None:
                return None
            return MethodResult(
                color=color,
                cluster_count=0,
                centroids=(),
                noise_count=noise,
                dominant_index=0,
                representative="no dense cluster found; fell back to the mean flame colour",
                fallback=True,
            )

        color = _to_lab_color(_mean_of_centroids(centroids), "dbscan")
        if color is None:
            return None
        return MethodResult(
            color=color,
            cluster_count=len(centroids),
            centroids=centroids,
            noise_count=noise,
            dominant_index=_dominant_index(centroids),
            representative="unweighted mean of the DBSCAN cluster means (noise excluded)",
        )
    except Exception:  # noqa: BLE001
        logger.exception("DBSCAN clustering failed")
        return None


def _agglomerative_result(
    scaled: np.ndarray, lab_samples: np.ndarray, k: int, settings: Settings
) -> MethodResult | None:
    """Ward-linkage agglomerative clustering; representative = mean of cluster means."""
    total = int(scaled.shape[0])
    if total < 2:
        return None
    clusters = max(1, min(int(k), total - 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = AgglomerativeClustering(n_clusters=clusters, linkage="ward").fit_predict(
                scaled
            )

        centroids = _centroids_from_labels(lab_samples, labels)
        color = _to_lab_color(_mean_of_centroids(centroids), "agglomerative")
        if color is None:
            return None
        return MethodResult(
            color=color,
            cluster_count=len(centroids),
            centroids=centroids,
            dominant_index=_dominant_index(centroids),
            representative="unweighted mean of the Ward agglomerative cluster means",
        )
    except Exception:  # noqa: BLE001
        logger.exception("Agglomerative clustering failed")
        return None


# ---------------------------------------------------------------------------
# 40/35/25 EDT zone-weighted final colour
# ---------------------------------------------------------------------------
def _flame_depths(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance of every flame pixel from the mask boundary.

    ``cv2.distanceTransform`` measures, for every non-zero pixel, the distance
    to the nearest zero pixel - i.e. how deep inside the flame each pixel is.
    The result is returned in the same row-major order as
    :func:`extract_flame_pixels`, so ``depths[i]`` belongs to ``pixels[i]``.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return np.empty(0, dtype=np.float64)
    edt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    return edt[binary.astype(bool)]


def _zone_indices(depths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank flame pixels by EDT depth and split into INNER/MIDDLE/OUTER.

    The deepest 40% of pixels form INNER, the next 35% MIDDLE and the
    outermost 25% OUTER.  Ties (large plateaus of equal depth are common) are
    broken deterministically by the stable sort's original order.
    """
    order = np.argsort(-depths, kind="stable")
    total = int(depths.shape[0])
    n_inner = int(round(ZONE_WEIGHTS[0] * total))
    n_middle = int(round(ZONE_WEIGHTS[1] * total))
    inner = order[:n_inner]
    middle = order[n_inner : n_inner + n_middle]
    outer = order[n_inner + n_middle :]
    return inner, middle, outer


def _zone_weighted_mean_color(
    samples: np.ndarray, lab_samples: np.ndarray, depths: np.ndarray
) -> tuple[RepresentativeColor, dict[str, Any]]:
    """Final flame colour from the 40/35/25 EDT zones.

    RGB and LAB are averaged *independently per zone* from the pixels in that
    zone, then combined as ``0.40*inner + 0.35*middle + 0.25*outer``.  No global
    average is computed first and LAB is never derived from the weighted RGB.

    Edge case: when the mask is too small to fill all three zones (fewer than
    3 sampled pixels), an empty zone falls back to the global flame mean so
    the weighted combination stays defined.  No pixel values are invented.
    """
    global_rgb = np.mean(samples, axis=0)
    global_lab = np.mean(lab_samples, axis=0)
    inner_idx, middle_idx, outer_idx = _zone_indices(depths)

    zone_rgb: dict[str, np.ndarray] = {}
    zone_lab: dict[str, np.ndarray] = {}
    for name, indices in (
        ("inner", inner_idx),
        ("middle", middle_idx),
        ("outer", outer_idx),
    ):
        if indices.size:
            zone_rgb[name] = np.mean(samples[indices], axis=0)
            zone_lab[name] = np.mean(lab_samples[indices], axis=0)
        else:
            zone_rgb[name] = global_rgb
            zone_lab[name] = global_lab

    weights = ZONE_WEIGHTS
    ultimate_rgb = (
        weights[0] * zone_rgb["inner"]
        + weights[1] * zone_rgb["middle"]
        + weights[2] * zone_rgb["outer"]
    )
    ultimate_lab = (
        weights[0] * zone_lab["inner"]
        + weights[1] * zone_lab["middle"]
        + weights[2] * zone_lab["outer"]
    )
    color = RepresentativeColor(
        method="mean",
        rgb=np.clip(np.round(ultimate_rgb), 0, 255).astype(np.uint8),
        lab=ultimate_lab,
    )
    zones = {
        "inner_rgb": [round(float(v), 2) for v in zone_rgb["inner"]],
        "middle_rgb": [round(float(v), 2) for v in zone_rgb["middle"]],
        "outer_rgb": [round(float(v), 2) for v in zone_rgb["outer"]],
        "inner_lab": [round(float(v), 2) for v in zone_lab["inner"]],
        "middle_lab": [round(float(v), 2) for v in zone_lab["middle"]],
        "outer_lab": [round(float(v), 2) for v in zone_lab["outer"]],
        "ultimate_rgb": [int(v) for v in color.rgb],
        "ultimate_lab": [round(float(v), 2) for v in color.lab],
        "zone_pixel_counts": {
            "inner": int(inner_idx.size),
            "middle": int(middle_idx.size),
            "outer": int(outer_idx.size),
        },
    }
    return color, zones


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyze_flame_colors(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    settings: Settings,
) -> FlameColorResult:
    """Compute the representative flame colours for a mask.

    Every algorithm in :data:`CLUSTERING_METHODS` runs on the LAB pixels inside
    ``mask`` - never on the bounding-box area.  The mean colour is always
    returned and doubles as the low-data fallback, so the result is defined for
    any non-empty mask.
    """
    pixels = extract_flame_pixels(image_bgr, mask)
    pixel_count = int(pixels.shape[0])
    if pixel_count == 0:
        empty = RepresentativeColor("mean", np.zeros(3, np.uint8), np.zeros(3, np.float64))
        return FlameColorResult(0, 0, False, 0, {}, empty)

    depths = _flame_depths(mask)
    samples, sampled = subsample(pixels, settings.max_flame_pixels, settings.random_seed)
    # Same seed and same row count -> same indices as the pixel subsample, so
    # sample_depths stays aligned with samples.
    sample_depths, _ = subsample(depths, settings.max_flame_pixels, settings.random_seed)
    lab_samples = _to_lab(samples)  # single conversion, shared by every algorithm
    mean_color, zones = _zone_weighted_mean_color(samples, lab_samples, sample_depths)
    assert mean_color is not None  # mean of finite rows is always finite

    if pixel_count < max(settings.min_flame_pixels, settings.n_clusters):
        # Too few pixels to cluster: report the mean only.
        return FlameColorResult(
            pixel_count=pixel_count,
            samples_used=int(samples.shape[0]),
            pixels_sampled=sampled,
            n_clusters=0,
            methods={},
            mean=mean_color,
            zones=zones,
        )

    scaler = StandardScaler().fit(lab_samples)
    scaled = scaler.transform(lab_samples)
    k = select_cluster_count(scaled, settings)

    # The scaler travels with the mixture models so their component means can be
    # mapped straight back into LAB; every other algorithm aggregates LAB rows.
    runners = {
        "kmeans": lambda: _kmeans_result(scaled, lab_samples, k, settings),
        "gmm": lambda: _gmm_result(scaled, k, settings, scaler),
        "bayesian_gmm": lambda: _bayesian_gmm_result(scaled, k, settings, scaler),
        "dbscan": lambda: _dbscan_result(scaled, lab_samples, settings),
        "agglomerative": lambda: _agglomerative_result(scaled, lab_samples, k, settings),
    }

    methods: dict[str, MethodResult] = {}
    for name in CLUSTERING_METHODS:
        result = runners[name]()
        if result is None or result.color is None:
            logger.warning("Clustering algorithm %s produced no representative colour", name)
            continue
        methods[name] = result

    return FlameColorResult(
        pixel_count=pixel_count,
        samples_used=int(samples.shape[0]),
        pixels_sampled=sampled,
        n_clusters=k,
        methods=methods,
        mean=mean_color,
        zones=zones,
    )
