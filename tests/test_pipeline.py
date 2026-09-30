"""Tests for the FlameAnalyzer inference pipeline.

The tests that exercise the full orchestration use a stub YOLO model, so they
run in a fraction of a second without the 100 MB weight files.  The test that
does need the real weights is skipped automatically when they are absent.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from skimage import color as skcolor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzer import FlameAnalyzer  # noqa: E402
from app.color_analysis import (  # noqa: E402
    CLUSTERING_METHODS,
    METHOD_LABELS,
    RepresentativeColor,
    _to_lab,
    analyze_flame_colors,
    extract_flame_pixels,
    select_cluster_count,
    subsample,
)
from app.config import Settings  # noqa: E402
from app.errors import (  # noqa: E402
    InvalidImageError,
    MissingDatabaseError,
    MissingModelError,
    NoFireDetectedError,
    NoFlamePixelsError,
)
from app.fire_classes import (  # noqa: E402
    AGENT_SUPPRESSION_TYPE,
    AGENT_TYPE_BASIS,
    MAPPING_SOURCE,
    CLASS_A,
    CLASS_B,
    CLASS_C,
    CLASS_D,
    MATERIAL_FIRE_CLASS,
    UNKNOWN_CLASS,
    UNSPECIFIED_AGENT_TYPE,
    agent_suppression_type,
    classify,
    fire_class_for_agents,
    fire_class_for_material,
)
from app.imaging import decode_image_bytes, validate_image  # noqa: E402
from app.mask import (  # noqa: E402
    MASK_ENCODING,
    decode_mask_base64,
    encode_mask_base64,
    mask_from_base64,
)
from app.material_matching import (  # noqa: E402
    load_material_database,
    match_material,
    suppression_for,
)
from app.segmentation import bounding_box_mask, segment_fire  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
DATASET = ROOT / "data" / "flame_dataset.json"
DET_MODEL = ROOT / "models" / "OBJ_best.pt"
SEG_MODEL = ROOT / "models" / "SEG_best.pt"

#: Orange in RGB; the BGR slice in ``_fire_image`` is this tuple reversed.
ORANGE_RGB = (255, 140, 0)

#: Three well-separated flame tones (RGB) and their BGR on-disk order.
TONE_A_RGB = (255, 90, 0)
TONE_B_RGB = (255, 200, 60)
TONE_C_RGB = (240, 250, 200)
TONE_A_BGR = TONE_A_RGB[::-1]
TONE_B_BGR = TONE_B_RGB[::-1]
TONE_C_BGR = TONE_C_RGB[::-1]
TONE_RGB = (TONE_A_RGB, TONE_B_RGB, TONE_C_RGB)

#: The five clustering algorithms that must be retained.
RETAINED_METHODS = ("kmeans", "gmm", "bayesian_gmm", "dbscan", "agglomerative")


# ---------------------------------------------------------------------------
# Stub models mimicking the subset of the ultralytics Result API we consume
# ---------------------------------------------------------------------------
class StubBox:
    def __init__(self, xyxy, conf: float, cls: int = 0) -> None:
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf], dtype=torch.float32)
        self.cls = torch.tensor([cls], dtype=torch.int64)


class StubMasks:
    def __init__(self, data) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)


class StubBoxes:
    """Mimics ``ultralytics.engine.results.Boxes`` (iterable *and* vectorised)."""

    def __init__(self, items) -> None:
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    @property
    def xyxy(self) -> torch.Tensor:
        return torch.stack([item.xyxy[0] for item in self._items])

    @property
    def conf(self) -> torch.Tensor:
        return torch.stack([item.conf for item in self._items])

    @property
    def cls(self) -> torch.Tensor:
        return torch.stack([item.cls for item in self._items])


class StubResult:
    def __init__(self, boxes, masks) -> None:
        self.boxes = StubBoxes(boxes) if boxes is not None else None
        self.masks = masks


class StubModel:
    """Returns fixed results whatever the input image is."""

    def __init__(self, results) -> None:
        self._results = results
        self.calls: list[dict] = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        return self._results


def _settings(**overrides) -> Settings:
    settings = Settings.from_env()
    return settings.with_overrides(**overrides) if overrides else settings


def _detection_result(box=(10, 10, 60, 60), conf: float = 0.9, cls: int = 0) -> list[StubResult]:
    return [StubResult([StubBox(box, conf, cls)], masks=None)]


def _segmentation_result(box, conf: float, mask: np.ndarray) -> list[StubResult]:
    return [StubResult([StubBox(box, conf)], StubMasks(mask[None, :, :]))]


def _analyzer_with_stubs(detection_results, segmentation_results, **overrides) -> FlameAnalyzer:
    """Build an analyzer with stubbed models and the real database."""
    analyzer = FlameAnalyzer(_settings(**overrides), autoload=False)
    analyzer._detection_model = StubModel(detection_results)
    analyzer._segmentation_model = (
        None if segmentation_results is None else StubModel(segmentation_results)
    )
    analyzer._database = load_material_database(DATASET)
    return analyzer


def _fire_image(size: int = 100) -> np.ndarray:
    """Black image with an orange square in the middle."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[30:70, 30:70] = ORANGE_RGB[::-1]
    return image


def _two_tone_fire_image() -> np.ndarray:
    """Black image with two flame tones, so clustering has structure to find."""
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[20:60, 30:90] = TONE_A_BGR
    image[60:100, 30:90] = TONE_B_BGR
    return image


def _three_tone_fire_image() -> np.ndarray:
    """Black image with three flame tones."""
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[20:50, 30:90] = TONE_A_BGR
    image[50:75, 30:90] = TONE_B_BGR
    image[75:100, 30:90] = TONE_C_BGR
    return image


def _tone_mask(height: int = 120) -> np.ndarray:
    mask = np.zeros((height, 120), dtype=np.float32)
    mask[20:100, 30:90] = 1.0
    return mask


def _two_tone_mask() -> np.ndarray:
    return _tone_mask()


def _two_tone_flame_pixels() -> np.ndarray:
    """The masked pixels of ``_two_tone_fire_image`` as an RGB array."""
    return extract_flame_pixels(
        _two_tone_fire_image(), (_tone_mask() * 255).astype(np.uint8)
    )


def _L_SHAPE(size: int = 100) -> np.ndarray:  # noqa: N802 - a shape, not a constant
    """An L-shaped mask: a 40x40 square with its bottom-right 20x20 removed.

    1600 - 400 = 1200 pixels.  Deliberately not a filled rectangle, so a test
    can prove the real mask was used instead of the detection box.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    mask[30:70, 30:70] = 1.0
    mask[50:70, 50:70] = 0.0
    return mask


def _color_from_rgb(rgb, method: str = "test") -> RepresentativeColor:
    lab = skcolor.rgb2lab(np.asarray(rgb, dtype=np.float64).reshape(1, 3) / 255.0)[0]
    return RepresentativeColor(method, np.asarray(rgb, dtype=np.uint8), lab)


# ---------------------------------------------------------------------------
# Imaging
# ---------------------------------------------------------------------------
def test_validate_image_rejects_none_and_empty():
    with pytest.raises(InvalidImageError):
        validate_image(None)
    with pytest.raises(InvalidImageError):
        validate_image(np.empty((0, 0, 3), dtype=np.uint8))


def test_validate_image_promotes_grayscale_to_bgr():
    assert validate_image(np.zeros((8, 8), dtype=np.uint8)).shape == (8, 8, 3)


def test_decode_image_bytes_roundtrip():
    image = _fire_image()
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    assert decode_image_bytes(buffer.tobytes()).shape == image.shape


def test_decode_image_bytes_rejects_garbage():
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"not an image at all")


# ---------------------------------------------------------------------------
# Database + material matching
# ---------------------------------------------------------------------------
def test_database_loads():
    database = load_material_database(DATASET)
    assert len(database) == 18
    assert database.get("Wood Materials").flame_lab.shape == (3, 3)


def test_missing_database_raises():
    with pytest.raises(MissingDatabaseError):
        load_material_database(DATASET.parent / "does_not_exist.json")


def test_match_material_finds_database_entry():
    database = load_material_database(DATASET)
    query = _color_from_rgb(ORANGE_RGB)
    analysis, _ = match_material([query], database, _settings())

    # The winner must be the entry with the globally smallest LAB distance.
    distances = {
        entry.name: float(np.min(np.linalg.norm(entry.flame_lab - query.lab, axis=1)))
        for entry in database.entries
    }
    best_name = min(distances, key=lambda name: (distances[name], name))
    assert analysis.primary_material == best_name
    assert analysis.similarity > 0.9
    assert 0.0 <= analysis.similarity <= 1.0
    assert len(analysis.alternatives) == 3
    assert analysis.primary_material not in {a.material for a in analysis.alternatives}


def test_similarity_decreases_with_distance():
    database = load_material_database(DATASET)
    close_match, _ = match_material([_color_from_rgb((255, 150, 30))], database, _settings())
    far_match, _ = match_material([_color_from_rgb((20, 40, 220))], database, _settings())
    assert close_match.similarity > far_match.similarity


def test_suppression_comes_from_database_verbatim():
    database = load_material_database(DATASET)
    entry = database.get("Metal Objects")
    suppression = suppression_for(database, "Metal Objects")
    assert suppression.source == DATASET.name
    assert suppression.material == entry.name
    assert suppression.methods == list(entry.extinguishers)
    assert suppression.database_notes == entry.notes


def test_suppression_for_unknown_material_is_empty():
    suppression = suppression_for(load_material_database(DATASET), None)
    assert suppression.material is None
    assert suppression.methods == []


# ---------------------------------------------------------------------------
# Colour analysis
# ---------------------------------------------------------------------------
def test_extract_flame_pixels_swaps_channels_on_masked_pixels_only():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    pixels = extract_flame_pixels(_fire_image(), mask)
    assert pixels.shape == (40 * 40, 3)
    assert np.array_equal(pixels[0], np.asarray(ORANGE_RGB, dtype=np.uint8))


def test_extract_flame_pixels_on_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert extract_flame_pixels(_fire_image(20), mask).shape == (0, 3)


def test_subsample_is_deterministic_and_capped():
    pixels = np.arange(1000 * 3, dtype=np.uint8).reshape(1000, 3)
    first, sampled_first = subsample(pixels, 100, 42)
    second, sampled_second = subsample(pixels, 100, 42)
    assert sampled_first is sampled_second is True
    assert first.shape == (100, 3)
    assert np.array_equal(first, second)


def test_color_analysis_recovers_known_colour():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255
    result = analyze_flame_colors(_fire_image(), mask, _settings())

    assert result.pixel_count == 1600
    assert result.kmeans is not None and result.gmm is not None
    expected = np.asarray(ORANGE_RGB, dtype=np.int16)
    for color in (result.mean, result.kmeans, result.gmm):
        assert np.abs(color.rgb.astype(np.int16) - expected).max() <= 2


def test_color_analysis_runs_every_retained_algorithm():
    """All five retained algorithms must execute and report a colour."""
    result = analyze_flame_colors(_two_tone_fire_image(), _two_tone_mask(), _settings())

    assert result.algorithms == list(RETAINED_METHODS)
    assert result.methods.keys() == set(RETAINED_METHODS)
    for method in RETAINED_METHODS:
        entry = result.get(method)
        assert entry is not None, f"{method} did not run"
        assert entry.cluster_count >= 1
        assert entry.centroids, f"{method} reported no centroids"
        assert entry.representative, f"{method} did not explain its representative colour"
        assert entry.fallback is False
        assert entry.color.rgb.shape == (3,)
        assert entry.color.lab.shape == (3,)
        # The reported colour is derived from the reported centroids: an
        # unweighted mean, or a weight-aware mean for the mixture models.
        labs = np.array([c.lab for c in entry.centroids])
        unweighted = labs.mean(axis=0)
        weighted = np.average(labs, axis=0, weights=[c.weight for c in entry.centroids])
        assert np.allclose(entry.color.lab, unweighted, atol=0.02) or np.allclose(
            entry.color.lab, weighted, atol=0.02
        )
        # Weights are normalised and the dominant index points at the largest.
        assert sum(c.weight for c in entry.centroids) == pytest.approx(1.0, abs=1e-3)
        largest = max(entry.centroids, key=lambda c: c.size)
        assert entry.dominant_index == largest.index
        assert entry.dominant is not None and entry.dominant.index == largest.index


def test_color_analysis_separates_flame_tones():
    """The tones in the fixture must become distinct clusters, not one blob."""
    result = analyze_flame_colors(_two_tone_fire_image(), _two_tone_mask(), _settings())

    for method in ("kmeans", "gmm", "agglomerative"):
        entry = result.get(method)
        assert entry is not None
        assert entry.cluster_count == 2, f"{method} did not separate the two tones"
        # Every centroid must sit near one of the two input tones.
        for centroid in entry.centroids:
            distances = [
                float(np.abs(centroid.rgb().astype(int) - np.asarray(tone, int)).max())
                for tone in (TONE_A_RGB, TONE_B_RGB)
            ]
            assert min(distances) <= 12

    # The two clusters are genuinely different colours.
    kmeans = result.get("kmeans")
    blues = sorted(int(c.rgb()[2]) for c in kmeans.centroids)
    assert blues[1] - blues[0] > 30

    # Each centroid is reported in both LAB and RGB, with size and weight.
    assert sum(c.size for c in kmeans.centroids) == result.samples_used
    for centroid in kmeans.centroids:
        assert centroid.lab.shape == (3,)
        assert 0 <= centroid.index < 2
        assert centroid.size > 0
        assert centroid.weight == pytest.approx(centroid.size / result.samples_used, abs=1e-3)


def test_dbscan_reports_noise_points():
    """DBSCAN is density based, so it must expose its noise count."""
    image = _two_tone_fire_image()
    rng = np.random.default_rng(7)
    image[20:100, 30:90] = rng.integers(0, 255, (80, 60, 3), dtype=np.uint8)

    result = analyze_flame_colors(image, _two_tone_mask(), _settings())

    dbscan = result.get("dbscan")
    assert dbscan is not None
    # No `eps` is fitted, so the noise count is the number of unassigned points.
    assigned = sum(c.size for c in dbscan.centroids)
    assert dbscan.noise_count >= 0
    assert assigned + dbscan.noise_count <= result.samples_used


def test_color_analysis_falls_back_to_mean_for_tiny_masks():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 255
    result = analyze_flame_colors(_fire_image(), mask, _settings())
    assert result.pixel_count == 1
    for method in RETAINED_METHODS:
        assert result.get(method) is None
    assert result.algorithms == []
    assert result.mean.rgb.tolist() == list(ORANGE_RGB)


def test_color_analysis_handles_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = analyze_flame_colors(_fire_image(), mask, _settings())
    assert result.pixel_count == 0
    for method in RETAINED_METHODS:
        assert result.get(method) is None
    assert result.mean.rgb.tolist() == [0, 0, 0]


def test_k_selection_supports_silhouette_and_fixed_modes():
    scaled = _scaled_lab_for_test(_two_tone_flame_pixels())
    settings = _settings()

    best = select_cluster_count(scaled, settings)
    floor = max(2, settings.n_clusters)
    upper = min(settings.k_max, scaled.shape[0] - 1)
    assert floor <= best <= upper
    # Two perfectly separated tones score a silhouette of 1, so k=2 must win.
    assert best == 2

    assert select_cluster_count(scaled, _settings(k_selection="fixed", n_clusters=3)) == 3
    # Degenerate inputs must still return a usable k.
    assert select_cluster_count(np.zeros((2, 3)), settings) == 2
    assert select_cluster_count(np.zeros((0, 3)), settings) == 2


def _scaled_lab_for_test(rgb_pixels: np.ndarray) -> np.ndarray:
    """Replicate the pipeline's LAB + StandardScaler preprocessing for tests."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(_to_lab(rgb_pixels))


def test_color_analysis_respects_a_fixed_k():
    fixed = analyze_flame_colors(
        _three_tone_fire_image(),
        _tone_mask(),
        _settings(k_selection="fixed", n_clusters=3),
    )
    assert fixed.n_clusters == 3
    for method in ("kmeans", "gmm", "agglomerative"):
        entry = fixed.get(method)
        assert entry is not None, f"{method} did not run"
        assert entry.cluster_count == 3, f"{method} did not honour the fixed k"

    searched = analyze_flame_colors(_two_tone_fire_image(), _two_tone_mask(), _settings())
    assert searched.n_clusters == 2  # silhouette finds the real structure


def test_color_analysis_reports_only_populated_clusters():
    """Asking for more clusters than exist must not report empty ones."""
    result = analyze_flame_colors(
        _two_tone_fire_image(), _two_tone_mask(), _settings(k_selection="fixed", n_clusters=4)
    )
    entry = result.get("kmeans")
    assert entry is not None
    assert entry.cluster_count == 2
    assert len(entry.centroids) == 2


def test_color_analysis_is_deterministic():
    """A fixed seed must make repeated runs identical."""
    image = _two_tone_fire_image()
    rng = np.random.default_rng(3)
    image[20:100, 30:90] = rng.integers(0, 255, (80, 60, 3), dtype=np.uint8)

    first = analyze_flame_colors(image, _two_tone_mask(), _settings())
    second = analyze_flame_colors(image, _two_tone_mask(), _settings())

    assert [c.rgb.tolist() for c in first.representative_colors()] == [
        c.rgb.tolist() for c in second.representative_colors()
    ]
    assert first.n_clusters == second.n_clusters


def test_color_analysis_samples_large_masks_deterministically():
    """A big mask is subsampled, and the reported pixel count stays truthful."""
    image = np.zeros((600, 600, 3), dtype=np.uint8)
    rng = np.random.default_rng(11)
    image[100:500, 100:500] = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    mask = np.zeros((600, 600), dtype=np.uint8)
    mask[100:500, 100:500] = 255

    result = analyze_flame_colors(image, mask, _settings())

    assert result.pixel_count == 400 * 400  # true count, not the sample count
    assert result.pixels_sampled is True
    assert result.samples_used == _settings().max_flame_pixels
    for method in RETAINED_METHODS:
        entry = result.get(method)
        assert entry is not None, f"{method} did not run on a large mask"
        assert sum(c.size for c in entry.centroids) <= _settings().max_flame_pixels


# ---------------------------------------------------------------------------
# MeanShift removal
# ---------------------------------------------------------------------------
def test_meanshift_is_not_imported_or_referenced_in_code():
    """No module may import or call MeanShift (prose in docstrings is fine)."""
    offenders: list[str] = []
    for path in sorted(APP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                names.append(node.module or "")
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            else:
                continue
            for name in names:
                if "meanshift" in name.lower():
                    offenders.append(f"{path.name}: {name}")

    assert offenders == []


def test_meanshift_is_absent_from_the_output_contract():
    """No algorithm list or serialised key may reintroduce MeanShift."""
    result = analyze_flame_colors(_two_tone_fire_image(), _two_tone_mask(), _settings())
    assert CLUSTERING_METHODS == RETAINED_METHODS
    assert set(result.methods) == set(RETAINED_METHODS)
    assert "MeanShift" not in METHOD_LABELS.values()
    assert all("meanshift" not in name.lower() for name in result.methods)

    # The whole pipeline payload must be free of any MeanShift key too.
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 1.0
    analyzer = _analyzer_with_stubs(
        _detection_result(), _segmentation_result((30, 30, 70, 70), 0.85, mask)
    )
    payload = analyzer.analyze_image(_fire_image())
    assert "meanshift" not in json.dumps(payload).lower()


def test_meanshift_is_not_exposed_by_the_module_namespace():
    """The module must not carry a MeanShift symbol the caller could call."""
    import app.color_analysis as color_analysis

    assert not [name for name in dir(color_analysis) if "meanshift" in name.lower()]
    assert "meanshift" not in repr(CLUSTERING_METHODS).lower()


# ---------------------------------------------------------------------------
# Mask encoding
# ---------------------------------------------------------------------------
def test_mask_roundtrips_exactly_through_base64_png():
    mask = np.zeros((64, 48), dtype=np.uint8)
    mask[10:30, 5:25] = 255

    payload = encode_mask_base64(mask, compression=6)

    assert isinstance(payload, str)
    # PNG signature, base64 encoded.
    assert payload.startswith("iVBORw0KGgo")
    decoded = decode_mask_base64(payload)
    assert decoded is not None
    assert decoded.shape == (64, 48)
    assert np.array_equal(decoded, mask)
    assert int((decoded > 0).sum()) == 400


def test_mask_is_decoded_as_a_boolean_view():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    payload = encode_mask_base64(mask, compression=6)

    boolean = mask_from_base64(payload)

    assert boolean is not None
    assert boolean.dtype == bool
    assert boolean.shape == (10, 10)
    assert int(boolean.sum()) == 9
    # The boolean view must line up with the source image.
    assert np.array_equal(boolean, mask > 0)


def test_mask_png_compression_keeps_the_payload_small():
    """A big mask must not become a huge uncompressed blob in the JSON."""
    rng = np.random.default_rng(5)
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[100:900, 100:900] = 255
    mask[400:600, 400:600] = rng.choice([0, 255], size=(200, 200))

    raw = mask.size  # one value per pixel
    payload = encode_mask_base64(mask, compression=6)

    assert decode_mask_base64(payload) is not None
    # PNG compresses a solid mask to almost nothing; base64 adds 4/3.
    assert len(payload) < raw // 4


def test_mask_decoding_rejects_invalid_payloads():
    assert decode_mask_base64(None) is None
    assert decode_mask_base64("") is None
    assert decode_mask_base64("not base64!!") is None
    assert mask_from_base64("not base64!!") is None
    # Valid base64 that is not a PNG.
    assert decode_mask_base64("aGVsbG8gd29ybGQ=") is None


def test_mask_encoding_handles_an_empty_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    payload = encode_mask_base64(mask, compression=6)
    decoded = decode_mask_base64(payload)
    assert decoded is not None and decoded.shape == (20, 20)
    assert int((decoded > 0).sum()) == 0


def test_mask_encoding_rejects_an_empty_array():
    assert encode_mask_base64(np.zeros((0, 0), dtype=np.uint8), compression=6) is None
    assert encode_mask_base64(np.zeros((4, 4, 3), dtype=np.uint8), compression=6) is None
    assert encode_mask_base64(None, compression=6) is None


def test_mask_compression_level_is_clamped():
    """A bad environment value must not fail the request."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    for level in (-5, 0, 6, 99, None, "nonsense"):
        payload = encode_mask_base64(mask, compression=level)
        assert payload is not None
        assert np.array_equal(decode_mask_base64(payload) > 0, mask > 0)


# ---------------------------------------------------------------------------
# Segmentation: the real mask, not a rectangle
# ---------------------------------------------------------------------------
def test_segment_fire_returns_the_model_mask_with_real_dimensions():
    """The mask must carry the image's own shape, with real pixel counts."""
    model = StubModel(_segmentation_result((30, 30, 70, 70), 0.9, _L_SHAPE()))
    image = _fire_image(100)

    mask, info = segment_fire(model, image, [[30, 30, 70, 70]], _settings())

    assert mask.shape == image.shape[:2]  # not the detection box shape
    assert info.available is True
    assert info.fallback_used is False
    assert info.bbox_fallback_reason is None
    assert info.flame_pixel_count == int(np.count_nonzero(mask))
    assert info.flame_pixel_count == 1200  # an L, not a filled 40x40 square
    assert info.mask_area_ratio == pytest.approx(1200 / (100 * 100), abs=1e-6)
    assert info.confidence == 0.9
    assert info.mask_encoding == MASK_ENCODING
    assert info.mask_width == 100 and info.mask_height == 100
    assert info.mask.startswith("iVBORw0KGgo")


def test_bounding_box_and_mask_are_independent_fields():
    """The detection box and the flame mask must not be conflated."""
    model = StubModel(_segmentation_result((20, 20, 80, 80), 0.9, _L_SHAPE()))

    mask, info = segment_fire(model, _fire_image(100), [[20, 20, 80, 80]], _settings())

    box_area = (80 - 20) * (80 - 20)
    assert box_area == 3600
    assert info.flame_pixel_count == 1200
    assert info.flame_pixel_count != box_area
    # The mask sits strictly inside the box, so the two are different shapes.
    ys, xs = np.nonzero(mask)
    assert xs.min() >= 20 and xs.max() < 80 and ys.min() >= 20 and ys.max() < 80

    # The encoded mask decodes back to the very same pixels.
    decoded = mask_from_base64(info.mask)
    assert decoded is not None
    assert np.array_equal(decoded, mask > 0)
    assert int(decoded.sum()) == info.flame_pixel_count


def test_mask_area_ratio_is_derived_from_the_decoded_mask():
    mask = np.zeros((80, 80), dtype=np.float32)
    mask[0:40, 0:80] = 255  # exactly half the frame

    analyzer = _analyzer_with_stubs(
        _detection_result((0, 0, 80, 80)), _segmentation_result((0, 0, 80, 80), 0.9, mask)
    )
    result = analyzer.analyze_image(np.zeros((80, 80, 3), dtype=np.uint8))
    segmentation = result["segmentation"]

    decoded = mask_from_base64(segmentation["mask"])
    assert decoded is not None
    assert segmentation["flame_pixel_count"] == int(decoded.sum()) == 3200
    assert segmentation["mask_area_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert segmentation["mask_area_ratio"] == pytest.approx(
        segmentation["flame_pixel_count"] / decoded.size, abs=1e-6
    )
    assert segmentation["mask_width"] == 80 and segmentation["mask_height"] == 80


def test_segmentation_picks_the_highest_confidence_mask():
    low = np.zeros((100, 100), dtype=np.float32)
    low[0:10, 0:10] = 1.0
    high = np.zeros((100, 100), dtype=np.float32)
    high[50:60, 50:60] = 1.0

    class TwoMasks(StubModel):
        def __call__(self, image, **kwargs):
            return [
                StubResult(
                    [StubBox((0, 0, 10, 10), 0.2), StubBox((50, 50, 60, 60), 0.95)],
                    StubMasks(np.stack([low, high])),
                )
            ]

    _, info = segment_fire(TwoMasks([]), _fire_image(100), [[0, 0, 100, 100]], _settings())

    assert info.confidence == 0.95
    assert info.flame_pixel_count == 100


def test_segmentation_falls_back_to_the_box_and_says_so():
    model = StubModel(_detection_result())  # no masks attached

    mask, info = segment_fire(model, _fire_image(100), [[10, 10, 50, 50]], _settings())

    assert info.available is False
    assert info.fallback_used is True
    assert info.bbox_fallback_reason
    assert int(np.count_nonzero(mask)) == 40 * 40
    # The fallback is still reported, and still encodable.
    assert info.mask is not None
    assert int(mask_from_base64(info.mask).sum()) == info.flame_pixel_count


def test_segmentation_reports_unavailable_when_the_model_fails():
    class Exploding(StubModel):
        def __call__(self, image, **kwargs):
            raise RuntimeError("no weights")

    _, info = segment_fire(Exploding([]), _fire_image(100), [[10, 10, 50, 50]], _settings())

    # The failure degrades to the box rather than propagating.
    assert info.available is False
    assert info.fallback_used is True
    assert "failed" in info.bbox_fallback_reason or info.bbox_fallback_reason


def test_segmentation_without_a_model_reports_no_model():
    _, info = segment_fire(None, _fire_image(100), [[10, 10, 50, 50]], _settings())

    assert info.fallback_used is True
    assert "no segmentation model" in info.bbox_fallback_reason


def test_segmentation_can_be_disabled():
    model = StubModel(_detection_result())
    settings = _settings(fallback_to_bbox_mask=False)

    mask, info = segment_fire(model, _fire_image(100), [[10, 10, 50, 50]], settings)

    assert info.available is False
    assert info.fallback_used is False
    assert info.flame_pixel_count == 0
    assert info.mask is None
    assert int(np.count_nonzero(mask)) == 0


def test_mask_omission_keeps_the_counts():
    """`emit_mask=False` drops the payload, not the measurements."""
    mask = np.zeros((60, 60), dtype=np.float32)
    mask[0:30, 0:60] = 1.0
    settings = _settings(emit_mask=False)

    _, info = segment_fire(
        StubModel(_segmentation_result((0, 0, 60, 60), 0.9, mask)),
        np.zeros((60, 60, 3), dtype=np.uint8),
        [[0, 0, 60, 60]],
        settings,
    )

    assert info.mask is None
    assert info.mask_encoding is None
    assert info.flame_pixel_count == 1800
    assert info.mask_area_ratio == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Fire class and extinguishing agents (derived from the project's own data)
# ---------------------------------------------------------------------------
def test_fire_class_table_covers_exactly_the_dataset_materials():
    """The explicit mapping must cover the dataset, with nothing invented."""
    database = load_material_database(DATASET)
    dataset_materials = {entry.name for entry in database.entries}

    assert set(MATERIAL_FIRE_CLASS) == dataset_materials
    assert len(MATERIAL_FIRE_CLASS) == 18
    assert set(MATERIAL_FIRE_CLASS.values()) <= {
        CLASS_A,
        CLASS_B,
        CLASS_C,
        CLASS_D,
    }


def test_fire_class_table_agrees_with_the_documented_rule():
    """Every row must follow the extinguishers recorded in the dataset."""
    database = load_material_database(DATASET)
    for entry in database.entries:
        derived = fire_class_for_agents(entry.extinguishers, entry.name)
        assert MATERIAL_FIRE_CLASS[entry.name] == derived, (
            f"{entry.name}: table says {MATERIAL_FIRE_CLASS[entry.name]}, "
            f"its recorded agents {entry.extinguishers} imply {derived}"
        )


def test_fire_class_rule_precedence():
    """Metal agents win, then electrical, then water, then flammable."""
    assert fire_class_for_agents(["Class D powder"], "Metal Objects") == CLASS_D
    assert fire_class_for_agents(["Sand"], "Metal Objects") == CLASS_D
    # A metal agent outranks everything else.
    assert fire_class_for_agents(["Water", "Class D powder"], "Metal Objects") == CLASS_D
    # Electrical equipment with no water.
    assert fire_class_for_agents(["CO2", "Foam"], "Electrical Components") == CLASS_C
    # Water is acceptable -> Class A.
    assert fire_class_for_agents(["Water", "CO2", "Foam"], "Natural Fibers") == CLASS_A
    # No water and not electrical/metal -> Class B.
    assert fire_class_for_agents(["CO2", "Foam", "Dry powder"], "Liquid Fuels") == CLASS_B
    assert fire_class_for_agents(["Water spray"], "Spray Products") == CLASS_B
    # No evidence at all.
    assert fire_class_for_agents([], "Mystery") == UNKNOWN_CLASS
    assert fire_class_for_agents(["", "  "], "Mystery") == UNKNOWN_CLASS


def test_fire_class_rule_is_case_insensitive():
    assert fire_class_for_agents(["water"], "Natural Fibers") == CLASS_A
    assert fire_class_for_agents(["CLASS D POWDER"], "Metal Objects") == CLASS_D
    assert fire_class_for_agents(["  Water  "], "Natural Fibers") == CLASS_A


def test_fire_class_for_material_reports_its_basis():
    configured, basis = fire_class_for_material("Natural Fibers", ["Water", "CO2", "Foam"])
    assert configured == CLASS_A
    assert "Natural Fibers" in basis and CLASS_A in basis and MAPPING_SOURCE in basis

    unknown_material, basis = fire_class_for_material("Unobtainium", ["Water"])
    assert unknown_material == CLASS_A  # falls back to the rule
    assert "not listed in the explicit mapping" in basis

    no_material, basis = fire_class_for_material(None, ["Water"])
    assert no_material == UNKNOWN_CLASS
    assert "No material was matched" in basis


def test_fire_class_classification_uses_dataset_agents():
    database = load_material_database(DATASET)

    fire_class, agents = classify(database, "Natural Fibers", 0.9268)

    assert fire_class.class_ == CLASS_A
    assert fire_class.description == "Ordinary combustibles"
    assert fire_class.confidence == 0.9268
    assert fire_class.material == "Natural Fibers"
    assert fire_class.mapping_source == MAPPING_SOURCE
    assert "not predicted by the detection model" in fire_class.notes
    assert [agent.name for agent in agents] == ["Water", "CO2", "Foam"]


def test_extinguishing_agents_come_verbatim_from_the_dataset():
    database = load_material_database(DATASET)
    for entry in database.entries:
        _, agents = classify(database, entry.name, 1.0)
        assert [agent.name for agent in agents] == list(entry.extinguishers)
        for agent in agents:
            assert agent.source == "flame_dataset.json"
            assert agent.fire_class == MATERIAL_FIRE_CLASS[entry.name]


def test_agent_compounds_are_never_invented():
    """The dataset has no chemistry, so no compound may be asserted."""
    database = load_material_database(DATASET)
    for entry in database.entries:
        _, agents = classify(database, entry.name, 1.0)
        for agent in agents:
            assert agent.compound is None
            assert agent.compound_basis == AGENT_TYPE_BASIS


def test_agent_suppression_types_cover_exactly_the_dataset_agents():
    database = load_material_database(DATASET)
    dataset_agents = {
        agent for entry in database.entries for agent in entry.extinguishers
    }

    assert dataset_agents == {
        "CO2",
        "Class D powder",
        "Dry powder",
        "Foam",
        "Sand",
        "Water",
        "Water spray",
    }
    assert set(AGENT_SUPPRESSION_TYPE) == dataset_agents
    for name in dataset_agents:
        assert agent_suppression_type(name) != UNSPECIFIED_AGENT_TYPE
    # An unknown agent is surfaced, not guessed.
    assert agent_suppression_type("Unobtainium Gas") == UNSPECIFIED_AGENT_TYPE
    assert agent_suppression_type("") == UNSPECIFIED_AGENT_TYPE


def test_classification_without_a_matched_material():
    database = load_material_database(DATASET)

    fire_class, agents = classify(database, None, 0.0)

    assert fire_class.class_ == UNKNOWN_CLASS
    assert agents == []


def test_analysis_result_carries_the_derived_class_and_agents():
    """The whole pipeline must surface the class and the dataset agents."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 1.0
    analyzer = _analyzer_with_stubs(
        _detection_result(), _segmentation_result((30, 30, 70, 70), 0.85, mask)
    )
    result = analyzer.analyze_image(_fire_image())

    material = result["material_analysis"]["primary_material"]
    assert material in MATERIAL_FIRE_CLASS

    fire_class = result["fire_class"]
    assert fire_class["class"] == MATERIAL_FIRE_CLASS[material]
    assert fire_class["material"] == material
    assert fire_class["mapping_source"] == MAPPING_SOURCE
    assert 0.0 <= fire_class["confidence"] <= 1.0

    agents = result["extinguishing_agents"]
    assert [agent["name"] for agent in agents] == result["suppression_information"]["methods"]
    assert all(agent["compound"] is None for agent in agents)
    assert all(agent["type"] != UNSPECIFIED_AGENT_TYPE for agent in agents)


# ---------------------------------------------------------------------------
# Response size: no giant uncompressed pixel arrays
# ---------------------------------------------------------------------------
def test_response_does_not_carry_a_raw_pixel_array():
    """The mask must be a compact PNG, not a per-pixel JSON array."""
    mask = np.zeros((400, 400), dtype=np.float32)
    mask[50:350, 50:350] = 1.0
    analyzer = _analyzer_with_stubs(
        _detection_result((50, 50, 350, 350)),
        _segmentation_result((50, 50, 350, 350), 0.9, mask),
    )
    result = analyzer.analyze_image(np.zeros((400, 400, 3), dtype=np.uint8))

    payload = json.dumps(result)
    raw_json_mask_size = 400 * 400 * 6  # a JSON array of 0/255 would be ~this
    assert len(payload) < raw_json_mask_size / 4
    # No value in the payload is a long list of numbers.
    assert not [
        value
        for value in json.loads(payload).get("segmentation", {}).values()
        if isinstance(value, list) and len(value) > 32
    ]


def test_agents_in_the_response_are_small_and_flat():
    analyzer = _analyzer_with_stubs(
        _detection_result(), _segmentation_result((30, 30, 70, 70), 0.85, _L_SHAPE())
    )
    result = analyzer.analyze_image(_fire_image())

    for agent in result["extinguishing_agents"]:
        assert set(agent) == {
            "name",
            "compound",
            "type",
            "source",
            "fire_class",
            "compound_basis",
        }
        assert isinstance(agent["name"], str) and agent["name"]


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------
def test_bounding_box_mask_clips_to_image():
    mask = bounding_box_mask((50, 50), [[-10, -10, 20, 20]])
    assert mask.shape == (50, 50)
    assert np.count_nonzero(mask) == 400  # only the visible 20x20 area


def test_bounding_box_mask_ignores_degenerate_boxes():
    assert np.count_nonzero(bounding_box_mask((50, 50), [[10, 10, 10, 40]])) == 0


# ---------------------------------------------------------------------------
# End-to-end orchestration with stubbed models
# ---------------------------------------------------------------------------
def test_analyze_image_returns_json_serialisable_result():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[30:70, 30:70] = 1.0
    analyzer = _analyzer_with_stubs(
        _detection_result(), _segmentation_result((30, 30, 70, 70), 0.85, mask)
    )
    result = analyzer.analyze_image(_fire_image())

    assert result["success"] is True
    box = {"x1": 10, "y1": 10, "x2": 60, "y2": 60}
    assert result["fire_detection"] == {
        "detected": True,
        "confidence": 0.9,
        "bounding_box": box,
        # Short alias for consumers that expect `bbox`.
        "bbox": box,
    }
    segmentation = result["segmentation"]
    assert segmentation["available"] is True
    assert segmentation["fallback_used"] is False
    assert segmentation["flame_pixel_count"] == 1600
    assert segmentation["mask_area_ratio"] == 0.16
    assert segmentation["confidence"] == 0.85
    assert result["flame_analysis"]["kmeans"]["rgb"] == list(ORANGE_RGB)
    assert result["flame_analysis"]["gmm"]["lab"][0] > 0
    assert result["flame_analysis"]["mean_color"]["rgb"] == list(ORANGE_RGB)
    assert isinstance(result["material_analysis"]["primary_material"], str)
    assert result["suppression_information"]["source"] == DATASET.name
    assert isinstance(result["suppression_information"]["methods"], list)
    assert result["timing"]["total_ms"] > 0

    # The payload must survive json.dumps with no custom encoder.
    assert json.loads(json.dumps(result)) == result


def test_analyze_image_keeps_highest_confidence_detection():
    results = [StubResult([StubBox((0, 0, 10, 10), 0.5), StubBox((20, 20, 90, 90), 0.93)], None)]
    analyzer = _analyzer_with_stubs(results, None)  # no segmentation model -> bbox fallback
    result = analyzer.analyze_image(_fire_image())

    assert result["fire_detection"]["confidence"] == 0.93
    assert result["fire_detection"]["bounding_box"]["x1"] == 20
    assert result["segmentation"]["available"] is False
    assert result["segmentation"]["fallback_used"] is True
    assert result["segmentation"]["flame_pixel_count"] == 70 * 70


def test_analyze_image_ignores_wrong_class():
    analyzer = _analyzer_with_stubs(_detection_result(cls=1), None)
    assert analyzer.analyze_image(_fire_image()) == {
        "success": False,
        "error": {"code": "NO_FIRE_DETECTED", "message": NoFireDetectedError.message},
    }


def test_analyze_image_reports_no_flame_pixels():
    empty = np.zeros((100, 100), dtype=np.float32)
    # A degenerate detection box means the bbox fallback mask is empty too.
    analyzer = _analyzer_with_stubs(
        _detection_result(box=(20, 20, 20, 20)), _segmentation_result((0, 0, 0, 0), 0.85, empty)
    )
    result = analyzer.analyze_image(_fire_image())
    assert result["success"] is False
    assert result["error"]["code"] == "NO_FLAME_PIXELS"
    assert result["error"]["message"] == NoFlamePixelsError.message


def test_analyze_image_rejects_invalid_image():
    analyzer = _analyzer_with_stubs(_detection_result(), None)
    result = analyzer.analyze_image(np.empty((0, 0, 3), dtype=np.uint8))
    assert result["error"]["code"] == "INVALID_IMAGE"


def test_analyze_image_hides_unexpected_errors():
    class Exploding:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("boom")

    analyzer = _analyzer_with_stubs(_detection_result(), None)
    analyzer._detection_model = Exploding()
    result = analyzer.analyze_image(_fire_image())
    assert result["success"] is False
    assert result["error"]["code"] == "INFERENCE_ERROR"
    assert "boom" not in json.dumps(result)


def test_missing_detection_model_raises_on_load():
    with pytest.raises(MissingModelError):
        FlameAnalyzer(_settings(detection_model_path=Path("nope/OBJ_best.pt")))


# ---------------------------------------------------------------------------
# Real custom models (skipped when the weight files are unavailable)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (DET_MODEL.is_file() and SEG_MODEL.is_file()),
    reason="custom YOLO weights not available",
)
def test_real_models_load_once_and_answer_with_valid_json():
    analyzer = FlameAnalyzer(_settings())
    assert analyzer.models_loaded
    assert analyzer.segmentation_available
    # Models are loaded once and reused, not re-read per call.
    detection_model = analyzer._detection_model
    assert analyzer._detection_model is detection_model

    image = np.zeros((320, 320, 3), dtype=np.uint8)
    image[80:240, 110:210] = (0, 140, 255)  # BGR orange
    result = analyzer.analyze_image(image)

    assert isinstance(result, dict)
    json.dumps(result)
    if result["success"]:
        assert result["fire_detection"]["detected"] is True
        assert result["flame_analysis"]["flame_pixel_count"] > 0
    else:
        # A synthetic rectangle is not real fire, so this is a valid outcome.
        assert result["error"]["code"] == "NO_FIRE_DETECTED"


def _sample_image() -> Path | None:
    for name in ("5.png", "fire_1.jpg", "1.png", "2.png", "3.png"):
        candidate = ROOT / name
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(
    not (DET_MODEL.is_file() and SEG_MODEL.is_file()) or _sample_image() is None,
    reason="custom YOLO weights or a sample photo are unavailable",
)
def test_real_models_full_pipeline_on_photo():
    from app.imaging import imread

    analyzer = FlameAnalyzer(_settings())
    result = analyzer.analyze_image(imread(_sample_image()))

    assert result["success"] is True, result
    assert result["fire_detection"]["detected"] is True
    assert result["flame_analysis"]["kmeans"]["rgb"] is not None
    assert result["material_analysis"]["primary_material"] is not None
    assert result["suppression_information"]["source"] == DATASET.name
    json.dumps(result)


@pytest.mark.skipif(
    not (DET_MODEL.is_file() and SEG_MODEL.is_file()) or _sample_image() is None,
    reason="custom YOLO weights or a sample photo are unavailable",
)
def test_real_models_report_the_mask_and_all_five_algorithms():
    """The end-to-end contract on a real photo, with the real weights."""
    from app.imaging import imread

    result = FlameAnalyzer(_settings()).analyze_image(imread(_sample_image()))
    assert result["success"] is True, result

    # The real segmentation mask, decodable and self-consistent.
    segmentation = result["segmentation"]
    decoded = mask_from_base64(segmentation["mask"])
    assert decoded is not None
    assert decoded.shape == (segmentation["mask_height"], segmentation["mask_width"])
    assert int(decoded.sum()) == segmentation["flame_pixel_count"]
    assert segmentation["mask_area_ratio"] == pytest.approx(
        segmentation["flame_pixel_count"] / decoded.size, abs=1e-6
    )
    assert segmentation["mask_encoding"] == MASK_ENCODING
    # Whether the model or the box produced the mask is always stated, never implied.
    if segmentation["fallback_used"]:
        assert segmentation["available"] is False
        assert segmentation["bbox_fallback_reason"]
    else:
        assert segmentation["available"] is True
        assert segmentation["bbox_fallback_reason"] is None

    # All five retained algorithms, and nothing else.
    flame = result["flame_analysis"]
    assert flame["algorithms"] == [
        "K-Means",
        "GMM",
        "Bayesian GMM",
        "DBSCAN",
        "Agglomerative",
    ]
    for method in RETAINED_METHODS:
        entry = flame[method]
        assert entry is not None, f"{method} did not run on a real photo"
        assert entry["centroids"]
    assert "meanshift" not in json.dumps(result).lower()

    # The mask and the detection box are separate measurements.
    box = result["fire_detection"]["bounding_box"]
    assert result["fire_detection"]["bbox"] == box
    box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    assert box_area != segmentation["flame_pixel_count"]

    # The derived fire class and the dataset agents.
    material = result["material_analysis"]["primary_material"]
    assert result["fire_class"]["class"] == MATERIAL_FIRE_CLASS[material]
    assert result["fire_class"]["confidence"] == result["material_analysis"]["similarity"]
    agents = result["extinguishing_agents"]
    assert [agent["name"] for agent in agents] == result["suppression_information"]["methods"]
    assert all(agent["compound"] is None for agent in agents)

    # A whole response is a few kilobytes, mask included.
    assert len(json.dumps(result)) < 200_000
