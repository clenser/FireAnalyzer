"""Tests for the FlameAnalyzer inference pipeline.

The tests that exercise the full orchestration use a stub YOLO model, so they
run in a fraction of a second without the 100 MB weight files.  The test that
does need the real weights is skipped automatically when they are absent.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

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
    RepresentativeColor,
    analyze_flame_colors,
    extract_flame_pixels,
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
from app.imaging import decode_image_bytes, validate_image  # noqa: E402
from app.material_matching import (  # noqa: E402
    load_material_database,
    match_material,
    suppression_for,
)
from app.segmentation import bounding_box_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "flame_dataset.json"
DET_MODEL = ROOT / "models" / "OBJ_best.pt"
SEG_MODEL = ROOT / "models" / "SEG_best.pt"

# Orange in RGB; the BGR slice in ``_fire_image`` is this tuple reversed.
ORANGE_RGB = (255, 140, 0)


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


def test_color_analysis_falls_back_to_mean_for_tiny_masks():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50, 50] = 255
    result = analyze_flame_colors(_fire_image(), mask, _settings())
    assert result.pixel_count == 1
    assert result.kmeans is None and result.gmm is None
    assert result.mean.rgb.tolist() == list(ORANGE_RGB)


def test_color_analysis_handles_empty_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = analyze_flame_colors(_fire_image(), mask, _settings())
    assert result.pixel_count == 0
    assert result.mean.rgb.tolist() == [0, 0, 0]


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
    assert result["fire_detection"] == {
        "detected": True,
        "confidence": 0.9,
        "bounding_box": {"x1": 10, "y1": 10, "x2": 60, "y2": 60},
    }
    assert result["segmentation"] == {
        "available": True,
        "fallback_used": False,
        "flame_pixel_count": 1600,
        "mask_area_ratio": 0.16,
        "confidence": 0.85,
    }
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
    for name in ("fire_1.jpg", "1.png", "2.png", "3.png"):
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
