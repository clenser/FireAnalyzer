"""Tests for the 40/35/25 EDT flame-zone RGB/LAB calculation.

The final flame colour is no longer one global mask average: the flame mask is
split by Euclidean Distance Transform depth into INNER (deepest 40%), MIDDLE
(next 35%) and OUTER (outermost 25%), and RGB/LAB are averaged independently
per zone and combined as 0.40*inner + 0.35*middle + 0.25*outer.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.color_analysis import (  # noqa: E402
    ZONE_WEIGHTS,
    _flame_depths,
    _to_lab,
    _zone_indices,
    _zone_weighted_mean_color,
    analyze_flame_colors,
    extract_flame_pixels,
)
from app.config import Settings  # noqa: E402

SETTINGS = Settings.from_env()

#: BGR orange used throughout the existing suite.
ORANGE_BGR = (0, 140, 255)
ORANGE_RGB = [255, 140, 0]


def _settings(**overrides) -> Settings:
    return SETTINGS.with_overrides(**overrides) if overrides else SETTINGS


def _square_mask(size: int = 100, box: int = 40) -> np.ndarray:
    """A centred ``box`` x ``box`` square mask."""
    mask = np.zeros((size, size), dtype=np.uint8)
    start = (size - box) // 2
    mask[start : start + box, start : start + box] = 255
    return mask


# ---------------------------------------------------------------------------
# 1. EDT-based zone ordering
# ---------------------------------------------------------------------------
def test_edt_measures_distance_from_boundary():
    depths = _flame_depths(_square_mask())
    assert depths.size == 1600
    # Boundary-adjacent pixels are shallow, centre pixels are deepest.
    assert depths.min() == pytest.approx(1.0, abs=0.2)
    assert depths.max() == pytest.approx(20.0, abs=1.5)


def test_deepest_pixels_belong_to_inner_zone():
    depths = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    inner, middle, outer = _zone_indices(depths)
    # Every INNER pixel is at least as deep as every MIDDLE pixel, and every
    # MIDDLE pixel is at least as deep as every OUTER pixel.
    assert depths[inner].min() >= depths[middle].max()
    assert depths[middle].min() >= depths[outer].max()
    # The single deepest pixel is in INNER.
    assert int(depths.argmax()) in set(inner.tolist())


def test_inner_zone_is_deeper_than_outer_zone_end_to_end():
    """Through the full pipeline: colour each flame pixel by its own EDT depth,
    so the deepest (INNER) zone must come out brightest."""
    mask = _square_mask()
    depths = _flame_depths(mask)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[mask > 0] = np.clip(depths[:, None] * np.ones((1, 3)), 0, 255).astype(np.uint8)
    result = analyze_flame_colors(image, mask, _settings())
    inner = np.array(result.zones["inner_rgb"])
    middle = np.array(result.zones["middle_rgb"])
    outer = np.array(result.zones["outer_rgb"])
    assert inner.mean() > middle.mean() > outer.mean()


# ---------------------------------------------------------------------------
# 2. 40/35/25 partitioning
# ---------------------------------------------------------------------------
def test_zone_partition_is_exactly_40_35_25():
    depths = np.arange(1.0, 1601.0)  # 1600 pixels
    inner, middle, outer = _zone_indices(depths)
    assert inner.size == 640  # 40%
    assert middle.size == 560  # 35%
    assert outer.size == 400  # 25%


def test_zone_partition_through_analyze_flame_colors():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = ORANGE_BGR
    result = analyze_flame_colors(image, _square_mask(), _settings())
    assert result.zones["zone_pixel_counts"] == {
        "inner": 640,
        "middle": 560,
        "outer": 400,
    }


# ---------------------------------------------------------------------------
# 3. Correct weighted RGB
# ---------------------------------------------------------------------------
def test_weighted_rgb_calculation():
    # 10 pixels: deepest 4 pure red (INNER), next 4 pure green (MIDDLE),
    # outermost 2 pure blue (OUTER).
    samples = np.array(
        [[255, 0, 0]] * 4 + [[0, 255, 0]] * 4 + [[0, 0, 255]] * 2,
        dtype=np.uint8,
    )
    depths = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    color, zones = _zone_weighted_mean_color(samples, _to_lab(samples), depths)

    assert zones["inner_rgb"] == [255.0, 0.0, 0.0]
    assert zones["middle_rgb"] == [0.0, 255.0, 0.0]
    assert zones["outer_rgb"] == [0.0, 0.0, 255.0]
    # 0.40*255 = 102, 0.35*255 = 89.25 -> 89, 0.25*255 = 63.75 -> 64
    assert zones["ultimate_rgb"] == [102, 89, 64]
    assert color.rgb.tolist() == [102, 89, 64]


# ---------------------------------------------------------------------------
# 4. Correct weighted LAB (per-zone, never from the weighted RGB)
# ---------------------------------------------------------------------------
def test_weighted_lab_calculation():
    samples = np.array(
        [[255, 0, 0]] * 4 + [[0, 255, 0]] * 4 + [[0, 0, 255]] * 2,
        dtype=np.uint8,
    )
    depths = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    lab_samples = _to_lab(samples)
    color, zones = _zone_weighted_mean_color(samples, lab_samples, depths)

    inner_lab = np.mean(lab_samples[:4], axis=0)
    middle_lab = np.mean(lab_samples[4:8], axis=0)
    outer_lab = np.mean(lab_samples[8:], axis=0)
    expected_lab = 0.40 * inner_lab + 0.35 * middle_lab + 0.25 * outer_lab

    assert np.allclose(zones["inner_lab"], np.round(inner_lab, 2))
    assert np.allclose(zones["middle_lab"], np.round(middle_lab, 2))
    assert np.allclose(zones["outer_lab"], np.round(outer_lab, 2))
    assert np.allclose(color.lab, expected_lab, atol=1e-9)
    assert np.allclose(zones["ultimate_lab"], np.round(expected_lab, 2))

    # LAB is averaged per zone, NOT converted from the weighted RGB.
    from skimage import color as skcolor

    weighted_rgb_as_lab = skcolor.rgb2lab(
        (np.array([102, 89, 64], dtype=np.float64).reshape(1, 3)) / 255.0
    )[0]
    assert not np.allclose(color.lab, weighted_rgb_as_lab, atol=1.0)


# ---------------------------------------------------------------------------
# 5. Ultimate RGB is NOT the global flame-mask average
# ---------------------------------------------------------------------------
def test_ultimate_rgb_is_not_the_global_mask_average():
    # 10 pixels shaded 0/100/200 by depth rank: the 40/35/25 weighting
    # (0.4*200 + 0.35*100 + 0.25*0 = 115) differs from the global mean (120).
    samples = np.array(
        [[0, 0, 0]] * 2 + [[100, 100, 100]] * 4 + [[200, 200, 200]] * 4,
        dtype=np.uint8,
    )
    depths = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    color, zones = _zone_weighted_mean_color(samples, _to_lab(samples), depths)

    global_mean = np.mean(samples, axis=0)
    assert np.allclose(global_mean, [120, 120, 120])
    assert np.allclose(zones["ultimate_rgb"], [115, 115, 115])
    assert not np.allclose(zones["ultimate_rgb"], np.round(global_mean))


def test_ultimate_rgb_differs_from_global_average_end_to_end():
    """3x3 mask (9 pixels -> 4/3/2 zones, not exactly 40/35/25) with a strong
    top-to-bottom gradient: the zone-weighted mean must differ from the global
    mask average."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10:13, 10:13] = 255
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    # BGR: blue channel increases with row index (top dark, bottom bright).
    image[10, :, 0] = 0
    image[11, :, 0] = 128
    image[12, :, 0] = 255
    result = analyze_flame_colors(image, mask, _settings())

    assert result.zones["zone_pixel_counts"] == {"inner": 4, "middle": 3, "outer": 2}
    global_mean = np.mean(extract_flame_pixels(image, mask), axis=0)
    ultimate = np.array(result.zones["ultimate_rgb"], dtype=float)
    assert not np.allclose(ultimate, np.round(global_mean), atol=5.0)


# ---------------------------------------------------------------------------
# 6. Small-mask fallback
# ---------------------------------------------------------------------------
def test_single_pixel_mask_falls_back_gracefully():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[10, 10] = ORANGE_BGR

    result = analyze_flame_colors(image, mask, _settings())

    # INNER and MIDDLE are empty and fall back to the global mean, so the
    # ultimate colour is exactly the single pixel - no crash, nothing invented.
    assert result.zones["zone_pixel_counts"] == {"inner": 0, "middle": 0, "outer": 1}
    assert result.mean.rgb.tolist() == ORANGE_RGB
    assert result.zones["ultimate_rgb"] == ORANGE_RGB


def test_two_pixel_mask_falls_back_gracefully():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 10] = 255
    mask[10, 11] = 255
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[10, 10] = ORANGE_BGR
    image[10, 11] = (255, 0, 0)  # pure blue

    result = analyze_flame_colors(image, mask, _settings())

    # OUTER is empty and falls back to the global mean:
    # 0.40*orange + 0.35*blue + 0.25*global = [134, 74, 121].
    assert result.zones["zone_pixel_counts"] == {"inner": 1, "middle": 1, "outer": 0}
    assert result.zones["ultimate_rgb"] == [134, 74, 121]


def test_empty_mask_still_reports_black():
    result = analyze_flame_colors(
        np.zeros((20, 20, 3), dtype=np.uint8),
        np.zeros((20, 20), dtype=np.uint8),
        _settings(),
    )
    assert result.mean.rgb.tolist() == [0, 0, 0]
    assert result.zones == {}


# ---------------------------------------------------------------------------
# Integration: uniform colour is unchanged, weights sum to 1
# ---------------------------------------------------------------------------
def test_uniform_flame_colour_is_preserved():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[30:70, 30:70] = ORANGE_BGR
    result = analyze_flame_colors(image, _square_mask(), _settings())
    assert result.mean.rgb.tolist() == ORANGE_RGB


def test_zone_weights_sum_to_one():
    assert sum(ZONE_WEIGHTS) == pytest.approx(1.0)
    assert ZONE_WEIGHTS == (0.40, 0.35, 0.25)
