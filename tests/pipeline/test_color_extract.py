"""Tests for pipeline.color_extract — ColorPalette + sampling helpers."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.color_extract import (
    ColorPalette,
    _sample_iris_hsv,
    _sample_lip_lab,
    _sample_patch_lab,
)


# ---------------------------------------------------------------------------
# ColorPalette dataclass
# ---------------------------------------------------------------------------

def _dummy_palette() -> ColorPalette:
    return ColorPalette(
        hair=np.array([50.0, 128.0, 128.0]),
        skin=np.array([200.0, 130.0, 140.0]),
        eye_color=100.0,
        eye_saturation=180.0,
        lip_color=np.array([120.0, 160.0, 130.0]),
        clothing=np.array([80.0, 120.0, 110.0]),
    )


class TestColorPalette:
    def test_field_shapes(self):
        p = _dummy_palette()
        assert p.hair.shape == (3,)
        assert p.skin.shape == (3,)
        assert p.lip_color.shape == (3,)
        assert p.clothing.shape == (3,)
        assert isinstance(p.eye_color, float)
        assert isinstance(p.eye_saturation, float)

    def test_roundtrip(self):
        p = _dummy_palette()
        d = p.to_dict()
        p2 = ColorPalette.from_dict(d)
        np.testing.assert_array_almost_equal(p.hair, p2.hair)
        np.testing.assert_array_almost_equal(p.skin, p2.skin)
        assert p.eye_color == pytest.approx(p2.eye_color)
        assert p.eye_saturation == pytest.approx(p2.eye_saturation)
        np.testing.assert_array_almost_equal(p.lip_color, p2.lip_color)
        np.testing.assert_array_almost_equal(p.clothing, p2.clothing)

    def test_to_dict_types(self):
        d = _dummy_palette().to_dict()
        assert isinstance(d["hair"], list)
        assert isinstance(d["eye_color"], float)


# ---------------------------------------------------------------------------
# _sample_patch_lab — solid-color synthetic images
# ---------------------------------------------------------------------------

def _solid_bgr(b: int, g: int, r: int, size: int = 100) -> np.ndarray:
    """Create a solid-color BGR image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = (b, g, r)
    return img


class TestSamplePatchLab:
    def test_red_high_a_star(self):
        """Pure red (BGR 0,0,255) should have high a* (> 128)."""
        img = _solid_bgr(0, 0, 255)
        lab = _sample_patch_lab(img, 50, 50, radius=10)
        assert lab.shape == (3,)
        assert lab[1] > 128, f"a* should be > 128 for red, got {lab[1]}"

    def test_green_low_a_star(self):
        """Pure green (BGR 0,255,0) should have low a* (< 128)."""
        img = _solid_bgr(0, 255, 0)
        lab = _sample_patch_lab(img, 50, 50, radius=10)
        assert lab[1] < 128, f"a* should be < 128 for green, got {lab[1]}"

    def test_edge_clamping(self):
        """Sampling near the image edge should not crash."""
        img = _solid_bgr(128, 128, 128, size=50)
        lab = _sample_patch_lab(img, 2, 2, radius=15)
        assert lab.shape == (3,)


# ---------------------------------------------------------------------------
# _sample_iris_hsv — solid blue
# ---------------------------------------------------------------------------

class TestSampleIrisHsv:
    def test_blue_hue(self):
        """Pure blue (BGR 255,0,0): OpenCV HSV hue ~120."""
        img = _solid_bgr(255, 0, 0, size=200)
        # Iris landmarks: center + 4 ring points (radius ~20)
        center = np.array([100, 100], dtype=np.float32)
        ring = np.array([
            [120, 100],
            [80, 100],
            [100, 120],
            [100, 80],
        ], dtype=np.float32)
        landmarks = np.vstack([center[None, :], ring])  # (5, 2)
        hue, sat = _sample_iris_hsv(img, landmarks)
        assert 95 <= hue <= 125, f"blue hue should be ~120, got {hue}"
        assert sat > 200, f"pure blue saturation should be high, got {sat}"

    def test_desaturated_gray(self):
        """Gray image should have low saturation."""
        img = _solid_bgr(128, 128, 128, size=200)
        landmarks = np.array([
            [100, 100],
            [120, 100],
            [80, 100],
            [100, 120],
            [100, 80],
        ], dtype=np.float32)
        _hue, sat = _sample_iris_hsv(img, landmarks)
        assert sat < 30, f"gray saturation should be near 0, got {sat}"


# ---------------------------------------------------------------------------
# _sample_lip_lab — solid red
# ---------------------------------------------------------------------------

class TestSampleLipLab:
    def test_red_high_a_star(self):
        """Red lip region should have high a*."""
        img = _solid_bgr(0, 0, 255, size=200)
        # Simple polygon: a small rectangle of landmarks
        lip_lm = np.array([
            [80, 90],
            [120, 90],
            [120, 110],
            [80, 110],
        ], dtype=np.float32)
        lab = _sample_lip_lab(img, lip_lm)
        assert lab.shape == (3,)
        assert lab[1] > 128, f"a* should be > 128 for red lips, got {lab[1]}"
