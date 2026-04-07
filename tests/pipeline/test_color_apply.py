"""Tests for pipeline.color_apply — deterministic color-space transforms."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pipeline.color_apply import _hue_rotate, _lab_shift, _tint_blend


def _solid_rgba(r: int, g: int, b: int, a: int = 255, size: int = 8) -> Image.Image:
    """Create a small solid-color RGBA image."""
    return Image.new("RGBA", (size, size), (r, g, b, a))


# ---------------------------------------------------------------------------
# _hue_rotate
# ---------------------------------------------------------------------------

class TestHueRotate:
    def test_red_to_blue(self):
        """Rotate red (hue~0) to blue (hue~120) — blue channel should dominate."""
        crop = _solid_rgba(220, 30, 30)  # saturated red
        result = _hue_rotate(crop, source_hue=0, target_hue=120)
        px = np.array(result)
        # Blue channel should be the dominant color channel
        assert px[0, 0, 2] > px[0, 0, 0], "blue should exceed red after rotation"
        assert px[0, 0, 2] > px[0, 0, 1], "blue should exceed green after rotation"

    def test_preserves_alpha(self):
        """RGBA with alpha=128 → alpha unchanged after rotation."""
        crop = _solid_rgba(200, 50, 50, a=128)
        result = _hue_rotate(crop, source_hue=0, target_hue=60)
        px = np.array(result)
        assert np.all(px[:, :, 3] == 128)

    def test_skips_desaturated(self):
        """Gray pixels (100,100,100) unchanged after rotation."""
        crop = _solid_rgba(100, 100, 100)
        result = _hue_rotate(crop, source_hue=0, target_hue=90)
        px = np.array(result)
        np.testing.assert_array_equal(px[:, :, :3], 100)

    def test_zero_delta_identity(self):
        """Same source and target → output equals input exactly."""
        crop = _solid_rgba(180, 60, 60)
        result = _hue_rotate(crop, source_hue=50, target_hue=50)
        np.testing.assert_array_equal(np.array(result), np.array(crop))


# ---------------------------------------------------------------------------
# _lab_shift
# ---------------------------------------------------------------------------

class TestLabShift:
    def test_preserves_luminance(self):
        """L* should not change (within ±2)."""
        import cv2
        crop = _solid_rgba(180, 120, 100)
        arr = np.array(crop)[:, :, :3]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        lab_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        source_lab = np.array([lab_before[0, 0, 0], 128.0, 128.0])
        target_lab = np.array([lab_before[0, 0, 0], 160.0, 100.0])
        result = _lab_shift(crop, source_lab, target_lab)

        res_arr = np.array(result)[:, :, :3]
        bgr_res = cv2.cvtColor(res_arr, cv2.COLOR_RGB2BGR)
        lab_after = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2LAB).astype(np.float32)

        np.testing.assert_allclose(
            lab_after[0, 0, 0], lab_before[0, 0, 0], atol=2,
        )

    def test_preserves_alpha(self):
        """Alpha unchanged after LAB shift."""
        crop = _solid_rgba(180, 100, 80, a=200)
        source_lab = np.array([128.0, 128.0, 128.0])
        target_lab = np.array([128.0, 150.0, 110.0])
        result = _lab_shift(crop, source_lab, target_lab)
        px = np.array(result)
        assert np.all(px[:, :, 3] == 200)

    def test_skips_desaturated(self):
        """Gray pixels unchanged after LAB shift."""
        crop = _solid_rgba(100, 100, 100)
        source_lab = np.array([128.0, 128.0, 128.0])
        target_lab = np.array([128.0, 160.0, 100.0])
        result = _lab_shift(crop, source_lab, target_lab)
        px = np.array(result)
        np.testing.assert_array_equal(px[:, :, :3], 100)


# ---------------------------------------------------------------------------
# _tint_blend
# ---------------------------------------------------------------------------

class TestTintBlend:
    def test_shifts_toward_target(self):
        """a* channel moves toward target value."""
        import cv2
        crop = _solid_rgba(180, 140, 140)
        arr = np.array(crop)[:, :, :3]
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        lab_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_before = lab_before[0, 0, 1]

        target_ab = np.array([180.0, 128.0])
        result = _tint_blend(crop, target_ab, strength=0.5)

        res_arr = np.array(result)[:, :, :3]
        bgr_res = cv2.cvtColor(res_arr, cv2.COLOR_RGB2BGR)
        lab_after = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_after = lab_after[0, 0, 1]

        # a* should have moved toward 180
        assert abs(a_after - 180.0) < abs(a_before - 180.0), (
            f"a* should move toward target: before={a_before}, after={a_after}"
        )

    def test_preserves_alpha(self):
        """Alpha unchanged after tint blend."""
        crop = _solid_rgba(150, 120, 120, a=100)
        target_ab = np.array([160.0, 130.0])
        result = _tint_blend(crop, target_ab, strength=0.3)
        px = np.array(result)
        assert np.all(px[:, :, 3] == 100)
