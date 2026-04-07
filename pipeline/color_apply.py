"""Atlas recoloring via deterministic color-space transforms.

Applies portrait-extracted colors to Live2D atlas regions using:
- HSV hue rotation for hair, eyes, clothing, eyebrows
- LAB a*b* shift for skin and lips
- LAB tint blend for cheek blush
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _hue_rotate(
    crop: Image.Image,
    source_hue: float,
    target_hue: float,
    sat_threshold: int = 15,
) -> Image.Image:
    """Rotate hue of saturated pixels from *source_hue* toward *target_hue*.

    Hue values use OpenCV scale (0-180).  Desaturated pixels (saturation
    <= *sat_threshold*) and the alpha channel are preserved unchanged.
    """
    arr = np.array(crop)  # RGBA uint8
    alpha = arr[:, :, 3].copy()
    rgb = arr[:, :, :3]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)

    delta = int(target_hue - source_hue)
    if delta == 0:
        return crop.copy()

    mask = hsv[:, :, 1] > sat_threshold

    hsv[:, :, 0][mask] = (hsv[:, :, 0][mask] + delta) % 180

    hsv = hsv.astype(np.uint8)
    bgr_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    out = np.dstack([rgb_out, alpha])
    return Image.fromarray(out, "RGBA")


def _lab_shift(
    crop: Image.Image,
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    sat_threshold: int = 15,
) -> Image.Image:
    """Shift a* and b* channels by the delta between *source_lab* and *target_lab*.

    L* is unchanged (preserves shading).  Desaturated pixels (HSV
    saturation <= *sat_threshold*) and alpha are preserved.
    """
    arr = np.array(crop)  # RGBA uint8
    alpha = arr[:, :, 3].copy()
    rgb = arr[:, :, :3]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Build saturation mask via HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] > sat_threshold

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    delta_a = float(target_lab[1] - source_lab[1])
    delta_b = float(target_lab[2] - source_lab[2])

    lab[:, :, 1][mask] = np.clip(lab[:, :, 1][mask] + delta_a, 0, 255)
    lab[:, :, 2][mask] = np.clip(lab[:, :, 2][mask] + delta_b, 0, 255)

    lab = lab.astype(np.uint8)
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    out = np.dstack([rgb_out, alpha])
    return Image.fromarray(out, "RGBA")


def _tint_blend(
    crop: Image.Image,
    target_ab: np.ndarray,
    strength: float = 0.3,
) -> Image.Image:
    """Blend a* and b* channels toward *target_ab* with given *strength*.

    Applies to all pixels (no saturation gate — intended for blush layers).
    L* and alpha are preserved.
    """
    arr = np.array(crop)  # RGBA uint8
    alpha = arr[:, :, 3].copy()
    rgb = arr[:, :, :3]

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    lab[:, :, 1] = lab[:, :, 1] * (1 - strength) + float(target_ab[0]) * strength
    lab[:, :, 2] = lab[:, :, 2] * (1 - strength) + float(target_ab[1]) * strength

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    out = np.dstack([rgb_out, alpha])
    return Image.fromarray(out, "RGBA")
