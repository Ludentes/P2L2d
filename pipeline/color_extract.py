"""Color extraction from portrait photos using MediaPipe landmarks.

Extracts a ColorPalette (hair, skin, eyes, lips, clothing) from a portrait,
using landmark-guided sampling regions and LAB/HSV color spaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Landmark index groups (MediaPipe FaceMesh 478-point topology)
# ---------------------------------------------------------------------------

_CHEEK_LEFT = [116, 123, 205]
_CHEEK_RIGHT = [345, 352, 425]
_IRIS_LEFT = [468, 469, 470, 471, 472]
_IRIS_RIGHT = [473, 474, 475, 476, 477]
_INNER_LIP = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]


# ---------------------------------------------------------------------------
# ColorPalette dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColorPalette:
    """Extracted color palette from a portrait photo.

    LAB arrays are [L, a, b] in OpenCV LAB scale (L: 0-255, a/b: 0-255 centered at 128).
    HSV values use OpenCV scale (hue: 0-180, saturation: 0-255).
    """

    hair: np.ndarray        # LAB [L, a, b]
    skin: np.ndarray        # LAB [L, a, b]
    eye_color: float        # HSV hue (0-180)
    eye_saturation: float   # HSV saturation (0-255)
    lip_color: np.ndarray   # LAB [L, a, b]
    clothing: np.ndarray    # LAB [L, a, b]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hair": self.hair.tolist(),
            "skin": self.skin.tolist(),
            "eye_color": self.eye_color,
            "eye_saturation": self.eye_saturation,
            "lip_color": self.lip_color.tolist(),
            "clothing": self.clothing.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColorPalette:
        return cls(
            hair=np.array(d["hair"], dtype=np.float64),
            skin=np.array(d["skin"], dtype=np.float64),
            eye_color=float(d["eye_color"]),
            eye_saturation=float(d["eye_saturation"]),
            lip_color=np.array(d["lip_color"], dtype=np.float64),
            clothing=np.array(d["clothing"], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_patch_lab(
    img_bgr: np.ndarray, x: int, y: int, radius: int = 15,
) -> np.ndarray:
    """Sample median LAB color from a circular patch around (x, y).

    Returns np.ndarray of shape (3,) with [L, a, b] float64.
    Handles edge clamping gracefully.
    """
    h, w = img_bgr.shape[:2]
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius + 1)
    y1 = min(h, y + radius + 1)

    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([128.0, 128.0, 128.0])

    # Build circular mask within the cropped patch
    ph, pw = patch.shape[:2]
    cy, cx = y - y0, x - x0
    yy, xx = np.ogrid[:ph, :pw]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    pixels = lab[mask]
    if len(pixels) == 0:
        return np.array([128.0, 128.0, 128.0])

    return np.median(pixels, axis=0).astype(np.float64)


def _sample_iris_hsv(
    img_bgr: np.ndarray, iris_landmarks: np.ndarray,
) -> tuple[float, float]:
    """Sample median hue and saturation from iris landmark region.

    Args:
        img_bgr: BGR image.
        iris_landmarks: (5, 2) array — center point + 4 ring points.

    Returns:
        (hue, saturation) in OpenCV HSV scale (hue 0-180, sat 0-255).
    """
    center = iris_landmarks[0].astype(np.float64)
    ring = iris_landmarks[1:]
    # Radius = mean distance from center to ring points
    dists = np.linalg.norm(ring.astype(np.float64) - center, axis=1)
    radius = max(int(np.mean(dists)), 1)

    cx, cy = int(center[0]), int(center[1])
    h, w = img_bgr.shape[:2]

    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(w, cx + radius + 1)
    y1 = min(h, cy + radius + 1)

    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0

    ph, pw = patch.shape[:2]
    local_cx, local_cy = cx - x0, cy - y0
    yy, xx = np.ogrid[:ph, :pw]
    mask = ((xx - local_cx) ** 2 + (yy - local_cy) ** 2) <= radius ** 2

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask]
    if len(pixels) == 0:
        return 0.0, 0.0

    med_hue = float(np.median(pixels[:, 0]))
    med_sat = float(np.median(pixels[:, 1]))
    return med_hue, med_sat


def _sample_lip_lab(
    img_bgr: np.ndarray, lip_landmarks: np.ndarray,
) -> np.ndarray:
    """Sample median LAB from inner lip polygon.

    Args:
        img_bgr: BGR image.
        lip_landmarks: (N, 2) array of polygon vertices (x, y).

    Returns:
        np.ndarray of shape (3,) with [L, a, b] float64.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = lip_landmarks.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[mask == 255]
    if len(pixels) == 0:
        return np.array([128.0, 128.0, 128.0])

    return np.median(pixels, axis=0).astype(np.float64)
