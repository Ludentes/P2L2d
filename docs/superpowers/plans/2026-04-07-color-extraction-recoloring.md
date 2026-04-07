# Color Extraction + Atlas Recoloring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract hair, skin, eye, lip, and clothing colors from a portrait photo and apply them to Live2D atlas textures via deterministic color-space remapping.

**Architecture:** Two-stage pipeline — `extract_palette()` uses MediaPipe landmarks + BiSeNet face parsing to sample colors from the portrait into a `ColorPalette` dataclass. `recolor_atlas()` applies those colors to atlas regions using hue rotation (hair/eyes/clothing) or LAB a*b* shift (skin/lips). Line art preserved by skipping low-saturation pixels.

**Tech Stack:** Python 3.12, MediaPipe (landmarks), BiSeNet face parsing (via `facer` or fallback), OpenCV (color space conversions), NumPy, scikit-learn (k-means for clothing color), PIL/Pillow.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `pipeline/color_extract.py` | `ColorPalette` dataclass + `extract_palette()` — samples colors from portrait |
| `pipeline/color_apply.py` | `recolor_atlas()` + color transform helpers (`_hue_rotate`, `_lab_shift`, `_tint_blend`) |
| `pipeline/template_palette.py` | `extract_template_palette()` — measures baseline colors from unmodified atlas |
| `tests/pipeline/test_color_extract.py` | Unit tests for palette extraction |
| `tests/pipeline/test_color_apply.py` | Unit tests for color transforms + atlas recoloring |
| `tests/pipeline/test_template_palette.py` | Unit tests for template palette measurement |
| `scripts/prototype_color_recolor.py` | Visual integration test — recolor Hiyori + render comparison |

---

### Task 1: ColorPalette dataclass + landmark-based color sampling

**Files:**
- Create: `pipeline/color_extract.py`
- Create: `tests/pipeline/test_color_extract.py`

- [ ] **Step 1: Write failing test for ColorPalette dataclass**

```python
# tests/pipeline/test_color_extract.py
"""Tests for portrait color extraction."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pipeline.color_extract import ColorPalette


class TestColorPalette:
    def test_create_palette(self):
        palette = ColorPalette(
            hair=np.array([30.0, 5.0, 10.0]),
            skin=np.array([70.0, 15.0, 20.0]),
            eye_color=120.0,
            eye_saturation=180.0,
            lip_color=np.array([50.0, 30.0, 15.0]),
            clothing=np.array([40.0, -10.0, 25.0]),
        )
        assert palette.hair.shape == (3,)
        assert palette.eye_color == 120.0
        assert palette.eye_saturation == 180.0

    def test_palette_to_dict_roundtrip(self):
        palette = ColorPalette(
            hair=np.array([30.0, 5.0, 10.0]),
            skin=np.array([70.0, 15.0, 20.0]),
            eye_color=120.0,
            eye_saturation=180.0,
            lip_color=np.array([50.0, 30.0, 15.0]),
            clothing=np.array([40.0, -10.0, 25.0]),
        )
        d = palette.to_dict()
        restored = ColorPalette.from_dict(d)
        np.testing.assert_array_almost_equal(restored.hair, palette.hair)
        np.testing.assert_array_almost_equal(restored.skin, palette.skin)
        assert restored.eye_color == palette.eye_color
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.color_extract'`

- [ ] **Step 3: Implement ColorPalette dataclass with serialization**

```python
# pipeline/color_extract.py
"""Portrait color extraction for atlas recoloring.

Samples hair, skin, eye, lip, and clothing colors from a portrait photo
using MediaPipe landmarks and optional BiSeNet face parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_extract.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test for landmark-based sampling helpers**

Add to `tests/pipeline/test_color_extract.py`:

```python
from pipeline.color_extract import (
    _sample_patch_lab,
    _sample_iris_hsv,
    _sample_lip_lab,
)


class TestLandmarkSampling:
    def _make_solid_image(self, rgb: tuple[int, int, int], size: tuple[int, int] = (200, 200)) -> Image.Image:
        return Image.new("RGB", size, rgb)

    def test_sample_patch_lab_solid_color(self):
        """Sampling from a solid red image should return consistent LAB."""
        img = self._make_solid_image((200, 100, 100))
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lab = _sample_patch_lab(img_bgr, x=100, y=100, radius=10)
        assert lab.shape == (3,)
        # Red has high a* (> 128) in OpenCV LAB
        assert lab[1] > 140

    def test_sample_iris_hsv_blue(self):
        """Blue iris should have hue around 100-120 in OpenCV HSV."""
        img = self._make_solid_image((50, 50, 200))
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        landmarks = np.array([
            [100.0, 100.0],  # iris center
            [105.0, 100.0],  # iris ring
            [100.0, 105.0],
            [95.0, 100.0],
            [100.0, 95.0],
        ], dtype=np.float32)
        hue, sat = _sample_iris_hsv(img_bgr, landmarks)
        assert 95 <= hue <= 125  # blue hue range
        assert sat > 100  # clearly saturated

    def test_sample_lip_lab_red_lips(self):
        """Red lips should have high a* in LAB."""
        img = self._make_solid_image((180, 60, 60))
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        landmarks = np.array([
            [90.0, 100.0], [100.0, 105.0], [110.0, 100.0],
            [100.0, 95.0], [95.0, 102.0], [105.0, 102.0],
        ], dtype=np.float32)
        lab = _sample_lip_lab(img_bgr, landmarks)
        assert lab.shape == (3,)
        assert lab[1] > 150  # strong red → high a*
```

- [ ] **Step 6: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_extract.py::TestLandmarkSampling -v`
Expected: FAIL — `ImportError: cannot import name '_sample_patch_lab'`

- [ ] **Step 7: Implement landmark-based sampling helpers**

Add to `pipeline/color_extract.py`:

```python
# ── Landmark-based color sampling helpers ─────────────────────────────────────

# MediaPipe landmark indices for sampling zones
_CHEEK_LEFT = [116, 123, 205]
_CHEEK_RIGHT = [345, 352, 425]
_IRIS_LEFT = [468, 469, 470, 471, 472]   # left iris center + ring
_IRIS_RIGHT = [473, 474, 475, 476, 477]  # right iris center + ring
_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def _sample_patch_lab(
    img_bgr: np.ndarray, x: int, y: int, radius: int = 15
) -> np.ndarray:
    """Sample median LAB color from a circular patch around (x, y)."""
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([128.0, 128.0, 128.0])
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0)


def _sample_iris_hsv(
    img_bgr: np.ndarray, iris_landmarks: np.ndarray
) -> tuple[float, float]:
    """Sample median hue and saturation from iris landmark region.

    Args:
        iris_landmarks: (5, 2) array — iris center + 4 ring points.

    Returns:
        (hue, saturation) in OpenCV HSV scale.
    """
    center = iris_landmarks[0].astype(int)
    ring = iris_landmarks[1:].astype(int)
    radius = max(3, int(np.mean(np.linalg.norm(ring - center, axis=1))))

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), radius, 255, -1)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        return 0.0, 0.0
    return float(np.median(pixels[:, 0])), float(np.median(pixels[:, 1]))


def _sample_lip_lab(
    img_bgr: np.ndarray, lip_landmarks: np.ndarray
) -> np.ndarray:
    """Sample median LAB from the inner lip region."""
    h, w = img_bgr.shape[:2]
    pts = lip_landmarks.astype(np.int32).reshape(-1, 1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]
    if len(pixels) == 0:
        return np.array([128.0, 128.0, 128.0])
    return np.median(pixels.reshape(-1, 3), axis=0)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_extract.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add pipeline/color_extract.py tests/pipeline/test_color_extract.py
git commit -m "feat: ColorPalette dataclass + landmark-based color sampling helpers"
```

---

### Task 2: `extract_palette()` — full portrait color extraction

**Files:**
- Modify: `pipeline/color_extract.py`
- Modify: `tests/pipeline/test_color_extract.py`

- [ ] **Step 1: Write failing test for extract_palette**

Add to `tests/pipeline/test_color_extract.py`:

```python
from unittest.mock import patch
from pipeline.color_extract import extract_palette


class TestExtractPalette:
    def test_extract_palette_returns_palette(self):
        """extract_palette on a real portrait should return a valid ColorPalette."""
        portrait_path = Path("assets/data/image1.png")
        if not portrait_path.exists():
            pytest.skip("Test portrait not available")
        portrait = Image.open(portrait_path).convert("RGB")
        palette = extract_palette(portrait)
        assert isinstance(palette, ColorPalette)
        assert palette.hair.shape == (3,)
        assert palette.skin.shape == (3,)
        assert 0 <= palette.eye_color <= 180
        assert 0 <= palette.eye_saturation <= 255
        assert palette.lip_color.shape == (3,)
        assert palette.clothing.shape == (3,)

    def test_extract_palette_skin_has_warm_tone(self):
        """Portrait skin should have a* > 128 (warm/reddish in LAB)."""
        portrait_path = Path("assets/data/image1.png")
        if not portrait_path.exists():
            pytest.skip("Test portrait not available")
        portrait = Image.open(portrait_path).convert("RGB")
        palette = extract_palette(portrait)
        # Human skin is always warm (a* > 128 in OpenCV LAB)
        assert palette.skin[1] > 125
```

Add `from pathlib import Path` to the imports at top of file.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_extract.py::TestExtractPalette -v`
Expected: FAIL — `ImportError: cannot import name 'extract_palette'`

- [ ] **Step 3: Implement extract_palette**

Add to `pipeline/color_extract.py`:

```python
from pipeline.face_align import detect_landmarks


def _sample_region_by_bbox(
    img_bgr: np.ndarray,
    landmarks: np.ndarray,
    region: str,
) -> np.ndarray:
    """Fallback: sample color from approximate portrait region (no face parsing).

    For 'hair': top 20% of image above face bounding box.
    For 'clothing': bottom 30% of image below face bounding box.
    Returns median LAB.
    """
    h, w = img_bgr.shape[:2]
    face_top = int(landmarks[:, 1].min())
    face_bottom = int(landmarks[:, 1].max())

    if region == "hair":
        y1, y2 = max(0, face_top - int(h * 0.15)), face_top
        if y2 - y1 < 10:
            y1, y2 = 0, max(10, face_top)
    elif region == "clothing":
        y1 = min(h - 10, face_bottom + int(h * 0.05))
        y2 = h
    else:
        return np.array([128.0, 128.0, 128.0])

    x1, x2 = int(w * 0.2), int(w * 0.8)
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([128.0, 128.0, 128.0])
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0)


def _try_face_parsing(img_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
    """Try BiSeNet face parsing via facer. Returns label masks or None."""
    try:
        import facer
        import torch
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    face_detector = facer.face_detector("retinaface/mobilenet", device=device)
    faces = face_detector(img_tensor / 255.0)
    if faces["rects"].shape[0] == 0:
        return None

    face_parser = facer.face_parser("farl/lapa/448", device=device)
    parsed = face_parser(img_tensor / 255.0, faces)
    seg = parsed["seg"][0].cpu().numpy()  # (H, W) label map

    return {
        "skin": (seg == 1).astype(np.uint8),
        "hair": (seg == 10).astype(np.uint8),
        "clothing": (seg == 16).astype(np.uint8),
        "upper_lip": (seg == 12).astype(np.uint8),
        "lower_lip": (seg == 13).astype(np.uint8),
    }


def _sample_masked_lab(
    img_bgr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Median LAB of pixels where mask > 0."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]
    if len(pixels) == 0:
        return np.array([128.0, 128.0, 128.0])
    return np.median(pixels.reshape(-1, 3), axis=0)


def _sample_clothing_lab(
    img_bgr: np.ndarray, mask: np.ndarray | None, landmarks: np.ndarray
) -> np.ndarray:
    """Sample dominant clothing color via k-means (k=2, pick largest cluster)."""
    if mask is not None and mask.sum() > 100:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab[mask > 0].reshape(-1, 3).astype(np.float32)
    else:
        # Fallback: bottom region of portrait
        fallback_lab = _sample_region_by_bbox(img_bgr, landmarks, "clothing")
        return fallback_lab

    if len(pixels) < 10:
        return np.median(pixels, axis=0)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(2, len(pixels)), n_init=3, random_state=42)
    kmeans.fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant = labels[np.argmax(counts)]
    return kmeans.cluster_centers_[dominant]


def extract_palette(portrait: Image.Image) -> ColorPalette:
    """Extract color palette from portrait using landmarks + optional face parsing.

    Uses MediaPipe 478-point landmarks for iris/lip/skin sampling.
    Uses BiSeNet face parsing (if available) for hair and clothing masks.
    Falls back to bbox-based sampling if face parsing unavailable.
    """
    img_rgb = np.array(portrait.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    landmarks = detect_landmarks(portrait)  # (478, 2)

    # Try face parsing for hair/clothing masks
    parsing = _try_face_parsing(img_bgr)

    # Hair color
    if parsing and parsing["hair"].sum() > 100:
        hair = _sample_masked_lab(img_bgr, parsing["hair"])
    else:
        hair = _sample_region_by_bbox(img_bgr, landmarks, "hair")

    # Skin color — always from landmarks (most reliable)
    cheek_indices = _CHEEK_LEFT + _CHEEK_RIGHT
    skin_samples = []
    for idx in cheek_indices:
        x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
        skin_samples.append(_sample_patch_lab(img_bgr, x, y, radius=10))
    skin = np.median(skin_samples, axis=0)

    # Eye color — from iris landmarks
    iris_left = landmarks[_IRIS_LEFT]
    iris_right = landmarks[_IRIS_RIGHT]
    hue_l, sat_l = _sample_iris_hsv(img_bgr, iris_left)
    hue_r, sat_r = _sample_iris_hsv(img_bgr, iris_right)
    eye_color = float(np.mean([hue_l, hue_r]))
    eye_saturation = float(np.mean([sat_l, sat_r]))

    # Lip color — from inner lip landmarks
    lip_pts = landmarks[_INNER_LIP]
    lip_color = _sample_lip_lab(img_bgr, lip_pts)

    # Clothing color
    clothing_mask = parsing["clothing"] if parsing else None
    clothing = _sample_clothing_lab(img_bgr, clothing_mask, landmarks)

    return ColorPalette(
        hair=hair,
        skin=skin,
        eye_color=eye_color,
        eye_saturation=eye_saturation,
        lip_color=lip_color,
        clothing=clothing,
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_extract.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/color_extract.py tests/pipeline/test_color_extract.py
git commit -m "feat: extract_palette() — full portrait color extraction with landmark + parsing fallback"
```

---

### Task 3: Color transform helpers (`_hue_rotate`, `_lab_shift`, `_tint_blend`)

**Files:**
- Create: `pipeline/color_apply.py`
- Create: `tests/pipeline/test_color_apply.py`

- [ ] **Step 1: Write failing tests for color transforms**

```python
# tests/pipeline/test_color_apply.py
"""Tests for atlas color transforms."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from pipeline.color_apply import _hue_rotate, _lab_shift, _tint_blend


class TestHueRotate:
    def _make_rgba_crop(self, rgb: tuple[int, int, int], size: tuple[int, int] = (50, 50)) -> Image.Image:
        """Solid color RGBA image with full opacity."""
        img = Image.new("RGBA", size, (*rgb, 255))
        return img

    def test_hue_rotate_red_to_blue(self):
        """Rotating red (hue~0) to blue (hue~120) should produce blue pixels."""
        crop = self._make_rgba_crop((200, 50, 50))
        result = _hue_rotate(crop, source_hue=0, target_hue=120, sat_threshold=15)
        arr = np.array(result)
        # Blue channel should dominate
        assert arr[:, :, 2].mean() > arr[:, :, 0].mean()  # B > R

    def test_hue_rotate_preserves_alpha(self):
        """Alpha channel must be unchanged."""
        crop = Image.new("RGBA", (50, 50), (200, 50, 50, 128))
        result = _hue_rotate(crop, source_hue=0, target_hue=60, sat_threshold=15)
        assert np.array(result)[:, :, 3].mean() == 128

    def test_hue_rotate_skips_desaturated_pixels(self):
        """Gray pixels (S < threshold) should be unchanged."""
        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        arr[:, :, :3] = [100, 100, 100]  # gray
        arr[:, :, 3] = 255
        crop = Image.fromarray(arr, "RGBA")
        result = _hue_rotate(crop, source_hue=0, target_hue=120, sat_threshold=15)
        np.testing.assert_array_equal(np.array(result)[:, :, :3], arr[:, :, :3])

    def test_hue_rotate_zero_delta_is_identity(self):
        """Same source and target hue should produce identical output."""
        crop = self._make_rgba_crop((200, 50, 50))
        result = _hue_rotate(crop, source_hue=0, target_hue=0, sat_threshold=15)
        np.testing.assert_array_equal(np.array(result), np.array(crop))


class TestLabShift:
    def test_lab_shift_preserves_luminance(self):
        """L* channel must not change."""
        crop = Image.new("RGBA", (50, 50), (200, 180, 160, 255))
        source_lab = np.array([180.0, 140.0, 145.0])  # warm
        target_lab = np.array([180.0, 120.0, 130.0])  # cooler
        result = _lab_shift(crop, source_lab, target_lab, sat_threshold=15)
        # Convert both to LAB and compare L*
        orig_bgr = cv2.cvtColor(np.array(crop)[:, :, :3], cv2.COLOR_RGB2BGR)
        res_bgr = cv2.cvtColor(np.array(result)[:, :, :3], cv2.COLOR_RGB2BGR)
        orig_l = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
        res_l = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
        assert abs(orig_l - res_l) < 2  # L* should be nearly identical

    def test_lab_shift_preserves_alpha(self):
        crop = Image.new("RGBA", (50, 50), (200, 180, 160, 100))
        source_lab = np.array([180.0, 140.0, 145.0])
        target_lab = np.array([180.0, 120.0, 130.0])
        result = _lab_shift(crop, source_lab, target_lab, sat_threshold=15)
        assert np.array(result)[:, :, 3].mean() == 100

    def test_lab_shift_skips_desaturated(self):
        """Gray pixels should not be shifted."""
        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        arr[:, :, :3] = [120, 120, 120]
        arr[:, :, 3] = 255
        crop = Image.fromarray(arr, "RGBA")
        source_lab = np.array([128.0, 128.0, 128.0])
        target_lab = np.array([128.0, 160.0, 160.0])
        result = _lab_shift(crop, source_lab, target_lab, sat_threshold=15)
        np.testing.assert_array_equal(np.array(result)[:, :, :3], arr[:, :, :3])


class TestTintBlend:
    def test_tint_blend_shifts_toward_target(self):
        """Tint blend should move a* toward target."""
        crop = Image.new("RGBA", (50, 50), (200, 180, 180, 255))
        target_ab = np.array([160.0, 128.0])  # more red, neutral b*
        result = _tint_blend(crop, target_ab, strength=0.5)
        # a* should have moved toward 160
        res_bgr = cv2.cvtColor(np.array(result)[:, :, :3], cv2.COLOR_RGB2BGR)
        res_lab = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2LAB)
        orig_bgr = cv2.cvtColor(np.array(crop)[:, :, :3], cv2.COLOR_RGB2BGR)
        orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB)
        # Direction should be toward target
        assert res_lab[:, :, 1].mean() != orig_lab[:, :, 1].mean()

    def test_tint_blend_preserves_alpha(self):
        crop = Image.new("RGBA", (50, 50), (200, 180, 180, 77))
        result = _tint_blend(crop, np.array([160.0, 128.0]), strength=0.3)
        assert np.array(result)[:, :, 3].mean() == 77
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_apply.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement color transform helpers**

```python
# pipeline/color_apply.py
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
    """Rotate hue of saturated pixels from source_hue to target_hue.

    Preserves value (shading), alpha, and desaturated pixels (line art).
    Hue values in OpenCV scale (0-180).
    """
    arr = np.array(crop)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None
    rgb = arr[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)

    mask = hsv[:, :, 1] > sat_threshold
    delta = target_hue - source_hue
    hsv[:, :, 0][mask] = (hsv[:, :, 0][mask] + int(delta)) % 180

    hsv = hsv.astype(np.uint8)
    bgr_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    if alpha is not None:
        result = np.dstack([rgb_out, alpha])
        return Image.fromarray(result, "RGBA")
    return Image.fromarray(rgb_out, "RGB")


def _lab_shift(
    crop: Image.Image,
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    sat_threshold: int = 15,
) -> Image.Image:
    """Shift a* and b* channels from source to target LAB. Preserves L* exactly.

    Only affects saturated pixels (HSV S > sat_threshold).
    """
    arr = np.array(crop)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None
    rgb = arr[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Saturation mask
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] > sat_threshold

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_a = target_lab[1] - source_lab[1]
    delta_b = target_lab[2] - source_lab[2]

    lab[:, :, 1][mask] = np.clip(lab[:, :, 1][mask] + delta_a, 0, 255)
    lab[:, :, 2][mask] = np.clip(lab[:, :, 2][mask] + delta_b, 0, 255)

    bgr_out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    if alpha is not None:
        result = np.dstack([rgb_out, alpha])
        return Image.fromarray(result, "RGBA")
    return Image.fromarray(rgb_out, "RGB")


def _tint_blend(
    crop: Image.Image,
    target_ab: np.ndarray,
    strength: float = 0.3,
) -> Image.Image:
    """Gently blend a* and b* toward target values. For blush/cheek layers."""
    arr = np.array(crop)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else None
    rgb = arr[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    lab[:, :, 1] = lab[:, :, 1] * (1 - strength) + target_ab[0] * strength
    lab[:, :, 2] = lab[:, :, 2] * (1 - strength) + target_ab[1] * strength
    lab = np.clip(lab, 0, 255).astype(np.uint8)

    bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)

    if alpha is not None:
        result = np.dstack([rgb_out, alpha])
        return Image.fromarray(result, "RGBA")
    return Image.fromarray(rgb_out, "RGB")
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_apply.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/color_apply.py tests/pipeline/test_color_apply.py
git commit -m "feat: color transform helpers — hue_rotate, lab_shift, tint_blend"
```

---

### Task 4: Template palette extraction

**Files:**
- Create: `pipeline/template_palette.py`
- Create: `tests/pipeline/test_template_palette.py`

- [ ] **Step 1: Write failing test**

```python
# tests/pipeline/test_template_palette.py
"""Tests for template palette extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.template_palette import extract_template_palette


class TestTemplatePalette:
    def test_extract_from_hiyori(self):
        """Extract template palette from Hiyori atlas — should return valid palette."""
        from pipeline.atlas_config import load_atlas_config
        from pipeline.run import load_atlases
        from rig.config import RIG_HIYORI

        atlas_config = load_atlas_config(Path("manifests/hiyori_atlas.toml"))
        atlases = load_atlases(RIG_HIYORI)
        palette = extract_template_palette(atlases, atlas_config)

        assert palette.hair.shape == (3,)
        assert palette.skin.shape == (3,)
        assert 0 <= palette.eye_color <= 180
        assert palette.lip_color.shape == (3,)
        assert palette.clothing.shape == (3,)

    def test_hair_is_brown_ish(self):
        """Hiyori's hair is dark brown — hue should be in warm range."""
        from pipeline.atlas_config import load_atlas_config
        from pipeline.run import load_atlases
        from rig.config import RIG_HIYORI

        atlas_config = load_atlas_config(Path("manifests/hiyori_atlas.toml"))
        atlases = load_atlases(RIG_HIYORI)
        palette = extract_template_palette(atlases, atlas_config)

        # Brown hair in LAB: L moderate, a* slightly above 128, b* above 128
        assert palette.hair[0] > 30  # not totally black
        assert palette.hair[0] < 200  # not white
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_template_palette.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement template palette extraction**

```python
# pipeline/template_palette.py
"""Template palette extraction — measures baseline colors from unmodified atlas.

Used to compute deltas when applying portrait colors to atlas regions.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from pipeline.atlas_config import AtlasConfig
from pipeline.color_extract import ColorPalette


def _dominant_color_lab(
    atlas: Image.Image, regions: list, sat_threshold: int = 15
) -> np.ndarray:
    """Compute median LAB of saturated opaque pixels across region drawables."""
    all_pixels = []
    arr = np.array(atlas)
    for r in regions:
        x1, y1 = r.x, r.y
        x2, y2 = r.x + r.w, r.y + r.h
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # Only opaque pixels
        if crop.shape[2] == 4:
            opaque = crop[:, :, 3] > 10
        else:
            opaque = np.ones(crop.shape[:2], dtype=bool)
        rgb = crop[:, :, :3][opaque]
        if len(rgb) == 0:
            continue
        # Only saturated pixels
        bgr = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat_mask = hsv[0, :, 1] > sat_threshold
        if sat_mask.sum() == 0:
            continue
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        all_pixels.append(lab[0, sat_mask])

    if not all_pixels:
        return np.array([128.0, 128.0, 128.0])
    combined = np.concatenate(all_pixels, axis=0)
    return np.median(combined, axis=0).astype(np.float64)


def _dominant_hue_sat(
    atlas: Image.Image, regions: list, sat_threshold: int = 15
) -> tuple[float, float]:
    """Compute median HSV hue and saturation of saturated opaque pixels."""
    all_pixels = []
    arr = np.array(atlas)
    for r in regions:
        x1, y1 = r.x, r.y
        x2, y2 = r.x + r.w, r.y + r.h
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if crop.shape[2] == 4:
            opaque = crop[:, :, 3] > 10
        else:
            opaque = np.ones(crop.shape[:2], dtype=bool)
        rgb = crop[:, :, :3][opaque]
        if len(rgb) == 0:
            continue
        bgr = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat_mask = hsv[0, :, 1] > sat_threshold
        if sat_mask.sum() == 0:
            continue
        all_pixels.append(hsv[0, sat_mask])

    if not all_pixels:
        return 0.0, 0.0
    combined = np.concatenate(all_pixels, axis=0)
    return float(np.median(combined[:, 0])), float(np.median(combined[:, 1]))


def extract_template_palette(
    atlases: dict[int, Image.Image] | list[Image.Image],
    atlas_config: AtlasConfig,
) -> ColorPalette:
    """Measure baseline colors from the unmodified template atlas."""
    if isinstance(atlases, list):
        atlas_map = {i: a for i, a in enumerate(atlases)}
    else:
        atlas_map = atlases

    def _get_regions(name: str):
        if atlas_config.has(name):
            r = atlas_config.get(name)
            return r.texture_index, [r]
        return 0, []

    def _get_all_regions(*names: str):
        result = []
        tex_idx = 0
        for name in names:
            if atlas_config.has(name):
                r = atlas_config.get(name)
                result.append(r)
                tex_idx = r.texture_index
        return tex_idx, result

    # Hair — from hair_front (most representative)
    hair_regions = []
    for name in ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]:
        if atlas_config.has(name):
            hair_regions.append(atlas_config.get(name))
    if hair_regions:
        hair = _dominant_color_lab(atlas_map[hair_regions[0].texture_index], hair_regions)
        hair_hue, _ = _dominant_hue_sat(atlas_map[hair_regions[0].texture_index], hair_regions)
    else:
        hair = np.array([128.0, 128.0, 128.0])
        hair_hue = 0.0

    # Skin — from face_skin
    skin_regions = []
    if atlas_config.has("face_skin"):
        skin_regions.append(atlas_config.get("face_skin"))
    skin = _dominant_color_lab(atlas_map[0], skin_regions) if skin_regions else np.array([128.0, 128.0, 128.0])

    # Eyes — from left_eye and right_eye
    eye_regions = []
    for name in ["left_eye", "right_eye"]:
        if atlas_config.has(name):
            eye_regions.append(atlas_config.get(name))
    if eye_regions:
        eye_hue, eye_sat = _dominant_hue_sat(atlas_map[eye_regions[0].texture_index], eye_regions)
    else:
        eye_hue, eye_sat = 0.0, 0.0

    # Lips — from mouth
    mouth_regions = []
    if atlas_config.has("mouth"):
        mouth_regions.append(atlas_config.get("mouth"))
    lip_color = _dominant_color_lab(atlas_map[0], mouth_regions) if mouth_regions else np.array([128.0, 128.0, 128.0])

    # Clothing
    clothing_regions = []
    if atlas_config.has("clothing"):
        clothing_regions.append(atlas_config.get("clothing"))
    if clothing_regions:
        tex_idx = clothing_regions[0].texture_index
        clothing = _dominant_color_lab(atlas_map[tex_idx], clothing_regions)
    else:
        clothing = np.array([128.0, 128.0, 128.0])

    return ColorPalette(
        hair=hair,
        skin=skin,
        eye_color=eye_hue,
        eye_saturation=eye_sat,
        lip_color=lip_color,
        clothing=clothing,
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_template_palette.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/template_palette.py tests/pipeline/test_template_palette.py
git commit -m "feat: template palette extraction — measure baseline atlas colors"
```

---

### Task 5: `recolor_atlas()` — apply palette to atlas regions

**Files:**
- Modify: `pipeline/color_apply.py`
- Modify: `tests/pipeline/test_color_apply.py`

- [ ] **Step 1: Write failing test for recolor_atlas**

Add to `tests/pipeline/test_color_apply.py`:

```python
from pathlib import Path
from pipeline.color_apply import recolor_atlas
from pipeline.color_extract import ColorPalette


class TestRecolorAtlas:
    def test_recolor_returns_same_number_of_atlases(self):
        """Output should have same number of textures as input."""
        from pipeline.atlas_config import load_atlas_config
        from pipeline.run import load_atlases
        from pipeline.template_palette import extract_template_palette
        from rig.config import RIG_HIYORI

        atlas_config = load_atlas_config(Path("manifests/hiyori_atlas.toml"))
        atlases = load_atlases(RIG_HIYORI)
        template_palette = extract_template_palette(atlases, atlas_config)

        # Fake palette with different colors
        portrait_palette = ColorPalette(
            hair=np.array([40.0, 128.0, 100.0]),     # cool dark hair
            skin=np.array([170.0, 140.0, 145.0]),     # warm skin
            eye_color=100.0,                           # blue eyes
            eye_saturation=180.0,
            lip_color=np.array([140.0, 155.0, 135.0]), # pink lips
            clothing=np.array([100.0, 110.0, 160.0]),  # blue clothing
        )

        result = recolor_atlas(atlases, portrait_palette, atlas_config, template_palette)
        assert len(result) == len(atlases)
        for idx in atlases:
            assert result[idx].size == atlases[idx].size
            assert result[idx].mode == atlases[idx].mode

    def test_recolor_changes_pixels(self):
        """Recolored atlas should differ from original (unless palette matches)."""
        from pipeline.atlas_config import load_atlas_config
        from pipeline.run import load_atlases
        from pipeline.template_palette import extract_template_palette
        from rig.config import RIG_HIYORI

        atlas_config = load_atlas_config(Path("manifests/hiyori_atlas.toml"))
        atlases = load_atlases(RIG_HIYORI)
        template_palette = extract_template_palette(atlases, atlas_config)

        portrait_palette = ColorPalette(
            hair=np.array([40.0, 128.0, 100.0]),
            skin=np.array([170.0, 140.0, 145.0]),
            eye_color=100.0,
            eye_saturation=180.0,
            lip_color=np.array([140.0, 155.0, 135.0]),
            clothing=np.array([100.0, 110.0, 160.0]),
        )

        result = recolor_atlas(atlases, portrait_palette, atlas_config, template_palette)
        # At least one texture should be different
        any_changed = False
        for idx in atlases:
            orig = np.array(atlases[idx])
            recolored = np.array(result[idx])
            if not np.array_equal(orig, recolored):
                any_changed = True
                break
        assert any_changed, "Recoloring should change at least some pixels"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_apply.py::TestRecolorAtlas -v`
Expected: FAIL — `ImportError: cannot import name 'recolor_atlas'`

- [ ] **Step 3: Implement recolor_atlas**

Add to `pipeline/color_apply.py`:

```python
from pipeline.atlas_config import AtlasConfig
from pipeline.color_extract import ColorPalette


# Region → recoloring strategy mapping
_HAIR_REGIONS = ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]
_EYE_REGIONS = ["left_eye", "right_eye"]
_EYEBROW_REGIONS = ["left_eyebrow", "right_eyebrow"]
_SKIN_REGIONS = ["face_skin", "body"]
_CHEEK_REGIONS = ["left_cheek", "right_cheek"]
_MOUTH_REGIONS = ["mouth"]
_CLOTHING_REGIONS = ["clothing"]
_MIXED_REGIONS = ["cloth_and_body"]


def _crop_region(atlas: Image.Image, region) -> Image.Image:
    """Crop a region from the atlas, preserving RGBA."""
    return atlas.crop((region.x, region.y, region.x + region.w, region.y + region.h)).copy()


def _paste_region(atlas: Image.Image, crop: Image.Image, region) -> Image.Image:
    """Paste a crop back into the atlas at region coordinates."""
    out = atlas.copy()
    out.paste(crop, (region.x, region.y))
    return out


def _lab_hue_of(lab: np.ndarray) -> float:
    """Convert LAB color to approximate HSV hue for delta computation."""
    lab_pixel = lab.reshape(1, 1, 3).astype(np.uint8)
    bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[0, 0, 0])


def recolor_atlas(
    atlases: dict[int, Image.Image],
    palette: ColorPalette,
    atlas_config: AtlasConfig,
    template_palette: ColorPalette | None = None,
) -> dict[int, Image.Image]:
    """Apply portrait-extracted colors to all atlas regions.

    Returns new dict of modified atlas images. Does not modify inputs.
    """
    if template_palette is None:
        from pipeline.template_palette import extract_template_palette
        template_palette = extract_template_palette(atlases, atlas_config)

    result = {idx: atlas.copy() for idx, atlas in atlases.items()}

    # Hair regions — hue rotation
    src_hair_hue = _lab_hue_of(template_palette.hair)
    tgt_hair_hue = _lab_hue_of(palette.hair)
    for name in _HAIR_REGIONS + _EYEBROW_REGIONS:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        crop = _crop_region(result[region.texture_index], region)
        crop = _hue_rotate(crop, src_hair_hue, tgt_hair_hue)
        result[region.texture_index] = _paste_region(result[region.texture_index], crop, region)

    # Eye regions — hue rotation to iris color
    src_eye_hue = template_palette.eye_color
    tgt_eye_hue = palette.eye_color
    for name in _EYE_REGIONS:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        crop = _crop_region(result[region.texture_index], region)
        crop = _hue_rotate(crop, src_eye_hue, tgt_eye_hue)
        result[region.texture_index] = _paste_region(result[region.texture_index], crop, region)

    # Skin regions — LAB a*b* shift
    for name in _SKIN_REGIONS:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        crop = _crop_region(result[region.texture_index], region)
        crop = _lab_shift(crop, template_palette.skin, palette.skin)
        result[region.texture_index] = _paste_region(result[region.texture_index], crop, region)

    # Cheek regions — tint blend
    skin_bgr = cv2.cvtColor(
        palette.skin.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR
    )
    skin_lab = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_ab = np.array([skin_lab[0, 0, 1], skin_lab[0, 0, 2]])
    for name in _CHEEK_REGIONS:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        crop = _crop_region(result[region.texture_index], region)
        crop = _tint_blend(crop, target_ab, strength=0.3)
        result[region.texture_index] = _paste_region(result[region.texture_index], crop, region)

    # Mouth regions — LAB shift with lip color
    for name in _MOUTH_REGIONS:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        crop = _crop_region(result[region.texture_index], region)
        crop = _lab_shift(crop, template_palette.lip_color, palette.lip_color)
        result[region.texture_index] = _paste_region(result[region.texture_index], crop, region)

    # Clothing regions — hue rotation
    src_cloth_hue = _lab_hue_of(template_palette.clothing)
    tgt_cloth_hue = _lab_hue_of(palette.clothing)
    for name in _CLOTHING_REGIONS:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        crop = _crop_region(result[region.texture_index], region)
        crop = _hue_rotate(crop, src_cloth_hue, tgt_cloth_hue)
        result[region.texture_index] = _paste_region(result[region.texture_index], crop, region)

    # Mixed cloth_and_body — apply clothing hue rotation (skin is minority)
    for name in _MIXED_REGIONS:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        crop = _crop_region(result[region.texture_index], region)
        crop = _hue_rotate(crop, src_cloth_hue, tgt_cloth_hue)
        result[region.texture_index] = _paste_region(result[region.texture_index], crop, region)

    return result
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_apply.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/color_apply.py tests/pipeline/test_color_apply.py
git commit -m "feat: recolor_atlas() — apply portrait colors to all atlas regions"
```

---

### Task 6: Visual integration test script

**Files:**
- Create: `scripts/prototype_color_recolor.py`

- [ ] **Step 1: Write the prototype script**

```python
# scripts/prototype_color_recolor.py
"""Prototype: color extraction + atlas recoloring — visual comparison.

Extracts colors from a portrait, recolors the Hiyori atlas, renders,
and saves a side-by-side comparison grid.

Run: PYTHONPATH=. uv run python scripts/prototype_color_recolor.py
     PYTHONPATH=. uv run python scripts/prototype_color_recolor.py --portrait assets/data/image1.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.atlas_config import load_atlas_config
from pipeline.color_apply import recolor_atlas
from pipeline.color_extract import ColorPalette, extract_palette
from pipeline.package import package_output
from pipeline.run import load_atlases
from pipeline.template_palette import extract_template_palette
from rig.config import RIG_HIYORI, RigConfig
from rig.render import RigRenderer

OUT = Path("test_output/color_recolor")
ATLAS_TOML = Path("manifests/hiyori_atlas.toml")


def make_comparison(original: np.ndarray, recolored: np.ndarray) -> Image.Image:
    """Side-by-side comparison image."""
    h, w = original.shape[:2]
    grid = Image.new("RGBA", (w * 2 + 20, h + 40), (30, 30, 30, 255))
    draw = ImageDraw.Draw(grid)
    draw.text((w // 2 - 20, 5), "Original", fill="white")
    draw.text((w + 20 + w // 2 - 20, 5), "Recolored", fill="white")
    grid.paste(Image.fromarray(original), (0, 30))
    grid.paste(Image.fromarray(recolored), (w + 20, 30))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", default="assets/data/image1.png")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # Load portrait
    portrait = Image.open(args.portrait).convert("RGB")
    print(f"Portrait: {portrait.size}")

    # Extract palette
    print("Extracting color palette...")
    palette = extract_palette(portrait)
    print(f"  Hair LAB: {palette.hair}")
    print(f"  Skin LAB: {palette.skin}")
    print(f"  Eye hue: {palette.eye_color:.1f}, sat: {palette.eye_saturation:.1f}")
    print(f"  Lip LAB: {palette.lip_color}")
    print(f"  Clothing LAB: {palette.clothing}")

    # Save palette
    palette_path = OUT / "palette.json"
    with open(palette_path, "w") as f:
        json.dump(palette.to_dict(), f, indent=2)
    print(f"  Saved palette to {palette_path}")

    # Load atlas + template palette
    atlas_config = load_atlas_config(ATLAS_TOML)
    atlases = load_atlases(RIG_HIYORI)
    template_palette = extract_template_palette(atlases, atlas_config)
    print(f"\nTemplate palette:")
    print(f"  Hair LAB: {template_palette.hair}")
    print(f"  Skin LAB: {template_palette.skin}")
    print(f"  Eye hue: {template_palette.eye_color:.1f}")

    # Recolor
    print("\nRecoloring atlas...")
    recolored = recolor_atlas(atlases, palette, atlas_config, template_palette)

    # Save recolored textures
    for idx, img in recolored.items():
        img.save(OUT / f"texture_{idx:02d}_recolored.png")
    for idx, img in atlases.items():
        img.save(OUT / f"texture_{idx:02d}_original.png")
    print("  Saved textures")

    # Package + render recolored
    pkg_dir = OUT / "pkg_recolored"
    package_output(RIG_HIYORI, recolored, pkg_dir)

    output_rig = RigConfig(
        name="hiyori_recolored",
        model_dir=pkg_dir,
        moc3_path=pkg_dir / "hiyori.moc3",
        model3_json_path=pkg_dir / "hiyori.model3.json",
        textures=[pkg_dir / "hiyori.2048" / f"texture_{i:02d}.png" for i in range(2)],
        param_ids=RIG_HIYORI.param_ids,
    )

    with RigRenderer(output_rig, width=512, height=512) as renderer:
        render_new = renderer.render()
        Image.fromarray(render_new).save(OUT / "render_recolored.png")

    # Render original
    with RigRenderer(RIG_HIYORI, width=512, height=512) as renderer:
        render_orig = renderer.render()
        Image.fromarray(render_orig).save(OUT / "render_original.png")

    # Comparison
    grid = make_comparison(render_orig, render_new)
    grid.save(OUT / "comparison.png")
    print(f"\nComparison saved to {OUT / 'comparison.png'}")
    print(f"All outputs in {OUT}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the prototype**

Run: `PYTHONPATH=. uv run python scripts/prototype_color_recolor.py`
Expected: Prints palette values, saves comparison grid. Inspect `test_output/color_recolor/comparison.png` visually.

- [ ] **Step 3: Commit**

```bash
git add scripts/prototype_color_recolor.py
git commit -m "feat: visual prototype for color extraction + atlas recoloring"
```

---

### Task 7: Pipeline integration

**Files:**
- Modify: `pipeline/run.py`

- [ ] **Step 1: Read current pipeline/run.py to understand integration point**

Read `pipeline/run.py` and identify where to add the color extraction + recoloring step. It should go between `load_atlases()` and `package_output()`, replacing or supplementing the ComfyUI-based texture generation.

- [ ] **Step 2: Add color-based recoloring as an alternative pipeline path**

Add a `--mode color` option to `pipeline/run.py` that uses color extraction instead of ComfyUI-based texture generation. The exact integration depends on the current structure of `run.py`, but the pattern is:

```python
# In the main pipeline function, after loading the portrait:
if mode == "color":
    from pipeline.atlas_config import load_atlas_config
    from pipeline.color_apply import recolor_atlas
    from pipeline.color_extract import extract_palette
    from pipeline.template_palette import extract_template_palette

    palette = extract_palette(portrait)
    atlas_config = load_atlas_config(atlas_toml_path)
    atlases = load_atlases(rig_config)
    template_palette = extract_template_palette(atlases, atlas_config)
    modified_atlases = recolor_atlas(atlases, palette, atlas_config, template_palette)
    package_output(rig_config, modified_atlases, output_dir)
```

- [ ] **Step 3: Run the full pipeline**

Run: `PYTHONPATH=. uv run python -m pipeline.run --portrait assets/data/image1.png --mode color --output test_output/color_pipeline`
Expected: Completes without error, produces packaged model in output directory.

- [ ] **Step 4: Commit**

```bash
git add pipeline/run.py
git commit -m "feat: integrate color extraction + recoloring into pipeline CLI"
```

---

### Task 8: Add `facer` dependency (optional face parsing)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add facer as optional dependency**

```bash
uv add --optional face-parsing facer
```

If `facer` has issues (heavy dependencies, version conflicts), the pipeline already has a fallback (bbox-based sampling), so this is truly optional. Document in the commit.

- [ ] **Step 2: Verify extract_palette works both with and without facer**

Run: `PYTHONPATH=. uv run pytest tests/pipeline/test_color_extract.py -v`
Expected: ALL PASS regardless of whether facer is installed.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add facer as optional dependency for face parsing"
```
