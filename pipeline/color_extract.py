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


# ---------------------------------------------------------------------------
# Region sampling & face parsing
# ---------------------------------------------------------------------------

def _sample_region_by_bbox(
    img_bgr: np.ndarray, landmarks: np.ndarray, region: str,
) -> np.ndarray:
    """Fallback: sample from approximate portrait regions when no face parser available.

    'hair': median LAB from top 20% of image above face top (between face_left and face_right).
    'clothing': median LAB from bottom 30% below face bottom.
    Returns median LAB array of shape (3,).
    """
    h, w = img_bgr.shape[:2]
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    face_left = int(np.min(xs))
    face_right = int(np.max(xs))
    face_top = int(np.min(ys))
    face_bottom = int(np.max(ys))

    if region == "hair":
        y0 = max(0, face_top - int(0.2 * h))
        y1 = face_top
        x0 = max(0, face_left)
        x1 = min(w, face_right)
    elif region == "clothing":
        y0 = face_bottom
        y1 = min(h, face_bottom + int(0.3 * h))
        x0 = max(0, face_left)
        x1 = min(w, face_right)
    else:
        return np.array([128.0, 128.0, 128.0])

    if y1 <= y0 or x1 <= x0:
        return np.array([128.0, 128.0, 128.0])

    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([128.0, 128.0, 128.0])

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3)
    return np.median(pixels, axis=0).astype(np.float64)


def _try_face_parsing(img_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
    """Try BiSeNet face parsing via facer package. Returns dict of masks or None.

    Returns dict with keys 'skin', 'hair', 'clothing', 'upper_lip', 'lower_lip'
    each being a uint8 mask array (H, W), or None if parsing fails.
    """
    try:
        import facer  # noqa: F811
        import torch
    except ImportError:
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # facer expects RGB HWC uint8
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        face_detector = facer.face_detector("retinaface/mobilenet", device=device)
        faces = face_detector(img_tensor)
        if faces is None or len(faces["scores"]) == 0:
            return None

        face_parser = facer.face_parser("farl/lapa/448", device=device)
        faces = face_parser(img_tensor, faces)
        seg_logits = faces["seg"]["logits"]  # (1, n_classes, H, W)
        seg = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # LaPa label map: 0=bg, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
        # 6=nose, 7=u_lip, 8=mouth, 9=l_lip, 10=hair, 11=...
        masks = {
            "skin": ((seg == 1) | (seg == 6)).astype(np.uint8) * 255,
            "hair": (seg == 10).astype(np.uint8) * 255,
            "upper_lip": (seg == 7).astype(np.uint8) * 255,
            "lower_lip": (seg == 9).astype(np.uint8) * 255,
            "clothing": (seg == 0).astype(np.uint8) * 255,  # approximate: background/clothing
        }
        return masks
    except Exception:
        return None


def _sample_masked_lab(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median LAB of pixels where mask > 0.

    Returns np.ndarray of shape (3,) with [L, a, b] float64.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]
    if len(pixels) == 0:
        return np.array([128.0, 128.0, 128.0])
    return np.median(pixels, axis=0).astype(np.float64)


def _sample_clothing_lab(
    img_bgr: np.ndarray,
    mask_or_none: np.ndarray | None,
    landmarks: np.ndarray,
) -> np.ndarray:
    """Extract dominant clothing color. Uses face parsing mask if available, else bbox fallback.

    If mask has enough pixels (>100), use k-means (k=2) to find dominant cluster.
    Otherwise fall back to _sample_region_by_bbox.
    Returns np.ndarray of shape (3,) with [L, a, b] float64.
    """
    if mask_or_none is not None and np.count_nonzero(mask_or_none) > 100:
        from sklearn.cluster import KMeans

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab[mask_or_none > 0].astype(np.float64)

        k = min(2, len(pixels))
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
        kmeans.fit(pixels)

        # Pick the cluster with most members
        labels = kmeans.labels_
        counts = np.bincount(labels, minlength=k)
        dominant = np.argmax(counts)
        return kmeans.cluster_centers_[dominant].astype(np.float64)

    return _sample_region_by_bbox(img_bgr, landmarks, "clothing")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_palette(portrait: "Image.Image") -> ColorPalette:
    """Extract color palette from portrait using landmarks + optional face parsing.

    Steps:
    1. Convert to BGR, detect landmarks
    2. Try face parsing (returns None if facer not installed)
    3. Hair: face parsing mask -> median LAB, or fallback to bbox
    4. Skin: median of _sample_patch_lab at cheek landmark points
    5. Eyes: average of _sample_iris_hsv on left and right iris landmarks
    6. Lips: _sample_lip_lab on inner lip polygon
    7. Clothing: _sample_clothing_lab with parsing mask or bbox fallback
    """
    from PIL import Image

    from pipeline.face_align import detect_landmarks

    # 1. Convert to BGR numpy, detect landmarks
    img_rgb = np.array(portrait.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    pil_image = portrait.convert("RGB") if portrait.mode != "RGB" else portrait
    landmarks = detect_landmarks(pil_image)  # (478, 2) float32 pixel coords

    # 2. Try face parsing
    parsing = _try_face_parsing(img_bgr)

    # 3. Hair
    if parsing is not None and np.count_nonzero(parsing["hair"]) > 100:
        hair = _sample_masked_lab(img_bgr, parsing["hair"])
    else:
        hair = _sample_region_by_bbox(img_bgr, landmarks, "hair")

    # 4. Skin — median of cheek patch samples
    cheek_indices = _CHEEK_LEFT + _CHEEK_RIGHT
    skin_samples = []
    for idx in cheek_indices:
        x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
        skin_samples.append(_sample_patch_lab(img_bgr, x, y))
    skin = np.median(np.array(skin_samples), axis=0).astype(np.float64)

    # 5. Eyes — average of left and right iris HSV
    iris_left = landmarks[_IRIS_LEFT]   # (5, 2)
    iris_right = landmarks[_IRIS_RIGHT]  # (5, 2)
    hue_l, sat_l = _sample_iris_hsv(img_bgr, iris_left)
    hue_r, sat_r = _sample_iris_hsv(img_bgr, iris_right)
    eye_color = (hue_l + hue_r) / 2.0
    eye_saturation = (sat_l + sat_r) / 2.0

    # 6. Lips
    lip_lm = landmarks[_INNER_LIP]  # (20, 2)
    lip_color = _sample_lip_lab(img_bgr, lip_lm)

    # 7. Clothing
    clothing_mask = parsing["clothing"] if parsing is not None else None
    clothing = _sample_clothing_lab(img_bgr, clothing_mask, landmarks)

    return ColorPalette(
        hair=hair,
        skin=skin,
        eye_color=eye_color,
        eye_saturation=eye_saturation,
        lip_color=lip_color,
        clothing=clothing,
    )
