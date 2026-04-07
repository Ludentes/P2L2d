"""Template palette extraction -- measure baseline colors from unmodified atlas textures.

Given the atlas textures and region config, extracts a ColorPalette representing the
template's default colors (hair, skin, eyes, lips, clothing). Used as the "from" palette
when recoloring atlas textures to match a portrait photo.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion
from pipeline.color_extract import ColorPalette


def _crop_region_pixels(atlas: Image.Image, region: AtlasRegion) -> np.ndarray:
    """Crop a region from the atlas and return as BGR+A numpy array.

    Returns array of shape (N, 4) with columns [B, G, R, A].
    """
    crop = atlas.crop((region.x, region.y, region.x + region.w, region.y + region.h))
    rgba = np.array(crop)  # (H, W, 4) in RGBA order
    # Reshape to (N, 4)
    pixels = rgba.reshape(-1, 4)
    # Convert RGB→BGR for OpenCV, keep alpha
    bgra = pixels[:, [2, 1, 0, 3]]
    return bgra


def _filter_opaque_saturated(bgra: np.ndarray, sat_threshold: int = 15) -> np.ndarray:
    """Filter to opaque (alpha > 10) and saturated (HSV S > sat_threshold) pixels.

    Args:
        bgra: (N, 4) array with [B, G, R, A] columns.
        sat_threshold: minimum HSV saturation to keep.

    Returns:
        (M, 3) BGR array of filtered pixels.
    """
    if len(bgra) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # Filter by alpha
    opaque_mask = bgra[:, 3] > 10
    bgr_opaque = bgra[opaque_mask][:, :3]

    if len(bgr_opaque) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # Convert to HSV to check saturation
    # cv2.cvtColor needs (N, 1, 3) shape
    bgr_3d = bgr_opaque.reshape(-1, 1, 3).astype(np.uint8)
    hsv_3d = cv2.cvtColor(bgr_3d, cv2.COLOR_BGR2HSV)
    hsv = hsv_3d.reshape(-1, 3)

    sat_mask = hsv[:, 1] > sat_threshold
    return bgr_opaque[sat_mask]


def _collect_pixels(
    atlases: dict[int, Image.Image],
    atlas_config: AtlasConfig,
    region_names: list[str],
    sat_threshold: int = 15,
) -> np.ndarray:
    """Collect filtered pixels from multiple atlas regions.

    Returns (M, 3) BGR array of all valid pixels concatenated.
    """
    all_pixels = []
    for name in region_names:
        if not atlas_config.has(name):
            continue
        region = atlas_config.get(name)
        atlas = atlases[region.texture_index]
        bgra = _crop_region_pixels(atlas, region)
        filtered = _filter_opaque_saturated(bgra, sat_threshold)
        if len(filtered) > 0:
            all_pixels.append(filtered)

    if not all_pixels:
        return np.empty((0, 3), dtype=np.uint8)
    return np.concatenate(all_pixels, axis=0)


def _dominant_color_lab(
    atlases: dict[int, Image.Image],
    atlas_config: AtlasConfig,
    region_names: list[str],
    sat_threshold: int = 15,
) -> np.ndarray:
    """Compute median LAB color from multiple atlas regions.

    Filters to opaque and saturated pixels, then returns median [L, a, b].
    Falls back to [128, 128, 128] if no valid pixels.
    """
    bgr_pixels = _collect_pixels(atlases, atlas_config, region_names, sat_threshold)
    if len(bgr_pixels) == 0:
        return np.array([128.0, 128.0, 128.0])

    bgr_3d = bgr_pixels.reshape(-1, 1, 3).astype(np.uint8)
    lab_3d = cv2.cvtColor(bgr_3d, cv2.COLOR_BGR2LAB)
    lab = lab_3d.reshape(-1, 3)

    return np.median(lab, axis=0).astype(np.float64)


def _dominant_hue_sat(
    atlases: dict[int, Image.Image],
    atlas_config: AtlasConfig,
    region_names: list[str],
    sat_threshold: int = 15,
) -> tuple[float, float]:
    """Compute median HSV hue and saturation from multiple atlas regions.

    Filters to opaque and saturated pixels, then returns (hue, saturation).
    Falls back to (0.0, 0.0) if no valid pixels.
    """
    bgr_pixels = _collect_pixels(atlases, atlas_config, region_names, sat_threshold)
    if len(bgr_pixels) == 0:
        return 0.0, 0.0

    bgr_3d = bgr_pixels.reshape(-1, 1, 3).astype(np.uint8)
    hsv_3d = cv2.cvtColor(bgr_3d, cv2.COLOR_BGR2HSV)
    hsv = hsv_3d.reshape(-1, 3)

    return float(np.median(hsv[:, 0])), float(np.median(hsv[:, 1]))


def extract_template_palette(
    atlases: dict[int, Image.Image],
    atlas_config: AtlasConfig,
) -> ColorPalette:
    """Measure baseline colors from the unmodified template atlas.

    For each color category, crop the relevant regions from the atlas,
    filter to opaque (alpha > 10) and saturated (HSV S > 15) pixels,
    and compute the median color.

    Hair/skin/lip/clothing measured as LAB. Eye colors measured as HSV hue + saturation.
    """
    hair_regions = ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]
    skin_regions = ["face_skin"]
    eye_regions = ["left_eye", "right_eye"]
    mouth_regions = ["mouth"]
    clothing_regions = ["clothing"]

    hair_lab = _dominant_color_lab(atlases, atlas_config, hair_regions)
    skin_lab = _dominant_color_lab(atlases, atlas_config, skin_regions)
    eye_hue, eye_sat = _dominant_hue_sat(atlases, atlas_config, eye_regions)
    lip_lab = _dominant_color_lab(atlases, atlas_config, mouth_regions)
    clothing_lab = _dominant_color_lab(atlases, atlas_config, clothing_regions)

    return ColorPalette(
        hair=hair_lab,
        skin=skin_lab,
        eye_color=eye_hue,
        eye_saturation=eye_sat,
        lip_color=lip_lab,
        clothing=clothing_lab,
    )
