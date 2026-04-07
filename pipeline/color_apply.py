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

from pipeline.atlas_config import AtlasConfig, AtlasRegion
from pipeline.color_extract import ColorPalette


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


# ---------------------------------------------------------------------------
# Region groupings
# ---------------------------------------------------------------------------

_HAIR_REGIONS = ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]
_EYE_REGIONS = ["left_eye", "right_eye"]
_EYEBROW_REGIONS = ["left_eyebrow", "right_eyebrow"]
_SKIN_REGIONS = ["face_skin", "body"]
_CHEEK_REGIONS = ["left_cheek", "right_cheek"]
_MOUTH_REGIONS = ["mouth"]
_CLOTHING_REGIONS = ["clothing"]
_MIXED_REGIONS = ["cloth_and_body"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crop_region(atlas: Image.Image, region: AtlasRegion) -> Image.Image:
    """Crop region from atlas, preserving RGBA."""
    return atlas.crop((region.x, region.y, region.x + region.w, region.y + region.h)).copy()


def _paste_region(atlas: Image.Image, crop: Image.Image, region: AtlasRegion) -> Image.Image:
    """Paste crop back into atlas at region coords. Returns new image."""
    out = atlas.copy()
    out.paste(crop, (region.x, region.y))
    return out


def _lab_hue_of(lab: np.ndarray) -> float:
    """Convert LAB color to approximate HSV hue."""
    lab_pixel = lab.reshape(1, 1, 3).astype(np.uint8)
    bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[0, 0, 0])


# ---------------------------------------------------------------------------
# Main recoloring entry point
# ---------------------------------------------------------------------------


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

    # Pre-compute hues from LAB palettes
    src_hair_hue = _lab_hue_of(template_palette.hair)
    tgt_hair_hue = _lab_hue_of(palette.hair)
    src_clothing_hue = _lab_hue_of(template_palette.clothing)
    tgt_clothing_hue = _lab_hue_of(palette.clothing)

    def _apply(region_names: list[str], fn):
        """Apply *fn* to each region that exists in atlas_config."""
        nonlocal result
        for name in region_names:
            if not atlas_config.has(name):
                continue
            region = atlas_config.get(name)
            crop = _crop_region(result[region.texture_index], region)
            transformed = fn(crop)
            result[region.texture_index] = _paste_region(
                result[region.texture_index], transformed, region,
            )

    # Hair + Eyebrows: hue rotation
    _apply(
        _HAIR_REGIONS + _EYEBROW_REGIONS,
        lambda crop: _hue_rotate(crop, src_hair_hue, tgt_hair_hue),
    )

    # Eyes: hue rotation
    _apply(
        _EYE_REGIONS,
        lambda crop: _hue_rotate(crop, template_palette.eye_color, palette.eye_color),
    )

    # Skin: LAB shift
    _apply(
        _SKIN_REGIONS,
        lambda crop: _lab_shift(crop, template_palette.skin, palette.skin),
    )

    # Cheeks: tint blend toward portrait skin a*b*
    cheek_target_ab = palette.skin[1:3]  # a*, b* from portrait skin
    _apply(
        _CHEEK_REGIONS,
        lambda crop: _tint_blend(crop, cheek_target_ab, strength=0.3),
    )

    # Mouth: LAB shift
    _apply(
        _MOUTH_REGIONS,
        lambda crop: _lab_shift(crop, template_palette.lip_color, palette.lip_color),
    )

    # Clothing: hue rotation
    _apply(
        _CLOTHING_REGIONS,
        lambda crop: _hue_rotate(crop, src_clothing_hue, tgt_clothing_hue),
    )

    # Mixed cloth_and_body: hue rotation with clothing hues
    _apply(
        _MIXED_REGIONS,
        lambda crop: _hue_rotate(crop, src_clothing_hue, tgt_clothing_hue),
    )

    return result
