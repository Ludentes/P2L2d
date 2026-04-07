# pipeline/texture_gen.py
"""Texture generation orchestrator — Phase 1 pipeline.

Approach: detect face bounding box from MediaPipe landmarks, style-transfer
the portrait, then crop+resize face regions to match each atlas drawable's
dimensions.  No affine warp to template space — we work directly in portrait
and atlas coordinate systems.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from comfyui.client import ComfyUIClient
from pipeline.atlas_config import AtlasConfig
from pipeline.face_align import detect_landmarks
from pipeline.face_inpaint import inpaint_face_skin
from pipeline.hair_segment import segment_hair
from pipeline.style_transfer import load_texture_gen_config, stylize_portrait
from rig.config import RigConfig

_FACE_REGIONS = [
    "face_skin", "left_eye", "right_eye",
    "left_eyebrow", "right_eyebrow", "mouth",
    "left_cheek", "right_cheek",
]
_HAIR_REGIONS = ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]
_INPAINT_SIZE = (512, 512)


def _face_bbox(landmarks: np.ndarray, margin: float = 0.15) -> tuple[int, int, int, int]:
    """Compute face bounding box from landmarks with margin.

    Returns (x, y, x2, y2) in pixel coordinates.
    """
    x_min, y_min = landmarks.min(axis=0)
    x_max, y_max = landmarks.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    # Expand by margin
    half_w = w * (1 + margin) / 2
    half_h = h * (1 + margin) / 2
    return (
        int(cx - half_w),
        int(cy - half_h),
        int(cx + half_w),
        int(cy + half_h),
    )


def _crop_face_for_region(
    stylized: Image.Image,
    face_bbox: tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> Image.Image:
    """Crop face area from stylized portrait and resize to target dimensions."""
    x1, y1, x2, y2 = face_bbox
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(stylized.width, x2)
    y2 = min(stylized.height, y2)
    crop = stylized.crop((x1, y1, x2, y2))
    return crop.resize((target_w, target_h), Image.Resampling.LANCZOS)


async def generate_textures(
    portrait: Image.Image,
    atlas_cfg: AtlasConfig,
    rig: RigConfig,
    client: ComfyUIClient,
    template_name: str = "humanoid-anime",
) -> dict[str, Image.Image]:
    """Generate per-region textures from portrait.

    Returns {region_name: PIL.Image} for all face + hair regions present in atlas_cfg.
    Skips regions absent from atlas_cfg without error.
    """
    cfg = load_texture_gen_config(template_name)

    # 1. Detect face bbox on original portrait
    landmarks = detect_landmarks(portrait)
    face_bbox = _face_bbox(landmarks)

    # 2. Style transfer
    stylized = await stylize_portrait(
        portrait, style=cfg.style_transfer, model=cfg.style_model,
        strength=cfg.style_strength, client=client,
    )

    # 3. Crop face regions — resize face crop to each region's atlas dimensions
    replacements: dict[str, Image.Image] = {}
    for region_name in _FACE_REGIONS:
        if region_name == "face_skin" or not atlas_cfg.has(region_name):
            continue
        region = atlas_cfg.get(region_name)
        replacements[region_name] = _crop_face_for_region(
            stylized, face_bbox, region.w, region.h,
        )

    # 4. Inpaint face skin (clean skin without facial features)
    if atlas_cfg.has("face_skin"):
        region = atlas_cfg.get("face_skin")
        face_crop = _crop_face_for_region(stylized, face_bbox, region.w, region.h)
        # Build a simple center-area mask for inpainting (features are in center)
        mask = _build_simple_inpaint_mask(region.w, region.h)
        # Resize both to inpaint size for ComfyUI
        face_large = face_crop.resize(_INPAINT_SIZE, Image.Resampling.LANCZOS)
        mask_large = mask.resize(_INPAINT_SIZE, Image.Resampling.NEAREST)
        inpainted = await inpaint_face_skin(face_large, mask_large, client)
        # Resize back to atlas region size
        replacements["face_skin"] = inpainted.resize(
            (region.w, region.h), Image.Resampling.LANCZOS,
        )

    # 5. Hair segmentation on full stylized portrait
    present_hair = [r for r in _HAIR_REGIONS if atlas_cfg.has(r)]
    if present_hair:
        hair_rgba = await segment_hair(stylized, client)
        for name in present_hair:
            region = atlas_cfg.get(name)
            # Crop from segmented hair and resize to atlas region
            replacements[name] = _crop_face_for_region(
                hair_rgba, (0, 0, hair_rgba.width, hair_rgba.height),
                region.w, region.h,
            )

    return replacements


def _build_simple_inpaint_mask(w: int, h: int) -> Image.Image:
    """Build L-mode mask: white ellipse in center (where features are), black edges.

    The mask covers the inner ~60% of the face where eyes/mouth/brows appear.
    """
    from PIL import ImageDraw

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    # Inpaint the central region where facial features typically are
    margin_x = int(w * 0.15)
    margin_top = int(h * 0.20)
    margin_bottom = int(h * 0.15)
    draw.ellipse(
        [margin_x, margin_top, w - margin_x, h - margin_bottom],
        fill=255,
    )
    return mask
