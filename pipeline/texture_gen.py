# pipeline/texture_gen.py
"""Texture generation orchestrator — Phase 1 pipeline."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from comfyui.client import ComfyUIClient
from pipeline.atlas_config import AtlasConfig
from pipeline.face_align import (
    align_portrait,
    build_face_inpaint_mask,
    crop_region,
    warp_image,
)
from pipeline.face_inpaint import inpaint_face_skin
from pipeline.hair_segment import extract_hair_regions, segment_hair
from pipeline.style_transfer import load_texture_gen_config, stylize_portrait
from rig.config import RigConfig

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

_FACE_REGIONS = [
    "face_skin", "left_eye", "right_eye",
    "left_eyebrow", "right_eyebrow", "mouth",
    "left_cheek", "right_cheek",
]
_HAIR_REGIONS = ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]
_WARP_SIZE = (512, 512)
_INPAINT_SIZE = (512, 512)


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
    lm_path = _TEMPLATES_DIR / template_name / "face_landmarks.json"

    # 1. Style transfer
    stylized = await stylize_portrait(
        portrait, style=cfg.style_transfer, model=cfg.style_model,
        strength=cfg.style_strength, client=client,
    )

    # 2. Align portrait to template space
    warped, M = align_portrait(stylized, lm_path, output_size=_WARP_SIZE)

    # 3. Crop face feature regions (not face_skin — that gets inpainted below)
    replacements: dict[str, Image.Image] = {}
    for region_name in _FACE_REGIONS:
        if region_name == "face_skin" or not atlas_cfg.has(region_name):
            continue
        replacements[region_name] = crop_region(warped, atlas_cfg, region_name)

    # 4. Inpaint face skin (remove portrait eyes/brows/mouth from the skin layer)
    if atlas_cfg.has("face_skin"):
        face_crop = crop_region(warped, atlas_cfg, "face_skin")
        face_large = face_crop.resize(_INPAINT_SIZE, Image.Resampling.LANCZOS)
        mask_raw = build_face_inpaint_mask(
            atlas_cfg, warped_size=_WARP_SIZE, face_region_name="face_skin",
            feature_regions=["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "mouth"],
            dilation_px=8,
        )
        mask_large = mask_raw.resize(_INPAINT_SIZE, Image.Resampling.NEAREST)
        replacements["face_skin"] = await inpaint_face_skin(face_large, mask_large, client)

    # 5. Hair segmentation (on stylized pre-warp portrait, then warp the mask)
    present_hair = [r for r in _HAIR_REGIONS if atlas_cfg.has(r)]
    if present_hair:
        hair_rgba = await segment_hair(stylized, client)
        warped_hair = warp_image(hair_rgba, M, output_size=_WARP_SIZE)
        replacements.update(extract_hair_regions(warped_hair, atlas_cfg, present_hair))

    return replacements
