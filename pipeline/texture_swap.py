"""Texture region replacement — paste images into a Live2D texture atlas.

All functions operate on PIL Images and AtlasConfig; no rig-specific logic.
"""
from __future__ import annotations

from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion


def swap_region(
    atlas: Image.Image,
    region: AtlasRegion,
    replacement: Image.Image,
) -> Image.Image:
    """Paste replacement into atlas at region coordinates.

    Scales replacement to region size. Alpha-composites if replacement has alpha.

    Args:
        atlas: The full texture atlas image (RGBA).
        region: Target bounding box within atlas.
        replacement: Image to paste. Resized to (region.w, region.h).

    Returns:
        New Image with replacement pasted. Input atlas is not modified.
    """
    out = atlas.copy()
    src = replacement.resize((region.w, region.h), Image.LANCZOS)
    if src.mode != "RGBA":
        src = src.convert("RGBA")
    out.paste(src, (region.x, region.y), src)
    return out


def swap_regions(
    atlases: dict[int, Image.Image],
    config: AtlasConfig,
    replacements: dict[str, Image.Image],
) -> dict[int, Image.Image]:
    """Batch region replacement across multiple texture atlas images.

    Args:
        atlases: Map of texture_index → PIL Image.
        config: Atlas config providing region coordinates.
        replacements: Map of region_name → replacement Image.

    Returns:
        New dict with modified atlas images. Inputs are not modified.
    """
    out: dict[int, Image.Image] = {k: v.copy() for k, v in atlases.items()}
    for name, replacement in replacements.items():
        region = config.get(name)
        out[region.texture_index] = swap_region(
            out[region.texture_index], region, replacement
        )
    return out
