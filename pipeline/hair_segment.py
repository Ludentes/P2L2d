"""Hair segmentation — extract RGBA hair image via BiRefNet ComfyUI workflow.

NOTE (Phase 1 limitation): The BiRefNet "General" model segments the full
foreground subject, not hair specifically.  Hair regions cropped from this
mask will include face/body/clothing pixels.  A hair-specific model or
post-processing step is needed for production quality.
"""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from comfyui.client import ComfyUIClient, extract_output_filename
from pipeline.atlas_config import AtlasConfig

_WORKFLOWS_DIR = Path(__file__).parent.parent / "comfyui" / "workflows"
_SPARSE_THRESHOLD = 0.01  # warn if < 1% pixels have any alpha


async def segment_hair(
    portrait: Image.Image,
    client: ComfyUIClient,
) -> Image.Image:
    """Segment hair from portrait via BiRefNet ComfyUI workflow.

    Returns RGBA: hair pixels kept, everything else transparent.
    """
    workflow_text = (_WORKFLOWS_DIR / "hair_segment.json").read_text()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        portrait_path = Path(f.name)
    try:
        portrait.convert("RGB").save(portrait_path, format="PNG")
        uploaded_name = await client.upload_image(portrait_path)
    finally:
        portrait_path.unlink(missing_ok=True)

    workflow_text = workflow_text.replace('"__IMAGE__"', json.dumps(uploaded_name))
    workflow = json.loads(workflow_text)

    prompt_id = await client.submit(workflow)
    outputs = await client.wait(prompt_id)
    out_filename = extract_output_filename(outputs)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dest = Path(f.name)
    try:
        await client.download(out_filename, dest)
        result = Image.open(dest).convert("RGBA").copy()
    finally:
        dest.unlink(missing_ok=True)

    return result


def extract_hair_regions(
    hair_rgba: Image.Image,
    atlas_cfg: AtlasConfig,
    region_names: list[str],
) -> dict[str, Image.Image]:
    """Crop per-region RGBA from warped (template-space) hair image.

    hair_rgba must already be warped to template space.
    Warns if < 1% pixels have any alpha (likely segmentation failure).
    """
    arr = np.array(hair_rgba)
    alpha_ratio = float((arr[:, :, 3] > 0).mean())
    if alpha_ratio < _SPARSE_THRESHOLD:
        warnings.warn(
            f"Hair segmentation: < {_SPARSE_THRESHOLD * 100:.0f}% non-transparent pixels. "
            "Hair regions will be mostly empty. Check portrait or hair model.",
            stacklevel=2,
        )

    result: dict[str, Image.Image] = {}
    scale_x = hair_rgba.width / atlas_cfg.texture_size
    scale_y = hair_rgba.height / atlas_cfg.texture_size

    for name in region_names:
        if not atlas_cfg.has(name):
            continue
        r = atlas_cfg.get(name)
        x = int(r.x * scale_x)
        y = int(r.y * scale_y)
        w = max(1, int(r.w * scale_x))
        h = max(1, int(r.h * scale_y))
        result[name] = hair_rgba.crop((x, y, x + w, y + h))

    return result
