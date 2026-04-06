# pipeline/face_inpaint.py
"""Face skin inpainting — remove portrait facial features, replace with clean skin."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from comfyui.client import ComfyUIClient

_WORKFLOWS_DIR = Path(__file__).parent.parent / "comfyui" / "workflows"
_DEFAULT_PROMPT = (
    "smooth skin texture, same skin tone as surrounding area, "
    "no eyes, no eyebrows, no mouth, no facial features"
)


async def inpaint_face_skin(
    face_crop: Image.Image,
    mask: Image.Image,
    client: ComfyUIClient,
    prompt: str = _DEFAULT_PROMPT,
) -> Image.Image:
    """Inpaint masked regions of face_crop with clean skin via FLUX Fill.

    Args:
        face_crop: RGB face skin crop.
        mask: L-mode image same size as face_crop. White = inpaint, black = keep.
        client: ComfyUI client.
        prompt: Inpainting text prompt.

    Returns:
        Inpainted RGB image.
    """
    workflow_text = (_WORKFLOWS_DIR / "face_inpaint.json").read_text()

    face_path = Path(tempfile.mktemp(suffix=".png"))
    try:
        face_crop.convert("RGB").save(face_path, format="PNG")
        face_name = await client.upload_image(face_path)
    finally:
        face_path.unlink(missing_ok=True)

    mask_path = Path(tempfile.mktemp(suffix=".png"))
    try:
        Image.merge("RGB", [mask, mask, mask]).save(mask_path, format="PNG")
        mask_name = await client.upload_image(mask_path)
    finally:
        mask_path.unlink(missing_ok=True)

    workflow_text = workflow_text.replace('"__IMAGE__"', json.dumps(face_name))
    workflow_text = workflow_text.replace('"__MASK__"', json.dumps(mask_name))
    workflow_text = workflow_text.replace('"__PROMPT__"', json.dumps(prompt))
    workflow = json.loads(workflow_text)

    prompt_id = await client.submit(workflow)
    outputs = await client.wait(prompt_id)
    out_filename = _extract_output_filename(outputs)

    dest = Path(tempfile.mktemp(suffix=".png"))
    try:
        await client.download(out_filename, dest)
        result = Image.open(dest).convert("RGB").copy()
    finally:
        dest.unlink(missing_ok=True)

    return result


def _extract_output_filename(outputs: dict) -> str:
    for node_outputs in outputs.values():
        images = node_outputs.get("images", [])
        if images:
            return images[0]["filename"]
    raise ValueError(f"No images in ComfyUI outputs: {outputs}")
