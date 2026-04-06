"""Style transfer — stylize portrait to match rig art style via ComfyUI."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tomllib
from PIL import Image

from comfyui.client import ComfyUIClient

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_WORKFLOWS_DIR = Path(__file__).parent.parent / "comfyui" / "workflows"


@dataclass
class TextureGenConfig:
    style_transfer: str    # "none" | "anime_flat_cell"
    style_model: str       # ComfyUI checkpoint name
    style_strength: float  # img2img denoise strength 0.0-1.0


def load_texture_gen_config(template_name: str) -> TextureGenConfig:
    """Load [texture_generation] from templates/{template_name}/schema.toml.

    Raises FileNotFoundError if template does not exist.
    """
    schema_path = _TEMPLATES_DIR / template_name / "schema.toml"
    if not schema_path.exists():
        raise FileNotFoundError(f"Template schema not found: {schema_path}")
    with open(schema_path, "rb") as f:
        data = tomllib.load(f)
    tg = data["texture_generation"]
    return TextureGenConfig(
        style_transfer=tg["style_transfer"],
        style_model=tg["style_model"],
        style_strength=float(tg["style_strength"]),
    )


async def stylize_portrait(
    portrait: Image.Image,
    style: str,
    model: str,
    strength: float,
    client: ComfyUIClient,
) -> Image.Image:
    """Stylize portrait via ComfyUI. Returns portrait unchanged if style == 'none'."""
    if style == "none":
        return portrait

    workflow_text = (_WORKFLOWS_DIR / "style_transfer_anime.json").read_text()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    portrait.save(tmp_path, format="PNG")
    uploaded_name = await client.upload_image(tmp_path)
    tmp_path.unlink(missing_ok=True)

    workflow_text = workflow_text.replace('"__IMAGE__"', json.dumps(uploaded_name))
    workflow_text = workflow_text.replace('"__MODEL__"', json.dumps(model))
    workflow_text = workflow_text.replace('"__STRENGTH__"', str(strength))
    workflow = json.loads(workflow_text)

    prompt_id = await client.submit(workflow)
    outputs = await client.wait(prompt_id)
    out_filename = _extract_output_filename(outputs)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        dest = Path(tmp.name)
    await client.download(out_filename, dest)
    result = Image.open(dest).copy()
    dest.unlink(missing_ok=True)

    return result


def _extract_output_filename(outputs: dict) -> str:
    for node_outputs in outputs.values():
        images = node_outputs.get("images", [])
        if images:
            return images[0]["filename"]
    raise ValueError(f"No images in ComfyUI outputs: {outputs}")
