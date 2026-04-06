"""Top-level orchestrator and CLI for texture generation pipeline.

Usage:
    python -m pipeline.run portrait.jpg \\
        --rig hiyori \\
        --atlas manifests/hiyori_atlas.toml \\
        --out ./output/hiyori_portrait/ \\
        [--template humanoid-anime] \\
        [--comfyui http://127.0.0.1:8188]
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from PIL import Image

from comfyui.client import ComfyUIClient
from pipeline.atlas_config import AtlasConfig, load_atlas_config
from pipeline.package import package_output
from pipeline.texture_gen import generate_textures
from pipeline.texture_swap import swap_regions
from rig.config import RIG_HIYORI, RigConfig


def load_atlases(rig_config: RigConfig) -> dict[int, Image.Image]:
    """Open each texture in rig_config.textures as RGBA PIL Image, keyed by index."""
    return {
        idx: Image.open(tex_path).convert("RGBA")
        for idx, tex_path in enumerate(rig_config.textures)
    }


async def run_portrait_to_rig(
    portrait_path: Path,
    rig_config: RigConfig,
    atlas_cfg: AtlasConfig,
    output_dir: Path,
    template_name: str = "humanoid-anime",
    client: ComfyUIClient | None = None,
) -> Path:
    """Full texture pipeline: portrait -> deliverable Live2D model directory."""
    portrait = Image.open(portrait_path).convert("RGB")

    own_client = client is None
    if client is None:
        client = ComfyUIClient()

    try:
        replacements = await generate_textures(
            portrait=portrait,
            atlas_cfg=atlas_cfg,
            rig=rig_config,
            client=client,
            template_name=template_name,
        )
        atlases = load_atlases(rig_config)
        modified_atlases = swap_regions(atlases, atlas_cfg, replacements)
        package_output(rig_config, modified_atlases, output_dir)
    finally:
        if own_client:
            await client.close()

    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portrait-styled Live2D model")
    parser.add_argument("portrait", type=Path)
    parser.add_argument("--rig", default="hiyori", choices=["hiyori"])
    parser.add_argument("--atlas", type=Path, default=Path("manifests/hiyori_atlas.toml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", default="humanoid-anime")
    parser.add_argument("--comfyui", default="http://127.0.0.1:8188")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    rig_map = {"hiyori": RIG_HIYORI}
    rig_config = rig_map[args.rig]
    atlas_cfg = load_atlas_config(args.atlas)

    async with ComfyUIClient(base_url=args.comfyui) as client:
        out = await run_portrait_to_rig(
            portrait_path=args.portrait,
            rig_config=rig_config,
            atlas_cfg=atlas_cfg,
            output_dir=args.out,
            template_name=args.template,
            client=client,
        )
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    asyncio.run(_main())
