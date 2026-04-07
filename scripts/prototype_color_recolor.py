"""Prototype: color extraction + atlas recoloring — visual comparison.

Extracts colors from a portrait, recolors the Hiyori atlas, renders,
and saves a side-by-side comparison grid.

Run: PYTHONPATH=. uv run python scripts/prototype_color_recolor.py
     PYTHONPATH=. uv run python scripts/prototype_color_recolor.py --portrait assets/data/image1.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.atlas_config import load_atlas_config
from pipeline.color_apply import recolor_atlas
from pipeline.color_extract import extract_palette
from pipeline.package import package_output
from pipeline.run import load_atlases
from pipeline.template_palette import extract_template_palette
from rig.config import RIG_HIYORI, RigConfig
from rig.render import RigRenderer

OUT = Path("test_output/color_recolor")
ATLAS_TOML = Path("manifests/hiyori_atlas.toml")


def make_comparison(original: np.ndarray, recolored: np.ndarray) -> Image.Image:
    """Side-by-side comparison image."""
    h, w = original.shape[:2]
    grid = Image.new("RGBA", (w * 2 + 20, h + 40), (30, 30, 30, 255))
    draw = ImageDraw.Draw(grid)
    draw.text((w // 2 - 20, 5), "Original", fill="white")
    draw.text((w + 20 + w // 2 - 20, 5), "Recolored", fill="white")
    grid.paste(Image.fromarray(original), (0, 30))
    grid.paste(Image.fromarray(recolored), (w + 20, 30))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", default="assets/data/image1.png")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # Load portrait
    portrait = Image.open(args.portrait).convert("RGB")
    print(f"Portrait: {portrait.size}")

    # Extract palette
    print("Extracting color palette...")
    palette = extract_palette(portrait)
    print(f"  Hair LAB: {palette.hair}")
    print(f"  Skin LAB: {palette.skin}")
    print(f"  Eye hue: {palette.eye_color:.1f}, sat: {palette.eye_saturation:.1f}")
    print(f"  Lip LAB: {palette.lip_color}")
    print(f"  Clothing LAB: {palette.clothing}")

    # Save palette
    palette_path = OUT / "palette.json"
    with open(palette_path, "w") as f:
        json.dump(palette.to_dict(), f, indent=2)
    print(f"  Saved palette to {palette_path}")

    # Load atlas + template palette
    atlas_config = load_atlas_config(ATLAS_TOML)
    atlases = load_atlases(RIG_HIYORI)
    template_palette = extract_template_palette(atlases, atlas_config)
    print(f"\nTemplate palette:")
    print(f"  Hair LAB: {template_palette.hair}")
    print(f"  Skin LAB: {template_palette.skin}")
    print(f"  Eye hue: {template_palette.eye_color:.1f}")
    print(f"  Clothing LAB: {template_palette.clothing}")

    # Recolor
    print("\nRecoloring atlas...")
    recolored = recolor_atlas(atlases, palette, atlas_config, template_palette)

    # Save recolored textures
    for idx, img in recolored.items():
        img.save(OUT / f"texture_{idx:02d}_recolored.png")
    for idx, img in atlases.items():
        img.save(OUT / f"texture_{idx:02d}_original.png")
    print("  Saved textures")

    # Package + render recolored
    pkg_dir = OUT / "pkg_recolored"
    package_output(RIG_HIYORI, recolored, pkg_dir)

    output_rig = RigConfig(
        name="hiyori_recolored",
        model_dir=pkg_dir,
        moc3_path=pkg_dir / "hiyori.moc3",
        model3_json_path=pkg_dir / "hiyori.model3.json",
        textures=[pkg_dir / "hiyori.2048" / f"texture_{i:02d}.png" for i in range(2)],
        param_ids=RIG_HIYORI.param_ids,
    )

    with RigRenderer(output_rig, width=512, height=512) as renderer:
        render_new = renderer.render()
        Image.fromarray(render_new).save(OUT / "render_recolored.png")

    # Render original
    with RigRenderer(RIG_HIYORI, width=512, height=512) as renderer:
        render_orig = renderer.render()
        Image.fromarray(render_orig).save(OUT / "render_original.png")

    # Comparison
    grid = make_comparison(render_orig, render_new)
    grid.save(OUT / "comparison.png")
    print(f"\nComparison saved to {OUT / 'comparison.png'}")
    print(f"All outputs in {OUT}/")


if __name__ == "__main__":
    main()
