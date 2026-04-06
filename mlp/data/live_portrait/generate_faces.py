"""Generate diverse realistic face photos via ComfyUI for LivePortrait training.

Uses JuggernautXL Lightning (4-step SDXL) to generate neutral-expression
portraits with randomized identity, lighting, and angle.

Usage:
    # Start ComfyUI first:
    #   conda activate comfyui && cd ~/w/ComfyUI && python main.py --port 8188

    uv run python -m mlp.data.live_portrait.generate_faces \\
        --n 100 \\
        --out assets/generated-faces \\
        --checkpoint "SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors"
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

from comfyui.client import ComfyUIClient

_DEFAULT_CKPT = "SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors"

# Prompt building blocks — randomized per face for diversity
_GENDERS = ["woman", "man"]
_AGES = ["young", "middle-aged", "elderly", "20-year-old", "35-year-old", "50-year-old"]
_ETHNICITIES = [
    "caucasian", "east asian", "south asian", "african",
    "latin american", "middle eastern", "southeast asian",
]
_LIGHTING = [
    "studio lighting", "natural daylight", "warm golden hour light",
    "cool overcast light", "soft diffused lighting", "dramatic side lighting",
]
_ANGLES = [
    "front facing", "slight left turn", "slight right turn",
    "looking slightly up", "looking slightly down",
]
_BACKGROUNDS = [
    "plain white background", "plain gray background",
    "blurred indoor background", "blurred outdoor background",
]

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, drawing, 3d render, cgi, "
    "deformed, ugly, blurry, low quality, watermark, text, logo, "
    "multiple people, group photo, full body, hands, fingers"
)


def build_prompt(rng: random.Random) -> str:
    """Build a randomized portrait prompt."""
    gender = rng.choice(_GENDERS)
    age = rng.choice(_AGES)
    ethnicity = rng.choice(_ETHNICITIES)
    lighting = rng.choice(_LIGHTING)
    angle = rng.choice(_ANGLES)
    background = rng.choice(_BACKGROUNDS)

    return (
        f"professional portrait photograph of a {age} {ethnicity} {gender}, "
        f"neutral relaxed expression, mouth closed, {angle}, "
        f"{lighting}, {background}, "
        f"sharp focus, high resolution, photorealistic, 85mm lens, f/2.8, "
        f"natural skin texture, detailed eyes"
    )


def build_workflow(prompt: str, checkpoint: str, seed: int) -> dict:
    """ComfyUI API-format workflow for JuggernautXL Lightning 4-step."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": NEGATIVE_PROMPT,
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 768, "height": 1024, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 6,
                "cfg": 2.0,
                "sampler_name": "dpmpp_sde",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2],
            },
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "face_gen",
            },
        },
    }


async def generate_faces(
    n: int,
    out_dir: Path,
    checkpoint: str,
    base_url: str = "http://127.0.0.1:8188",
    seed_base: int = 1000,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed_base)

    async with ComfyUIClient(base_url) as client:
        # Verify connection
        stats = await client.health()
        vram = stats.get("devices", [{}])[0].get("vram_total", 0)
        print(f"ComfyUI connected. VRAM: {vram / 1e9:.1f} GB")

        for i in range(n):
            prompt = build_prompt(rng)
            seed = seed_base + i
            workflow = build_workflow(prompt, checkpoint, seed)

            prompt_id = await client.submit(workflow)
            outputs = await client.wait(prompt_id, timeout=60.0)

            # Find the SaveImage output
            images = []
            for node_out in outputs.values():
                if "images" in node_out:
                    images.extend(node_out["images"])

            if not images:
                print(f"  [{i+1}/{n}] WARN: no images in output, skipping")
                continue

            img_info = images[0]
            dest = out_dir / f"face_{i:04d}.png"
            await client.download(
                img_info["filename"],
                dest,
                subfolder=img_info.get("subfolder", ""),
                file_type=img_info.get("type", "output"),
            )
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{n}] saved {dest.name}")

    print(f"\nDone: {n} faces saved to {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("assets/generated-faces"))
    ap.add_argument("--checkpoint", default=_DEFAULT_CKPT)
    ap.add_argument("--base-url", default="http://127.0.0.1:8188")
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    print(f"Generating {args.n} faces using {args.checkpoint}")
    print(f"Output: {args.out}")
    return asyncio.run(
        generate_faces(args.n, args.out, args.checkpoint, args.base_url, args.seed)
    )


if __name__ == "__main__":
    sys.exit(main())
