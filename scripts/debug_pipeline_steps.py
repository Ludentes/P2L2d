"""Run pipeline step by step, saving intermediates for visual inspection."""
import asyncio
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path("test_output/debug_steps")


async def main():
    from comfyui.client import ComfyUIClient
    from pipeline.atlas_config import load_atlas_config
    from pipeline.face_align import detect_landmarks
    from pipeline.style_transfer import load_texture_gen_config, stylize_portrait
    from pipeline.texture_gen import _face_bbox, _crop_face_for_region, _build_simple_inpaint_mask

    OUT.mkdir(parents=True, exist_ok=True)

    # --- Setup ---
    portrait_path = Path(
        "third_party/LivePortrait/src/utils/dependencies/"
        "insightface/data/images/Tom_Hanks_54745.png"
    )
    portrait = Image.open(portrait_path).resize((512, 512), Image.Resampling.LANCZOS).convert("RGB")
    portrait.save(OUT / "00_input_portrait.png")
    print(f"[0] Input portrait saved: {portrait.size}")

    atlas_cfg = load_atlas_config(Path("manifests/hiyori_atlas.toml"))
    cfg = load_texture_gen_config("humanoid-anime")

    # --- Step 1: Landmark detection + face bbox ---
    landmarks = detect_landmarks(portrait)
    face_bbox = _face_bbox(landmarks)
    print(f"[1] Landmarks: {landmarks.shape}, face_bbox={face_bbox}")

    # Draw bbox on portrait for visualization
    viz = portrait.copy()
    draw = ImageDraw.Draw(viz)
    draw.rectangle(face_bbox, outline="red", width=2)
    viz.save(OUT / "01_landmarks_bbox.png")

    # --- Step 2: Style transfer ---
    print(f"[2] Running style transfer (model={cfg.style_model}, strength={cfg.style_strength})...")
    async with ComfyUIClient() as client:
        stylized = await stylize_portrait(
            portrait, style=cfg.style_transfer, model=cfg.style_model,
            strength=cfg.style_strength, client=client,
        )
    stylized.save(OUT / "02_stylized.png")
    print(f"[2] Stylized: {stylized.size} mode={stylized.mode}")

    # --- Step 3: Crop face for each region ---
    FACE_REGIONS = [
        "face_skin", "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow", "mouth",
        "left_cheek", "right_cheek",
    ]
    for name in FACE_REGIONS:
        if not atlas_cfg.has(name):
            print(f"[3] {name}: NOT in atlas config, skipping")
            continue
        region = atlas_cfg.get(name)
        crop = _crop_face_for_region(stylized, face_bbox, region.w, region.h)
        crop.save(OUT / f"03_crop_{name}.png")
        print(f"[3] {name}: {crop.size} (atlas target: {region.w}x{region.h})")

    # --- Step 4: Face inpaint mask + inpainting ---
    if atlas_cfg.has("face_skin"):
        region = atlas_cfg.get("face_skin")
        face_crop = _crop_face_for_region(stylized, face_bbox, region.w, region.h)
        mask = _build_simple_inpaint_mask(region.w, region.h)
        mask.save(OUT / "04a_inpaint_mask.png")

        face_large = face_crop.resize((512, 512), Image.Resampling.LANCZOS)
        mask_large = mask.resize((512, 512), Image.Resampling.NEAREST)
        face_large.save(OUT / "04b_face_for_inpaint.png")
        mask_large.save(OUT / "04c_mask_for_inpaint.png")

        print("[4] Running face inpainting...")
        async with ComfyUIClient() as client:
            from pipeline.face_inpaint import inpaint_face_skin
            inpainted = await inpaint_face_skin(face_large, mask_large, client)
        inpainted.save(OUT / "04d_inpainted.png")
        inpainted_resized = inpainted.resize((region.w, region.h), Image.Resampling.LANCZOS)
        inpainted_resized.save(OUT / "04e_inpainted_resized.png")
        print(f"[4] Inpainted: {inpainted.size} → resized to {inpainted_resized.size}")

    # --- Step 5: Hair segmentation ---
    print("[5] Running hair segmentation...")
    async with ComfyUIClient() as client:
        from pipeline.hair_segment import segment_hair
        hair_rgba = await segment_hair(stylized, client)
    hair_rgba.save(OUT / "05a_hair_rgba.png")

    HAIR_REGIONS = ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]
    for name in HAIR_REGIONS:
        if not atlas_cfg.has(name):
            continue
        region = atlas_cfg.get(name)
        crop = _crop_face_for_region(
            hair_rgba, (0, 0, hair_rgba.width, hair_rgba.height),
            region.w, region.h,
        )
        crop.save(OUT / f"05b_hair_{name}.png")
        print(f"[5] {name}: {crop.size}")

    print(f"\nAll intermediates saved to {OUT}/")


if __name__ == "__main__":
    asyncio.run(main())
