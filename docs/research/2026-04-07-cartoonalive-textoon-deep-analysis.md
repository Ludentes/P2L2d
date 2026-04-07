# CartoonAlive / Textoon Deep Analysis & Recommended Way Forward

**Date:** 2026-04-07
**Status:** Research — informs next implementation direction

---

## Executive Summary

After thorough analysis of both CartoonAlive (arXiv:2507.17327) and Textoon (arXiv:2501.10020), plus the HaiMeng rig assets, the critical finding is: **neither system solves the portrait-to-anime texture problem**. CartoonAlive assumes the input is *already stylized anime*. Textoon generates from *text*, not a photo. The photo-to-anime domain gap is left unaddressed by both papers.

Our deterministic color recoloring (LAB shift / HSV rotation) produces technically correct color transfers but visually underwhelming results — it changes tints but can't change structure, add character-specific detail, or make the output feel personalized beyond color.

The way forward is to adopt Textoon's proven architecture — **generate a full-body anime image with SDXL + ControlNet, then crop-and-paste into atlas positions** — but replace Textoon's text input with portrait-conditioned generation (IP-Adapter for identity, ControlNet for structure).

---

## What CartoonAlive Actually Does

CartoonAlive is **not** a texture generation system. It is a texture *placement* system that assumes anime-styled input.

### The 4-Stage Pipeline

| Stage | What It Does | Key Detail |
|---|---|---|
| 1. Facial feature alignment | Affine-warp portrait features onto template UV positions | MediaPipe landmarks → affine transform per component |
| 2. Parameter estimation | MLP predicts Live2D positional params (x, y, scale per feature) | 4-layer MLP, trained on 100K synthetic renders, 3 dims × 5 components |
| 3. Face inpainting | Fill pixels under movable features to prevent seam bleed | Binary masks from inferred params → inpaint occluded regions |
| 4. Hair extraction | Segment and transfer hair as separate texture layer | Dedicated hair segmentation model; HairMapper for bang-occluded eyebrows |

### What CartoonAlive Does NOT Do

- No diffusion models, no GANs, no neural style transfer
- No photo-to-anime conversion
- No clothing generation or transfer
- The paper's Figure 1 shows a "stylized cartoon version" as input — this is a **prerequisite**, not something CartoonAlive creates
- The paper never mentions how to obtain this stylized input

### What We Can Adopt

- **Landmark-driven affine warp** for facial feature placement (simpler than neural methods)
- **Feature-stripped rendering** for parameter estimation (render rig without features → re-detect landmarks)
- **Inpainting under movable features** (load-bearing for animation quality)
- **HairMapper** for bang-occluded eyebrow recovery

---

## What Textoon Actually Does

Textoon is the generative system. It uses SDXL to create anime characters from text, then injects textures into the HaiMeng rig via a deterministic crop-and-paste pipeline.

### The Complete Pipeline

```
Text Input
  → Qwen2.5-1.5B parses attributes (hair, clothing, etc.)
  → Deterministic assembly of pre-drawn part silhouettes into control image
  → SDXL txt2img (realcartoonXL_v7, ControlNet Union ProMax, 1024×1536)
  → Upscale to 3360×5040 (PSD canvas size)
  → Per-part crop from generated image at "photo" coordinates
  → Per-part paste into texture sheet at "texture" coordinates
  → SDXL img2img inpainting for back-hair occlusion (sdxl-anime_2.0, denoise=0.5)
  → Package: copy moc3 + write modified textures
```

### The Two Coordinate Systems

This is Textoon's core innovation. `model_configuration.json` defines two pixel-coordinate rectangles per body part:

```json
"bang_hair": {
    "photo":   {"x": 1232, "y": 550,  "w": 519, "h": 450},
    "texture": {"x": 3138, "y": 804,  "w": 519, "h": 450, "name": "texture_01"}
}
```

- **"photo" coords**: where the part appears in the full-body generated image (2951×5444 canvas)
- **"texture" coords**: where the part must be pasted in the target 4096×4096 texture sheet

**The w and h are identical between photo and texture** (no scaling). The only exception is thighs, which get a -90 degree rotation (w/h swap). This means the generated image and the atlas share the same pixel resolution per part — the crop-paste is lossless.

### HaiMeng Atlas Structure

| Sheet | Content | Parts |
|---|---|---|
| texture_00 | Body, thighs, calves, eyes, face skin | left/right_thigh, left/right_calf, eyes |
| texture_01 | All hair variants | bang, back, sides, ponytails (8 part slots) |
| texture_02 | Sleeves | left/right_long_sleeve (6 length variants each) |
| texture_03 | Skirts | skirt (5 length variants) |
| texture_04 | Trousers | trousers (6 length variants) |
| texture_05 | Shirt body | turtleneck_shirt (5 neckline variants) |
| texture_06 | Unused | — |
| texture_07 | Boots | left/right_boot (6 height variants each) |
| texture_08 | Breast variant shirts | alternative shirt for breast_1 body type |

**Key design principle**: one semantic category per sheet. Swapping hair = write to texture_01 only. No cross-sheet dependencies.

### SDXL Generation Details

**txt2img pass:**
- Model: `realcartoonXL_v7.safetensors` (CivitAI anime SDXL fine-tune)
- ControlNet: `xinsir_controlnet_union_sdxl_promax` (accepts canny + color hints)
- Control image: Canny edges of the silhouette composite blended (screen) with the color-coded silhouette
- Sampler: euler_ancestral, karras, 20 steps, CFG 7.04, denoise 1.0
- Resolution: 1024×1536

**img2img inpainting pass (back-hair only):**
- Model: `sdxl-anime_2.0.safetensors` (different from txt2img)
- ControlNet: same union model, lineart preprocessing
- Prompt: auto-generated by Joy Caption (Llama-3.1-8B)
- Sampler: euler_ancestral, karras, 20 steps, CFG 7, denoise 0.5
- Only runs for non-ponytail hairstyles where back hair is occluded

### What Textoon Does NOT Do

- No portrait input (text only)
- No identity preservation from a photo
- No face generation (face comes from static basemap + pre-colored eye templates)
- No per-drawable editing (works at body-part level, not UV mesh level)

---

## Why Our Color Recoloring Falls Short

Our current approach (LAB shift for hair/skin/clothing, HSV hue rotation for eyes) is technically correct but produces visually weak results because:

1. **Color is not identity.** Shifting the Hiyori template's hair from brown to black doesn't make it look like a specific person — it just makes it darker. The shape, style, highlights, and texture remain Hiyori's.

2. **The Hiyori atlas was not designed for swapping.** Unlike HaiMeng's 9-sheet semantic layout, Hiyori packs everything into 2 × 2048px sheets with overlapping regions, baked-in line art, and no clean part boundaries. Color shifting is the *only* thing that can work on this atlas without a complete re-rig.

3. **No structural changes.** Color recoloring can't change hair style, eye shape, clothing design, or facial proportions. The result always looks like Hiyori with a color filter — not like the person in the portrait.

4. **Low-saturation targets produce gray.** When the portrait has dark/neutral hair (common), the LAB shift correctly moves the template toward neutral, but the result looks washed out rather than characterful.

---

## Recommended Way Forward

### Option A: Portrait-Conditioned SDXL Generation (Textoon-Style)

**Approach**: Adopt Textoon's proven generate-then-crop architecture, but replace text input with portrait conditioning via IP-Adapter.

```
Portrait Photo
  → Face analysis (MediaPipe landmarks, hair/clothing color extraction)
  → IP-Adapter encodes portrait identity features
  → ControlNet provides structural constraint (Hiyori silhouette as control image)
  → SDXL generates anime character matching portrait identity + template structure
  → Crop generated image per-part using atlas coordinate table
  → Paste into Hiyori texture sheets
```

**What this requires:**
1. A **Hiyori-specific control image** (composite silhouette like Textoon's part assembly) at the PSD canvas resolution
2. A **Hiyori-specific `model_configuration.json`** — the "photo" → "texture" coordinate mapping
3. **IP-Adapter SDXL** for identity conditioning (face similarity from portrait)
4. **ControlNet Union** for structural conditioning (character pose/silhouette)
5. An anime SDXL checkpoint (realcartoonXL_v7 or similar)

**Pros:**
- Proven architecture (Textoon ships this)
- Generates actual anime textures (not just color-shifted templates)
- Full character appearance (hair style, clothing design, not just colors)
- Crop-paste is simple and reliable once coordinate table exists

**Cons:**
- Requires building the Hiyori coordinate table and control silhouette (significant one-time effort)
- IP-Adapter identity preservation in anime style is imperfect (known domain gap issue)
- Needs ComfyUI + GPU at generation time
- Hiyori's 2 × 2048px atlas is much more constrained than HaiMeng's 9 × 4096px
- Generated textures may not perfectly match Hiyori's art style (line weight, shading)

**Estimated effort:** 1-2 weeks

### Option B: Get HaiMeng Access, Use Textoon Directly

**Approach**: Obtain the HaiMeng rig (EULA-gated), clone Textoon, and adapt their pipeline for portrait input.

**What this requires:**
1. HaiMeng EULA approval (university account, pending)
2. Clone Textoon repo, set up their ComfyUI workflows + model downloads
3. Replace Qwen2.5 text parsing with portrait analysis (hair/clothing detection → attribute list)
4. Add IP-Adapter to their SDXL workflow for face identity
5. Use their existing `model_configuration.json` and control images as-is

**Pros:**
- Most of the hard work (atlas design, coordinate tables, control images, part masks) is already done
- 9-sheet atlas is purpose-built for texture swapping
- 107 params including ARKit blendshapes = superior facial animation
- Textoon's pipeline is tested and functional

**Cons:**
- EULA-gated (blocked until approval)
- HaiMeng is a specific character design — may not match desired aesthetic
- Large model downloads (SDXL + ControlNet + Joy Caption)
- Tight coupling to HaiMeng rig (not generalizable to other rigs)

**Estimated effort:** 3-5 days (after EULA approval)

### Option C: Hybrid — Color Recoloring + Selective AI Inpainting

**Approach**: Keep our working color recoloring as the base, but add targeted AI inpainting for high-impact regions only (face, hair front, eyes).

```
Portrait Photo
  → Color extraction (existing pipeline)
  → Deterministic recoloring of all regions (existing)
  → For hair_front + hair_back: SDXL inpainting with portrait conditioning
  → For face_skin: SDXL inpainting with portrait conditioning
  → Merge AI-inpainted regions with color-recolored base
```

**Pros:**
- Builds on working code (color recoloring already functional)
- Limits AI generation to specific regions (less chance of full-image artifacts)
- Uses our existing mask files (the 132 per-drawable masks just generated)
- No atlas redesign needed

**Cons:**
- Inpainting at atlas-texture level (not rendered-character level) is the approach that already failed — AI models struggle with disassembled atlas regions
- Boundary artifacts between AI-inpainted and color-shifted regions
- Still fundamentally limited by Hiyori's 2-sheet atlas

**Estimated effort:** 1 week

### Option D: Custom Rig with Semantic Atlas (Long-term)

**Approach**: Build a custom rig in Cubism Editor with HaiMeng-style semantic texture sheets. Then use Textoon-style generation.

**Pros:**
- Full control over atlas design, parameter set, and art style
- Can include semantic drawable naming (Type B) for automation
- One-time investment that makes all future texture generation trivial

**Cons:**
- Cubism Editor rigging is a significant skill and time investment (weeks to months)
- The user is currently learning Cubism Editor (documented in project memory)
- Blocks all texture work until rig is complete

**Estimated effort:** 4-8 weeks

---

## Recommendation

**Short-term (now):** **Option B** — push for HaiMeng EULA approval. This unblocks the fastest path to a working end-to-end pipeline. Textoon's infrastructure is production-grade and the coordinate mapping system is the proven solution.

**If HaiMeng is blocked:** **Option A** — build the Hiyori coordinate table and control image, then use IP-Adapter + ControlNet SDXL generation. This requires the most original engineering but works with our existing dev rig.

**Keep:** The color recoloring pipeline. It's useful as a fast fallback (no GPU, sub-second), for previews, and as a baseline comparison. It can also be the "level 1" personalization with AI generation as "level 2".

**Deprioritize:** Option C (hybrid inpainting on atlas regions already proved fragile) and Option D (too long-term to be actionable now).

---

## Key Technical Lessons from Textoon

1. **Generate at character level, not atlas level.** Textoon generates a full-body character image (1024×1536), then crops parts out. It does NOT generate into atlas space directly. This is why it works and our per-region atlas inpainting failed.

2. **The coordinate table is everything.** `model_configuration.json` is a simple JSON mapping part names to pixel rectangles in two spaces. This is the critical piece that enables crop-and-paste. Building this for Hiyori is the highest-value task.

3. **Pre-authored silhouette parts enable ControlNet.** Textoon's control image is NOT derived from the generated image — it's assembled from pre-drawn template silhouettes. This gives ControlNet a clean structural signal without the noise of a half-generated image.

4. **Face is treated separately from body.** Both CartoonAlive and Textoon handle face/eyes as a separate system (landmarks, pre-colored templates) rather than trying to generate them with the same diffusion pass. This is wise — face identity is the hardest part and benefits from dedicated handling.

5. **One semantic category per texture sheet.** The 9-sheet design of HaiMeng is not accidental — it eliminates cross-region contamination during texture injection. Hiyori's 2-sheet atlas fundamentally constrains what we can do.
