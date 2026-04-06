# CartoonAlive — Texture Editability Research

**Date:** 2026-04-06

---

## Executive Summary

CartoonAlive (arXiv:2507.17327) achieves portrait-driven Live2D personalization through landmark-guided affine alignment and per-component texture extraction — not neural texture synthesis. Each facial region (face skin, eyes, brows, nose, mouth, hair) is treated as an independent texture layer pasted at a known UV destination on the rig's atlas, driven by affine transforms computed between detected landmarks and a fixed template skeleton. Textoon (arXiv:2501.10020), the companion text-driven system from the same team, extends this with an explicit pixel-coordinate UV crop table (`model_configuration.json`) that maps named clothing/hair regions to exact atlas positions, enabling fully automated texture injection into the HaiMeng rig. The key architectural decision enabling editability in both systems is that the HaiMeng rig deliberately places each swappable semantic region on its own dedicated texture sheet (9 sheets total), so texture injection never requires knowledge of UV layout beyond a pre-authored coordinate table.

---

## Method

### CartoonAlive — portrait-driven texture extraction

CartoonAlive (July 2025) is a four-stage pipeline from a single portrait to an animatable Live2D character, completing in under 30 seconds.

**Stage 1 — Facial feature alignment**

The input portrait is eye-rotation-normalised, then MediaPipe is used to detect facial keypoints for eyes, nose, mouth, eyebrows, and facial contour. An affine transform between detected landmarks and the fixed-template landmark positions is computed. The aligned image `I_transformed` is used to extract all facial feature textures, which are mapped onto the Live2D texture atlas at the corresponding UV destinations.

**Stage 2 — Facial feature parameter estimation**

With facial feature textures temporarily removed from the composition, the rig renders using only the underlying face texture. MediaPipe re-runs on this rendering. A 4-layer MLP (trained on 100,000 synthetic samples rendered at 1024×1024 via Pygame) maps the landmark coordinates to Live2D positional parameters (x-shift, y-shift, scale) for each facial component. Parameter space per component: three dimensions (x, y, scale), each ranging from −30 to +30.

**Stage 3 — Underlying face repainting**

After parameters are inferred, the inferred parameters drive a rendering of binary masks for each facial feature. These masks identify which pixels of the underlying face texture are occluded by features during animation. Those pixels are inpainted to prevent visual seams or bleeding during expression changes. Quote from the paper: "we render facial masks based on the inferred parameters and use them to precisely identify the regions requiring inpainting."

**Stage 4 — Hair texture extraction**

Hair segmentation (a dedicated hair-segmentation model) extracts the hair mask from the original portrait. The hair region is transferred into the Live2D model as a separate texture layer. If bangs occlude eyebrows in the input, hair is first removed (using HairMapper, a GAN-based inpainting tool) before eyebrow texture extraction and parameter prediction proceed.

### Textoon — text/image-driven texture injection

Textoon (January 2025) uses an SDXL-generated character image as its source, but the texture injection mechanism is identical in principle to CartoonAlive: affine-warp the source image into the rig's coordinate frame and splat each semantic region into its known UV position.

The critical implementation detail is in `utils/transfer_part_texture.py::extract_part_to_texture()`. For each clothing/hair/face piece, `model_configuration.json` stores explicit pixel-coordinate crop rectangles:

```json
"skirt":     {"x": 72,  "y": 61,  "w": 1268, "h": 2554, "name": "texture_03"}
"trousers":  {"x": 82,  "y": 46,  "w": 871,  "h": 2520, "name": "texture_04"}
"left_boot": {"x": 506, "y": 63,  "w": 284,  "h": 1326, "name": "texture_07"}
```

The pipeline renders the generated character to a working canvas (3360×5040 px from PSD), then uses these coordinates to crop each region and write it into the corresponding 4096×4096 texture sheet at the known destination. No UV unwrapping or atlas parsing at runtime — all UV destinations are pre-authored once per rig and stored as pixel coordinates.

For occluded body parts (e.g. a shirt arm hidden behind a skirt), completion is: "fill the occluded areas with pixels from the unoccluded regions, followed by image-to-image control generation for refinement."

---

## Atlas / UV Design

### HaiMeng rig atlas structure (the rig both systems target)

The HaiMeng production rig uses **9 dedicated 4096×4096 texture sheets**, each carrying a distinct semantic category:

| Sheet | Content |
|---|---|
| `texture_00.png` | Body base, thighs, calves, eyes, face skin |
| `texture_01.png` | All hair variants |
| `texture_02.png` | Sleeves |
| `texture_03.png` | Skirts |
| `texture_04.png` | Trousers |
| `texture_05.png` | Shirt body |
| `texture_06.png` | Unused / extra |
| `texture_07.png` | Boots |
| `texture_08.png` | Breast variant shirts |

Each sheet carries one garment category. Swapping hair means writing to `texture_01` only. Swapping skirt means writing to `texture_03` only. No other sheet is touched.

This is the architectural decision that makes the pipeline tractable: **by assigning one semantic category per texture sheet, texture injection is pixel-painting at a known rectangle, requiring no UV unwrapping or mesh knowledge at generation time.**

Within each sheet, regions are positioned at specific pixel coordinates (the crop table in `model_configuration.json`). The rig's UV mesh was authored in Cubism Editor so that each art mesh maps to exactly that pixel rectangle.

### ARKit parameter rig

The HaiMeng rig has 107 parameters, including 24 ARKit-compatible mouth/jaw blendshapes (`ParamJawOpen`, `ParamMouthSmileLeft`, `ParamMouthFrownLeft`, etc.). This is what the CartoonAlive MLP targets: the paper states it leverages "a more comprehensive parameter space consisting of 52 facial expression controls provided by ARKit" compared to the standard 2-parameter mouth.

### Official sample models (Hiyori, Natori, etc.) — not compatible

Official SDK sample models use 1–2 × 2048 px sheets with all clothing baked into one atlas as a fixed outfit. No per-category sheets, no UV crop table, no boolean garment toggles. Swapping texture regions on these rigs requires re-rigging in Cubism Editor. They are not compatible with either CartoonAlive or Textoon without rebuilding the atlas from scratch.

---

## Key Insights for portrait-to-live2d

### 1. Texture editability is an atlas design problem, not a model problem

Neither CartoonAlive nor Textoon uses neural texture synthesis (no NeRF, no GAN inversion, no diffusion for texture swapping). Both work by pre-authoring a UV crop table once per rig, then using affine-warped pixel crops from the source portrait/image. The intelligence is in the alignment step (landmark → affine transform), not the texture writing step.

**Implication for P2L:** when we generate our own rig, we must author the atlas layout so each swappable region occupies a predictable, documented UV rectangle. The crop table (`atlas_config.toml` in our existing convention) is the contract that makes automated texture injection possible.

### 2. Separate texture sheets per semantic category is the right architecture

One sheet per major swappable category (hair, face, clothing) eliminates any ambiguity about which pixels to write. The HaiMeng team chose 9 × 4096 sheets specifically to achieve this. For a simpler initial rig, even 3–4 sheets (face/hair/body/clothing) following the same principle would work.

### 3. Face texture injection uses affine warp, not segmentation model

The face skin region is handled by computing a correspondence between the detected facial contour and the template contour, then applying a homographic or thin-plate-spline warp. There is no separate SAM2 / segmentation step for the face — the contour correspondence is enough. Hair is the only region that requires a dedicated segmentation model (because hair extends outside the facial landmark convex hull).

**Implication for P2L:** our existing plan to use SAM2 for region segmentation may be heavier than necessary for the face itself. A landmark-driven affine warp to a fixed template UV is the simpler and proven path. SAM2 remains useful for hair.

### 4. Inpainting the underlying face is load-bearing

Without repainting the areas under eyes/brows/mouth with plausible skin, animation artefacts appear as features move away from neutral position, revealing the uncleaned underlying texture. CartoonAlive makes this an explicit stage. For our pipeline, any portrait-to-texture step must include inpainting the eye sockets, brow ridges, and lip area of the face-base texture.

### 5. Component separation order matters

CartoonAlive's pipeline order:
1. Detect and warp all face features (eyes, brows, nose, mouth) into atlas positions
2. Remove features, render underlying face only, run MLP to get placement params
3. Repaint underlying face using inferred masks
4. Extract and transfer hair separately (requires hair segmentation)

The key insight is that parameter estimation happens on the feature-stripped rendering, not the original portrait. This avoids the hair/feature occlusion problem entirely for the parameter prediction step.

### 6. MLP training data: 100k synthetic samples from Pygame rendering

CartoonAlive trains its placement MLP purely on synthetic data: 100,000 face renders at known Live2D parameter values, generated programmatically via Pygame. This is identical in spirit to our approach (Live2D rig renders at known param values for pose training). The scale is larger (100k vs our current ~10k) but the methodology is the same.

### 7. Hair removal before eyebrow extraction

If bangs occlude eyebrows, CartoonAlive removes the hair first (HairMapper GAN), then extracts the brow texture. This is a specific failure mode we should handle: for portraits with bangs that cover the brows, the MLP will mispredict brow position unless we preprocess. HairMapper is a lightweight dedicated model for this. Alternatively, we can run hair segmentation first and in-paint hair over the portrait before the landmark step.

---

## What we should adopt

| CartoonAlive/Textoon technique | Adopt? | Notes |
|---|---|---|
| One texture sheet per semantic category | Yes | Design our generated rig's atlas this way from the start |
| Pre-authored pixel-coordinate UV crop table | Yes | Our `atlas_config.toml` convention already does this |
| Landmark-affine warp for face skin extraction | Yes | Simpler than SAM2 for face region |
| Hair segmentation for hair layer extraction | Yes | Needed because hair extends outside landmark hull |
| Inpaint underlying face after feature removal | Yes | Required to avoid animation artefacts |
| Feature-stripped rendering for parameter estimation | Consider | Relevant if we ever add a placement-prediction MLP |
| Pygame synthetic renders for MLP training | Already doing | Same approach, different param space |
| HairMapper for bang removal | Optional | Only needed for portraits with brow-occluding bangs |
| 9 × 4096 atlas like HaiMeng | No | Overkill for our dev rig; 3-4 × 2048 is sufficient |
| Full ARKit 52-param rig | Aspirational | HaiMeng has this; Hiyori does not |

---

## Sources

- [CartoonAlive paper (arXiv:2507.17327)](https://arxiv.org/abs/2507.17327)
- [CartoonAlive HTML full paper](https://arxiv.org/html/2507.17327v1)
- [CartoonAlive project page](https://human3daigc.github.io/CartoonAlive_webpage/)
- [CartoonAlive GitHub repo](https://github.com/Human3DAIGC/CartoonAlive)
- [Textoon paper (arXiv:2501.10020)](https://arxiv.org/abs/2501.10020)
- [Textoon HTML full paper](https://arxiv.org/html/2501.10020v1)
- [Textoon project page](https://human3daigc.github.io/Textoon_webpage/)
- [Textoon GitHub repo](https://github.com/Human3DAIGC/Textoon)
- [Project doc: Live2D Official Samples vs. HaiMeng](../2026-04-03-live2d-official-samples-vs-haimeng.md)
- [Live2D Texture Atlas Editor Manual](https://docs.live2d.com/en/cubism-editor-manual/texture-atlas-edit/)
