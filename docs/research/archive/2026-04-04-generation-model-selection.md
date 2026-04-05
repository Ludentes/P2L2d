# Research: Generation Model Selection for Style-Diverse Texture Pipeline

**Date:** 2026-04-04  
**Sources:** 14 sources  
**Question:** What replaces SDXL in the Textoon pipeline for April 2026, supporting diverse input styles rather than being locked to anime?

---

## Executive Summary

Textoon hardcodes two anime-specific Chinese SDXL checkpoints (`realcartoonXL_v7`, `sdxl-动漫二次元_2.0`) plus `xinsir ControlNet Union SDXL`. These are wrong for a style-diverse pipeline. The correct replacement is **FLUX.1 Kontext [dev]** — already downloaded at `checkpoints/flux1-dev-kontext_fp8_scaled.safetensors` — which takes a reference portrait and generates character outputs that preserve identity and match the input's visual style across multiple sheets. Community ComfyUI workflows for Kontext-based multi-view character sheets already exist and are production-tested. The main trade-off vs. SDXL is a less mature ControlNet ecosystem; the workaround is img2img + inpainting rather than ControlNet-guided generation. For users who explicitly want anime output from a non-anime input, an Illustrious XL pass with a flat-style LoRA can be added as an optional second stage.

---

## Key Findings

### What Textoon Actually Uses

The Textoon `generate_texture.py` uses two models that are both Chinese-market anime-specific SDXL checkpoints [1]:

- **`realcartoonXL_v7.safetensors`** — from CivitAI, anime cartoon style
- **`sdxl-动漫二次元_2.0.safetensors`** — from LibLib.art (Chinese platform), used by default in all API workflows
- **`xinsir_controlnet_union_sdxl_promax.safetensors`** — ControlNet Union SDXL, supports 10+ control types

The Qwen2.5-1.5B component in Textoon is only the *text parser* (text → structured appearance attributes), not the image generator. The image generator is entirely SDXL.

For a style-diverse pipeline, all three of the above are replaced.

### FLUX.1 Kontext — Best Fit for Multi-Sheet Consistency

**FLUX.1 Kontext [dev]**, released May 2025 by Black Forest Labs, is the strongest candidate for this pipeline [2][3]. Its defining capability is multi-output character consistency: given a reference portrait, it generates variations (different angles, expressions, components) that preserve the subject's identity and the original image's visual style without requiring a LoRA or fine-tuning. The community has already built ComfyUI workflows specifically for this use case:

- "Flux Kontext Character Turnaround Sheet LoRA" — generates front, 3/4, side, back views from one reference [4]
- "Multi-View Turnaround Sheet" — 5-viewpoint character sheets with consistent proportions [4]
- "Consistent Character Creator 3.0" — combines Flux, ControlNet, and face detailer [4]

This maps directly onto the Live2D texture problem: each of the 9 texture sheets (body, hair, sleeves, skirts, etc.) can be generated as a Kontext-conditioned output from the input portrait, inheriting style and identity automatically.

Style handling is the key advantage over SDXL: Kontext "works with AI illustrations, 3D renders, photography, or stylized art while preserving consistent character traits" [4]. Rather than choosing an anime checkpoint and getting locked into that aesthetic, the pipeline inherits the portrait's own style. A painted portrait stays painterly; a semi-realistic portrait stays semi-realistic; an anime portrait stays anime.

**Critical caveat:** Kontext's identity preservation is limited to ~2 megapixels; above that, quality degrades [3]. For 4096×4096 texture sheets, generation must happen at lower resolution and upscale afterward.

The model is already in the local setup: `checkpoints/flux1-dev-kontext_fp8_scaled.safetensors`.

### Qwen-Image-Edit — Secondary Option, Not Primary

**Qwen-Image-Edit-2509** (also already downloaded: `diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors`, `checkpoints/qwen_image_edit_2509_fp8_e4m3fn.safetensors`) is Alibaba's 20B MMDiT image editing model. It supports text-guided style transfer, semantic editing, and identity-preserving portrait manipulation [5][6]. A direct head-to-head comparison against Flux Kontext found [3]:

- Qwen loses facial identity at resolutions above ~1 megapixel with "visible artifacts along edges"
- Flux Kontext "maintained facial consistency" while Qwen produced an "AI polish look that can lose identity"
- Flux Kontext captured subtle details (specific hand positions, micro-expressions) that Qwen missed
- Qwen is faster with Lightning LoRA

Qwen is a reasonable fallback for quick lower-resolution previews but should not be the primary generation model for the 4096×4096 texture sheets. The newer `Qwen-Image-2.0` (Feb 2026, 7B parameters, native 2K) has improved consistency but has not yet been benchmarked directly for this use case.

### ControlNet — The Main Trade-Off

SDXL's `xinsir ControlNet Union` gives Textoon precise structural guidance per component (using canny/depth/lineart conditions). Flux's ControlNet ecosystem is less mature [7]. The practical workaround for the Flux-based pipeline is:

1. Use img2img (reference portrait → component region, denoise 0.35–0.55) instead of ControlNet for structure preservation
2. Use inpainting for components that require hallucinating hidden regions (back hair, body under clothing) — same as Textoon's LaMa inpainting step

This is a usability regression vs. SDXL's precise ControlNet but is workable and maintains style diversity.

### Illustrious XL — Best SDXL Option If Anime Output Is Wanted

If a user specifically wants anime-style output regardless of input portrait style, the SDXL path remains competitive. **Illustrious XL v2.0** with its ecosystem (NoobAI-XL V-Pred 1.0, Nova Anime XL IL v17.0 as of March 2026) is the current leading SDXL anime stack [8]. For flat cell-shaded VTuber-specific output, **Diving-Illustrious Flat Anime Paradigm Shift v6.3+VAE** (updated March 30, 2026, 32,000+ downloads, 5-star) is the community's preferred model [9]. The local setup already has `checkpoints/SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors` as a semi-realistic SDXL option.

This is an optional "anime output" mode, not the default pipeline.

---

## Comparison

| Dimension | SDXL + anime checkpoint (Textoon default) | Flux Kontext [dev] | Qwen-Image-Edit-2509 |
|---|---|---|---|
| **Style flexibility** | Locked to anime | Inherits input style | Good, but "AI polish" tendency |
| **Character consistency (multi-sheet)** | Poor (each generation independent) | Excellent (reference-conditioned) | Moderate (loses identity >1MP) |
| **Max usable resolution** | 1024px native, tile upscale | ~1400px (~2MP), then upscale | ~1024px, degrades above |
| **ControlNet support** | Excellent (xinsir Union) | Limited / immature | Not applicable (edit model) |
| **img2img** | Yes | Yes (reference conditioning) | Yes (native edit model) |
| **ComfyUI character sheet workflows** | Minimal | Multiple production-tested workflows | None found |
| **Already downloaded** | Partially (juggernautXL) | Yes ✓ | Yes ✓ |
| **CivitAI LoRA ecosystem** | Vast | Growing | N/A |

---

## Recommended Architecture for This Pipeline

**Default path:** Flux Kontext [dev] as the primary texture generator. Input portrait → Kontext-conditioned generation per component region → upscale to 4096px. Style is inherited from the portrait automatically.

**Anime output mode:** Optional SDXL stage using Diving-Illustrious Flat Anime Paradigm Shift + flat-style LoRA for users who want stylized anime output from a realistic portrait.

**Quick preview mode:** Qwen-Image-Edit-2509 at 1024px for fast iteration before committing to a full Flux Kontext run.

---

## Open Questions

- Does Flux Kontext with `fp8_scaled` quantization maintain sufficient identity fidelity at the detail level needed for 9 texture sheets? Needs hands-on testing.
- Are there production Flux ControlNet models that handle the structural guidance Textoon used SDXL ControlNet for? None identified in this research.
- How well does Kontext handle non-face components (clothing, hair back-view)? Face consistency is well-documented; clothing/hair consistency across UV regions is not.

---

## Sources

[1] Human3DAIGC. "Textoon README". https://github.com/Human3DAIGC/Textoon (Retrieved: 2026-04-04)  
[2] Black Forest Labs. "FLUX.1 Kontext". https://bfl.ai/models/flux-kontext (Retrieved: 2026-04-04)  
[3] MyAIForce. "Flux Kontext vs Qwen Edit 2509: Pose Transfer Test". https://myaiforce.com/flux-kontext-pose-transfer/ (Retrieved: 2026-04-04)  
[4] RunComfy. "Flux Kontext Character Sheet Workflows". https://www.runcomfy.com/comfyui-workflows/flux-kontext-character-turnaround-sheet-lora (Retrieved: 2026-04-04)  
[5] QwenLM. "Qwen-Image-Edit". https://huggingface.co/Qwen/Qwen-Image-Edit (Retrieved: 2026-04-04)  
[6] Qwen team. "Qwen-Image-Edit release". https://qwenimages.com/blog/qwen-image-edit-release (Retrieved: 2026-04-04)  
[7] Replicate. "FLUX.1 Kontext". https://replicate.com/blog/flux-kontext (Retrieved: 2026-04-04)  
[8] CivitAI. "NoobAI-XL V-Pred 1.0". https://civitai.com/models/833294/noobai-xl-nai-xl (Retrieved: 2026-04-04)  
[9] CivitAI. "Diving-Illustrious Flat Anime Paradigm Shift v6.3". https://civitai.com/models/1620407/diving-illustrious-flat-anime-paradigm-shift (Retrieved: 2026-04-04)  
[10] Comfy.org. "Complete Style Transfer Handbook". https://blog.comfy.org/p/the-complete-style-transfer-handbook (Retrieved: 2026-04-04)  
[11] CivitAI. "FLUX.1-DEV Kontext Workflows Megapack". https://civitai.com/models/618578 (Retrieved: 2026-04-04)  
