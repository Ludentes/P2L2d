# Research: FLUX Fill Dev Inpainting in ComfyUI

**Date:** 2026-04-06
**Sources:** 8 sources, key ones listed below

---

## Executive Summary

Our `face_inpaint.json` workflow was incorrectly using SDXL-style nodes (`CheckpointLoaderSimple`, `VAEEncodeForInpaint`) with a FLUX Fill model. FLUX models require separate loaders for the diffusion model (`UNETLoader`), text encoders (`DualCLIPLoader` with type="flux"), and VAE (`VAELoader`). The correct inpainting conditioning node is `InpaintModelConditioning` (not `VAEEncodeForInpaint`), and FLUX uses its own guidance mechanism (`FluxGuidance` or `CLIPTextEncodeFlux` guidance parameter) instead of traditional CFG. All required models are already installed locally.

## Key Findings

### Correct Node Graph for FLUX Fill Inpainting

The FLUX Fill inpainting workflow requires these nodes in this order:

1. **UNETLoader** — loads `flux1-fill-dev.safetensors` from `models/diffusion_models/`. Unlike SDXL which uses `CheckpointLoaderSimple` (bundles model + CLIP + VAE), FLUX requires separate loaders [1][2][3].

2. **DualCLIPLoader** with `type: "flux"` — loads both `clip_l.safetensors` and `t5xxl_fp8_e4m3fn.safetensors` (or fp16 variant). FLUX uses dual text encoding: CLIP-L for short features and T5-XXL for detailed understanding [1][3].

3. **VAELoader** — loads `ae.safetensors` (FLUX autoencoder). Separate from the model loader [1][2].

4. **CLIPTextEncode** (or `CLIPTextEncodeFlux`) — encode the inpainting prompt. Standard `CLIPTextEncode` works; `CLIPTextEncodeFlux` provides separate `clip_l` and `t5xxl` prompt fields plus built-in guidance [4].

5. **FluxGuidance** — sets FLUX guidance scale. For inpainting, guidance of 30.0 is common (vs 3.5 for generation). This replaces traditional CFG which should be set to 1.0 [2][5].

6. **InpaintModelConditioning** — the correct node for FLUX Fill inpainting. Takes (positive, negative, vae, pixels, mask) and returns (conditioned_positive, conditioned_negative, latent). `noise_mask=true` ensures sampling only within the masked region [5][6].

7. **KSampler** — `sampler_name: "euler"`, `scheduler: "simple"`, `cfg: 1.0` (disabled — FLUX uses FluxGuidance), `denoise: 1.0` (mask controls the region), `steps: 20` [2][5].

8. **VAEDecode** + **SaveImage** — standard decode and save.

### VAEEncodeForInpaint vs InpaintModelConditioning

`VAEEncodeForInpaint` only encodes the image into latent space with a mask. `InpaintModelConditioning` does the same PLUS prepares the conditioning data for models that have inpainting built into the architecture (like FLUX Fill). For FLUX Fill, `InpaintModelConditioning` is required because the model expects inpainting context in its conditioning, not just a masked latent [5][6].

### Models Available Locally

All required models are already installed in `/home/newub/w/ComfyUI/models/`:

| Model | Path | Size | Status |
|---|---|---|---|
| FLUX Fill diffusion model | `diffusion_models/flux1-fill-dev.safetensors` | 23.8 GB | Installed |
| FLUX Fill FP8 quant | `diffusion_models/fluxFillFP8_v10.safetensors` | 11.9 GB | Installed |
| FLUX Fill GGUF | `diffusion_models/FLUX1/flux1-fill-dev-Q8_0.gguf` | 12.8 GB | Installed |
| CLIP-L text encoder | `text_encoders/clip_l.safetensors` | - | Installed |
| T5-XXL FP8 text encoder | `text_encoders/t5/t5xxl_fp8_e4m3fn.safetensors` | - | Installed |
| T5-XXL FP16 text encoder | `text_encoders/t5/t5xxl_fp16.safetensors` | - | Installed |
| FLUX VAE (autoencoder) | `vae/FLUX1/ae.safetensors` | - | Installed |

No downloads required.

### UNETLoader weight_dtype for VRAM Management

The `UNETLoader` node has a `weight_dtype` parameter: `["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]`. Setting `weight_dtype: "fp8_e4m3fn"` loads the full 23.8 GB model but casts weights to FP8 at load time, reducing VRAM usage. Alternatively, use the pre-quantized `fluxFillFP8_v10.safetensors` (11.9 GB) with `weight_dtype: "default"`.

## Corrected API-Format Workflow

```json
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {
      "unet_name": "fluxFillFP8_v10.safetensors",
      "weight_dtype": "default"
    }
  },
  "2": {
    "class_type": "DualCLIPLoader",
    "inputs": {
      "clip_name1": "clip_l.safetensors",
      "clip_name2": "t5/t5xxl_fp8_e4m3fn.safetensors",
      "type": "flux"
    }
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": { "vae_name": "FLUX1/ae.safetensors" }
  },
  "4": {
    "class_type": "LoadImage",
    "inputs": { "image": "__IMAGE__", "upload": "image" }
  },
  "5": {
    "class_type": "LoadImage",
    "inputs": { "image": "__MASK__", "upload": "image" }
  },
  "6": {
    "class_type": "ImageToMask",
    "inputs": { "image": ["5", 0], "channel": "red" }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": { "text": "__PROMPT__", "clip": ["2", 0] }
  },
  "8": {
    "class_type": "FluxGuidance",
    "inputs": { "conditioning": ["7", 0], "guidance": 30.0 }
  },
  "9": {
    "class_type": "CLIPTextEncode",
    "inputs": { "text": "", "clip": ["2", 0] }
  },
  "10": {
    "class_type": "InpaintModelConditioning",
    "inputs": {
      "positive": ["8", 0],
      "negative": ["9", 0],
      "vae": ["3", 0],
      "pixels": ["4", 0],
      "mask": ["6", 0],
      "noise_mask": true
    }
  },
  "11": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["1", 0],
      "positive": ["10", 0],
      "negative": ["10", 1],
      "latent_image": ["10", 2],
      "seed": 42,
      "steps": 20,
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1.0
    }
  },
  "12": {
    "class_type": "VAEDecode",
    "inputs": { "samples": ["11", 0], "vae": ["3", 0] }
  },
  "13": {
    "class_type": "SaveImage",
    "inputs": { "images": ["12", 0], "filename_prefix": "p2l_inpaint" }
  }
}
```

## Open Questions

- **Guidance value**: 30.0 is commonly used for FLUX Fill inpainting, but may need tuning for anime face skin specifically.
- **Steps**: 20 is a reasonable default. More steps (30-50) may improve quality at the cost of speed.
- **BiRefNet "General" for hair**: Separate issue — the hair segmentation workflow segments the entire foreground, not just hair. Not addressed in this research.

## Sources

[1] ComfyUI Official. "Flux.1 fill dev Example". https://docs.comfy.org/tutorials/flux/flux-1-fill-dev
[2] ComfyUI Wiki. "Flux Fill Workflow Step-by-Step Guide". https://comfyui-wiki.com/en/tutorial/advanced/image/flux/flux-1-dev-fill
[3] ComfyUI Wiki. "Dual CLIP Loader". https://comfyui-wiki.com/en/comfyui-nodes/advanced/loaders/dual-clip-loader
[4] ComfyUI Wiki. "CLIPTextEncodeFlux Node". https://comfyui-wiki.com/en/comfyui-nodes/advanced/conditioning/flux/clip-text-encode-flux
[5] ComfyUI Wiki. "Inpaint Model Conditioning". https://comfyui-wiki.com/en/comfyui-nodes/conditioning/inpaint/inpaint-model-conditioning
[6] RunComfy. "InpaintModelConditioning". https://www.runcomfy.com/comfyui-nodes/ComfyUI/InpaintModelConditioning
[7] ComfyUI Examples. "Flux Examples". https://comfyanonymous.github.io/ComfyUI_examples/flux/
[8] RunComfy. "FLUX Inpainting Workflow". https://www.runcomfy.com/comfyui-workflows/comfyui-flux-inpainting-workflow
