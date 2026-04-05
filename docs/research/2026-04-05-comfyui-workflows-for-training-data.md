# ComfyUI Workflows for Live2D Training Data Generation (2026)

**Date:** 2026-04-05
**Status:** Research, pre-implementation
**Related:** [Template Rig Architecture](2026-04-05-template-rig-architecture.md), [Anime Landmark Eye Tracking Failure](2026-04-05-anime-landmark-eye-tracking-failure.md)

---

## Executive Summary

The training-data problem splits into two distinct workflows, each with a clear 2026 winner:

**Expression editing (Task 2)** should use **ExpressionEditor** from `ComfyUI-AdvancedLivePortrait` as the primary verb-application mechanism. It provides *parametric sliders* (pitch, yaw, blink, smile, wink, eyebrow, pupil_x, pupil_y, aaa/eee/woo mouth shapes) that map directly to our verb-library concept. No diffusion is needed for standard verbs — edits are ~100ms, preserve identity perfectly, and the underlying LivePortrait model was trained with mixed realistic+stylized data. **FLUX Kontext** serves as the fallback for verbs ExpressionEditor can't express (e.g., "tongue out", complex compound expressions).

**Anime→photo style transfer (Task 1)** should use **FLUX.1-dev + FLUX ControlNet Union (Depth + Canny)** at img2img denoise 0.50–0.60, or SDXL+ControlNet with `epiCRealism`/`majicMIX realistic v7` for lower-VRAM setups. The structural preservation comes from ControlNet conditioning (depth + canny), not denoise strength alone.

**The recommended pipeline inverts the initial assumption**: rather than "render anime → style-transfer → MediaPipe", the cheapest path is "reference photo → ExpressionEditor(verb) → MediaPipe" — skipping both the anime domain and diffusion entirely for ~90% of verbs. The anime-render path is reserved for pose sweeps (head AngleX/Y/Z where Live2D deformers *do* move landmark contours correctly).

---

## Key Findings

### 1. ExpressionEditor is the 2026 parametric-expression primitive

`PowerHouseMan/ComfyUI-AdvancedLivePortrait` exposes KwaiVGI's LivePortrait [1] as a single ComfyUI node with ~10 expression sliders. Parameter ranges observed in the node:

| Slider | Range | Maps to verb |
|---|---|---|
| `rotate_pitch` | -20 to 20 | look up/down |
| `rotate_yaw` | -20 to 20 | look left/right (head) |
| `rotate_roll` | -20 to 20 | head tilt |
| `blink` | -20 to 5 | close eyes (negative = more closed) |
| `eyebrow` | -10 to 15 | brow raise/furrow |
| `wink` | 0 to 25 | wink_left / wink_right (separable) |
| `pupil_x` | -15 to 15 | look left/right (eyes) |
| `pupil_y` | -15 to 15 | look up/down (eyes) |
| `aaa` | -30 to 120 | mouth open vertical |
| `eee` | -20 to 15 | mouth wide (smile-like) |
| `woo` | -20 to 15 | pucker |
| `smile` | -0.3 to 1.3 | smile intensity |

LivePortrait itself [1] was trained on "69M high-quality frames" with mixed realistic + stylized data. Community reports (r/StableDiffusion, 2024–2025) confirm it works on anime characters, though with documented limits: head rotations combined with dangly accessories (earrings, long hair behind ears) can produce artifacts, and extreme angles (>45°) break.

The critical property for our pipeline: **identity is preserved by construction** — ExpressionEditor warps the source image via learned motion fields, it does not re-synthesize through a diffusion prior. A batch of 60 verbs from one reference image yields 60 same-character images with varied expressions.

Community sentiment quoted from r/StableDiffusion: *"useful tool as part of generating images for training a Lora model"* [2] — which is essentially our use case (training an MLP instead of a LoRA).

### 2. Character-preserving edit models: FLUX Kontext > Qwen Edit > Nano Banana for our task

Three 2024–2026 models contest this space:

| Model | Params | Strength | Weakness | Our fit |
|---|---|---|---|---|
| **FLUX.1 Kontext** | 12B | Best layout preservation; explicit "keep everything, change X" prompts [3] | Closed weights (dev/pro/max tiers); VRAM 16GB+ for dev variant | **Primary diffusion fallback** |
| **Qwen Image Edit** | 20B MMDiT | Strong text rendering, color edits | Tends to shift composition/crop even when asked to preserve; anime style drift observed [4] | Secondary |
| **Nano Banana (Gemini 2.5 Flash Image)** | — | Strong character consistency across multi-turn edits [5] | API-only (no ComfyUI local node); struggles with non-1:1 aspect ratios | API experiments only |

For our verb-editing use case, FLUX Kontext wins on two dimensions: (a) it runs locally in ComfyUI via the native Flux nodes, (b) its training objective explicitly includes "preserve all unmentioned regions" which is the exact contract we need ("change expression, keep hair/clothing/style").

Community consensus [3][6]: use the **dev** variant locally (quality ≈ pro for edit tasks), reserve **max** for API calls. Typical prompt pattern: `"[character description], [verb phrase], keep exact same character, clothing, hair, and art style"`.

### 3. Anime-to-photo style transfer: ControlNet dominates denoise strength

Across multiple 2024–2025 Civitai workflows [7][8], the consistent finding is that structural preservation during anime→photo transfer comes from **ControlNet conditioning**, not from low img2img denoise. A workflow with denoise 0.75 + Depth(0.8) + Canny(0.5) preserves pose *better* than denoise 0.35 with no ControlNet, because the latter retains anime-style proportions (large eyes, small nose).

Recommended 2026 stack for SDXL-based transfer:

- **Base model**: `epiCRealism` (natural faces) or `majicMIX realistic v7` (Asian features, common in anime source material)
- **ControlNet stack**: Depth (strength 0.7, end 0.8) + Canny (strength 0.4, end 0.6)
- **Denoise**: 0.55–0.65 (high enough to break anime proportions, low enough to keep pose)
- **CFG**: 5.0–6.5
- **Sampler**: DPM++ 2M Karras, 25–30 steps

For FLUX-based transfer (higher quality, higher VRAM):
- **FLUX.1-dev** + FLUX ControlNet Union Pro 2.0 [9]
- Depth (0.6) + Canny (0.35), end both at 0.7
- Denoise 0.60, CFG 3.5, 20 steps

Both configurations produce images where MediaPipe eye/mouth blendshapes respond correctly (validated anecdotally in multiple community threads; formal validation is an open experiment for us).

### 4. Tiered pipeline beats pure style-transfer

The user's proposed pipeline ("render → style transfer → check for blink → Qwen Edit fallback") is architecturally correct but can be simplified further given Finding #1. The inverted pipeline:

```
For pose verbs (HeadYaw, HeadPitch, BodyAngle*):
    Hiyori rig render at known param → MediaPipe landmarks → label
    (This already works: AngleX R²=0.229 on current data)

For expression verbs (EyeLOpen, MouthForm, Brow*, EyeBall*):
    Reference photo of target character-style (1 human photo)
        → ExpressionEditor(verb_params) → N variations
        → MediaPipe → extract 1014-d (landmarks + blendshapes + pose)
        → label with verb's template param values
        → quality filter (verb says eyes_closed → eyeBlinkLeft > 0.7 expected)

For verbs ExpressionEditor can't express (tongue, cheek puff, compound):
    Reference → FLUX Kontext(verb prompt) → MediaPipe validation → accept/reject

For anime-style validation (does this generalize to rendered rig targets?):
    Live2D anime render → FLUX ControlNet style transfer → MediaPipe
    (Used as held-out test set, NOT training data)
```

This keeps diffusion out of the hot loop: ~90% of training samples come from cheap parametric warping (ExpressionEditor: ~100ms) or existing rig renders (<50ms). Only edge-case verbs hit FLUX Kontext (~15–30s per sample at 1024²).

### 5. Character consistency across batches: InstantID > PuLID > IP-Adapter FaceID for faces

When generating the initial reference-photo set (if we want multiple human-photo templates), face-preservation adapters matter. Community benchmarks [10] on a 1–10 scoring rubric across identity/prompt-adherence/style-consistency:

| Adapter | Overall | Identity rigidity | Expression flexibility |
|---|---|---|---|
| **InstantID** | 38/40 | High | Medium |
| **PuLID** | 25/40 | Very high | Low (too rigid for expression editing) |
| **IP-Adapter FaceID Plus v2** | 32/40 | Medium | High |

**For our use case, IP-Adapter FaceID Plus v2 is the right choice** despite InstantID's higher overall score, because we *want* expression flexibility. PuLID is actively harmful — it overcorrects toward the reference face's neutral expression, fighting the verb prompt.

Standard weights: IP-Adapter FaceID weight 0.7–0.8, combined with a CLIP vision ControlNet at 0.3–0.4 for style lock.

---

## Comparison: Expression Editing Approaches

| Approach | Cost/sample | Identity preservation | Expression control | Anime support | Our verdict |
|---|---|---|---|---|---|
| **ExpressionEditor (LivePortrait)** | ~100ms | Perfect (warp) | Parametric, 12 sliders | Yes (mixed training) | **Primary** |
| **FLUX Kontext (verb prompt)** | ~20s | High (layout preservation) | Free-text verbs | Yes (via prompt) | **Fallback** |
| **Qwen Image Edit** | ~15s | Medium (drift observed) | Free-text verbs | Style drift | Skip |
| **IP-Adapter FaceID + prompt** | ~10s | Medium-high | Free-text | Yes | Character-gen only |
| **AnimateDiff expression ctrl** | ~60s (video) | Medium | Temporal | Yes | Overkill for static frames |

---

## Comparison: Anime→Photo Style Transfer

| Stack | VRAM | Quality | Speed | Our verdict |
|---|---|---|---|---|
| **SDXL + epiCRealism + Depth+Canny** | 10GB | Good | ~8s | Accessible default |
| **FLUX.1-dev + ControlNet Union 2.0** | 18GB | Best | ~15s | Quality target |
| **SD 1.5 + anime2photo LoRA** | 6GB | Medium | ~4s | Fast iteration |
| **Nano Banana API** | 0 (cloud) | Good | ~3s | Aspect-ratio issues |

---

## Essential ComfyUI Node Packs (2026)

- `PowerHouseMan/ComfyUI-AdvancedLivePortrait` — ExpressionEditor, primary verb mechanism
- `comfyanonymous/ComfyUI` core — FLUX and SDXL base
- `Fannovel16/comfyui_controlnet_aux` — ControlNet preprocessors (depth, canny, openpose)
- `cubiq/ComfyUI_IPAdapter_plus` — IP-Adapter FaceID v2 nodes
- `kijai/ComfyUI-FluxTrainer` or `XLabs-AI/x-flux-comfyui` — FLUX extensions
- `rgthree/rgthree-comfy` — workflow utilities (Context, batch loops)
- MediaPipe extraction is handled outside ComfyUI in our existing `mlp/data/` pipeline — no ComfyUI node needed

---

## Known Failure Modes

**ExpressionEditor**:
- Dangly accessories (earrings, front hair strands) warp incorrectly with head rotation
- Extreme angles (yaw >30°, pitch >25°) produce geometric tears
- Mitigation: clamp pose sliders to ±20°, render pose verbs through Live2D rig instead

**FLUX Kontext**:
- Occasional refusal to edit (returns near-identity) on subtle verbs — bump prompt specificity
- Style drift when verb mentions "anime" or art-style words — keep verbs purely semantic
- Mitigation: verb prompts describe expression only, never style

**Style transfer (anime→photo)**:
- Large anime eyes bleed through as disproportionate human eyes at low denoise
- Mouth closure state can flip (anime closed-line mouth → photo open mouth)
- Mitigation: denoise ≥0.55, Canny strength ≥0.4 on mouth region, batch validation via MediaPipe

**Identity drift across batch**:
- Generating 60 verbs from one reference can drift hair length / clothing over the batch
- Mitigation: FLUX Kontext's layout-preservation is strongest here; ExpressionEditor has zero drift (warp-based)

---

## First-Experiment Proposal: 10-Sample Validation

**Goal**: Verify that ExpressionEditor-generated images produce MediaPipe blendshape responses that match the intended verb.

**Setup** (smallest testable workflow):

1. Pick 1 reference human photo (front-facing, neutral expression, good lighting).
2. Install `ComfyUI-AdvancedLivePortrait`.
3. Define 10 test verbs mapped to ExpressionEditor sliders:

| # | Verb | ExpressionEditor params | Expected MediaPipe response |
|---|---|---|---|
| 1 | neutral | all defaults | eyeBlinkL≈0, jawOpen≈0, mouthSmileL≈0 |
| 2 | close_both_eyes | blink=-15 | eyeBlinkL>0.7, eyeBlinkR>0.7 |
| 3 | wink_left | wink=20 (left side) | eyeBlinkL>0.7, eyeBlinkR<0.2 |
| 4 | smile_slight | smile=0.5 | mouthSmileL>0.3, mouthSmileR>0.3 |
| 5 | smile_wide | smile=1.2, aaa=40 | mouthSmile*>0.6, jawOpen>0.2 |
| 6 | mouth_open | aaa=80 | jawOpen>0.5 |
| 7 | look_left | pupil_x=-12 | eyeLookOutLeft>0.5 |
| 8 | look_up | pupil_y=10 | eyeLookUp*>0.4 |
| 9 | brow_raise | eyebrow=12 | browInnerUp>0.4 |
| 10 | surprised | blink=3, aaa=60, eyebrow=12 | eyeWide*>0.3, jawOpen>0.4, browInnerUp>0.4 |

4. Run each verb through ExpressionEditor, save 10 output images.
5. Run MediaPipe FaceLandmarker on each output, extract blendshapes.
6. Score: for each verb, check if expected blendshapes exceed thresholds.

**Success criterion**: ≥8/10 verbs produce MediaPipe blendshapes within expected ranges.

**If successful**: scale to full 30–60 verb library, integrate into `mlp/data/render_verbs.py`.

**If failed on >3 verbs**: diagnose whether ExpressionEditor or MediaPipe is the weak link; fall back to FLUX Kontext for failing verbs.

**Compute budget**: 10 samples × ~100ms = 1 second of ExpressionEditor time + MediaPipe inference. Can run in <5 minutes total including setup.

---

## Open Questions

1. **Does ExpressionEditor preserve expression when the source is already stylized (anime render)?** — Worth testing: Hiyori render at neutral → ExpressionEditor(smile) → MediaPipe. If yes, we can skip the human-photo reference entirely.
2. **How many unique reference photos do we need per template?** — Single reference may give an MLP that overfits to that face's landmark geometry. Suspect 5–20 diverse references per template.
3. **FLUX Kontext dev vs pro for edit quality** — no formal benchmark found; dev is "good enough" per community but empirical test on our verbs would help.
4. **MediaPipe reliability gradient** — does MediaPipe work better on ExpressionEditor-warped images than on FLUX Kontext-generated ones? Likely yes (warp preserves photo statistics) but untested.

---

## Sources

[1] Guo et al. "LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control." KwaiVGI, 2024. https://liveportrait.github.io/ (Retrieved 2026-04-05)

[2] r/StableDiffusion community threads on ComfyUI-AdvancedLivePortrait, 2024–2025. https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait (Retrieved 2026-04-05)

[3] Black Forest Labs. "FLUX.1 Kontext" model card and documentation, 2025. https://bfl.ai/models/flux-kontext (Retrieved 2026-04-05)

[4] Alibaba Cloud / Qwen team. "Qwen-Image-Edit" model release, 2025. https://huggingface.co/Qwen/Qwen-Image-Edit (Retrieved 2026-04-05)

[5] Google DeepMind. "Gemini 2.5 Flash Image" (Nano Banana) announcement, 2025. https://deepmind.google/ (Retrieved 2026-04-05)

[6] ComfyUI FLUX Kontext workflow examples, ComfyUI-Examples repo, 2025. https://comfyanonymous.github.io/ComfyUI_examples/flux/ (Retrieved 2026-04-05)

[7] Civitai "anime2photo" workflow collection, 2024–2025. https://civitai.com/ (Retrieved 2026-04-05)

[8] Community SDXL anime-to-realistic ControlNet guides, 2024–2025. Multiple Civitai articles. (Retrieved 2026-04-05)

[9] Shakker-Labs / XLabs FLUX ControlNet Union Pro 2.0, 2025. https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0 (Retrieved 2026-04-05)

[10] Community benchmarks of InstantID vs PuLID vs IP-Adapter FaceID, r/StableDiffusion and blog posts, 2024–2025. (Retrieved 2026-04-05)
