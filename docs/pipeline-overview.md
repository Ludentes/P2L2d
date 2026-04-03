# AI Live2D Character Generation Pipeline — Project Overview

**Created:** 2026-04-03  
**Status:** Research phase complete, implementation not started  
**Goal:** From a single portrait image → fully animatable Live2D VTuber model, driven by face tracking + BCI signals in real time.

---

## What We're Building

An automated pipeline that takes a single anime-style portrait and produces a rigged Live2D model (`.moc3` + `.model3.json`) that:

1. Animates in real time via MediaPipe face tracking
2. Supports the Muse 2 EEG-specific parameters (jaw clench, focus, relaxation, heartbeat)
3. Works in VTube Studio without manual rigging

This is a cleaned-up, extended version of what **Textoon** and **CartoonAlive** describe in their papers — with portrait input (not text), BCI parameter slots built into the base rig, and an open path to commercial use.

---

## The Foundation: Textoon + HaiMeng

### What Textoon actually does

Textoon does **not** auto-rig. It works by:

1. Parsing text input (Qwen2.5-1.5B) → structured clothing/appearance attributes
2. Generating textures with SDXL + ControlNet within known UV regions
3. **Copying the pre-baked HaiMeng `.moc3` template** and swapping texture sheets

The rig is entirely pre-built. The "generation" is texture substitution within a hardcoded atlas.

### HaiMeng dataset

The base character behind Textoon. Female + male characters, rigged professionally in Live2D Cubism.

**Texture structure:**

| Sheet | Content | Size |
|-------|---------|------|
| `texture_00` | Body base, thighs, calves, eyes | 4096×4096 |
| `texture_01` | All hair variants | 4096×4096 |
| `texture_02` | Sleeves | 4096×4096 |
| `texture_03` | Skirts | 4096×4096 |
| `texture_04` | Trousers | 4096×4096 |
| `texture_05` | Shirt body | 4096×4096 |
| `texture_06` | Unused/extra | 4096×4096 |
| `texture_07` | Boots | 4096×4096 |
| `texture_08` | Breast variant shirts | 4096×4096 |

**Parameters (107 total):**
- Standard Live2D face/body params (angle, eyes, brows, body angle)
- **24 ARKit mouth blendshapes** (`ParamJawOpen`, `MouthSmileLeft`, `MouthFrownLeft`, `MouthCheekPuff`, etc.)
- 7 boolean rig-toggle params for clothing switching (ponytail, skirt, trousers, breast size)
- ~50 internal body/physics params

**Public vs. EULA-gated:**
- `assets/haimeng/` in the public GitHub repo contains: sprite catalog (66 clothing/hair PNGs), eye color variants (24), basemaps — **already accessible**
- `runtime20250208/` (`.moc3`, `.model3.json`, 9 texture sheets) — behind EULA

**HaiMeng EULA:** Restrictive (academic/research only, no commercial redistribution). User has accepted this — outputs derived from HaiMeng are fine as academic-only. Submit the EULA form to `github.com/Human3DAIGC/Textoon` to get runtime access.

### Textoon code quality

The codebase is production-quality Python. Key modules:
- `utils/transfer_part_texture.py` — UV crop/paste between generation canvas and texture sheets
- `utils/generate_texture.py` — SDXL + ControlNet generation per component
- `utils/sam2_predict.py` — SAM2 segmentation for component boundaries
- `utils/inpaint.py` — LaMa inpainting for occluded regions (back hair, body under clothes)

The Qwen2.5-1.5B text parser and SDXL generation components are replaceable — for portrait input (our goal), these become portrait segmentation + texture extraction rather than generation from scratch.

**Models available:** Via HuggingFace (`human3daigc/textoon`). All weights are released.

---

## The Target Architecture: CartoonAlive

CartoonAlive (by the same team, July 2025) extends Textoon to portrait input. Code is **not released** — the GitHub repo is a placeholder. The paper describes the architecture fully enough to reimplement.

### Key addition over Textoon

A **4-layer MLP** that maps 478 MediaPipe facial landmarks → 107 Live2D parameter values.

Training data generation:
1. Randomly sample Live2D parameter sets (covering the parameter space)
2. Render each parameter state via PyGame + Cubism SDK → get pixel output
3. Run MediaPipe on the rendered face → get landmark ground truth
4. Store (landmarks → parameter set) pairs; 100,000 samples total

This MLP generalizes across face shapes and expression variations.

### CartoonAlive pipeline (to implement)

```
Portrait image
    ↓ face crop + segmentation
Component extraction (hair, skin, clothing regions)
    ↓ SDXL img2img or direct texture crop + inpaint
Texture sheets (matching HaiMeng atlas UV structure)
    ↓ copy HaiMeng .moc3 template
Assembled .model3.json with new textures
    ↓ MediaPipe at runtime
Landmark stream → MLP → Live2D parameter stream
    ↓ live2d-py or VTS
Animated character
```

---

## BCI Integration

The Muse 2 adds signals that face tracking cannot provide. These map to Live2D custom parameters.

### Unique BCI signals

| Signal | Source | What no face tracker can do |
|--------|--------|----------------------------|
| `MuseClench` | EMG (jaw electrode) | Jaw clench without visible face movement |
| `MuseFocus` | EEG theta/beta ratio | Concentration level as ambient expression |
| `MuseRelaxation` | EEG alpha | Calm/drowsy ambient expression |
| `MuseHeartbeat` | PPG | Heartbeat-driven subtle breathing animation |

### Design requirement

**The base HaiMeng rig needs BCI parameter slots added before we freeze the template.** These are custom parameters in the `.model3.json` that the Muse VTuber Bridge sends via VTS custom parameter injection. Because HaiMeng's `.moc3` binary is not editable, this means either:

1. Submit EULA → get `.cmo3` source → add params in Cubism Editor → re-export `.moc3` *(preferred if team has Cubism Editor access)*
2. Add the parameters purely in the runtime (VTS handles injection of custom parameters even if the rig has no deformers for them — they just don't drive anything until the rig is wired up)

Option 2 is the immediate path: VTS custom parameters can be injected without rig changes; the parameters exist in VTS but have no visual effect until a rigger wires them into the rig. This means we can build and test the BCI pipeline now and defer the rig wiring.

---

## Official Sample Models vs. HaiMeng (Compatibility)

Investigated as a fallback if HaiMeng EULA access fails. **Not compatible without substantial re-engineering.**

Three blockers:
1. **Texture architecture**: official samples use 1–2 × 2048 px sheets (4–8M pixels total); HaiMeng uses 9 × 4096 px (150M pixels). The `transfer_part_texture.py` UV coordinate tables would need complete redefinition.
2. **No ARKit**: official samples use generic `ParamMouthOpenY`/`ParamMouthForm`. No 24-blendshape mouth rig.
3. **No garment toggles**: no boolean visibility parameters for clothing variants.

Fixing all three requires Cubism Editor work (multi-week). Best candidate if this path is needed: **Natori** (96 params, 32 parts, `.cmo3` available from Live2D download page).

Full analysis: `docs/research/2026-04-03-live2d-official-samples-vs-haimeng.md`

---

## Neural Warp Alternative

If Live2D output is not required — or for a quick demo:

**FasterLivePortrait** (MIT license, weights on HuggingFace `warmshao/FasterLivePortrait`):
- 30+ FPS on RTX 3090 with TensorRT
- Real-time webcam mode: `python run.py --src_image portrait.jpg --dri_video 0 --cfg configs/trt_infer.yaml --realtime`
- Output is a warped version of the source portrait, NOT a Live2D rig
- No virtual camera built in — pipe to OBS via `v4l2loopback` + `ffmpeg`

This is "ship today" for a demo, not the target for production VTubing.

---

## Implementation Phases

### Phase 0 — Access
- Submit HaiMeng EULA → get `runtime20250208/` (`.moc3` + textures)
- Clone `github.com/Human3DAIGC/Textoon`
- Verify Textoon pipeline runs end-to-end with text input on this machine (baseline)

### Phase 1 — Portrait input
- Replace Textoon's Qwen2.5 text parser + SDXL generation with portrait pipeline:
  - Anime face segmentation (anime-seg) for hair extraction
  - HumanParser node for clothing regions
  - SDXL img2img for each component (or direct texture crop if art style is clean enough)
  - SAM2 segmentation remains useful for boundary refinement
- Input: single portrait. Output: `.model3.json` using HaiMeng rig.

### Phase 2 — CartoonAlive MLP
- Implement 4-layer MLP (478 landmarks → 107 Live2D params)
- Generate training data: random parameter sampling → PyGame+Cubism SDK render → MediaPipe landmark extraction
- 100,000 samples, train to convergence
- Replace Textoon's hardcoded MediaPipe mapping with learned MLP

### Phase 3 — BCI parameter slots
- Wire `MuseClench`, `MuseFocus`, `MuseRelaxation`, `MuseHeartbeat` into the rig
- Either via Cubism Editor (`.cmo3` source) or as VTS custom parameter overlays
- Test end-to-end: Muse 2 → muse-vtuber bridge → VTS → animated character

### Phase 4 — Production loop
- MediaPipe → MLP → Live2D parameter injection → live2d-py or VTS
- Virtual camera out via VTS built-in or `v4l2loopback`
- OBS integration (transparent background, Game Capture)

---

## Licensing Summary

| Component | License | Notes |
|-----------|---------|-------|
| Textoon code | Apache 2.0 | Freely usable, modifiable |
| HaiMeng dataset | EULA (academic) | Outputs may be academic-only; user accepts this |
| Live2D Cubism SDK | Proprietary (free non-commercial) | Required for `.moc3` render during training data gen |
| live2d-py | MIT | Python bindings to Cubism Native SDK |
| Inochi2D | BSD 2-clause | Alternative if Cubism licensing becomes a problem |
| FasterLivePortrait | MIT | Neural warp fallback, no Live2D |
| MediaPipe | Apache 2.0 | Face landmark extraction |
| SDXL / ControlNet | Apache 2.0 / CreativeML | Image generation for textures |
| SAM2 | Apache 2.0 | Segmentation |

Commercial path (if ever needed): replace Cubism SDK with Inochi2D for training renders; negotiate separate Cubism license for commercial distribution. For current non-commercial use, existing stack is clean.

---

## Related Research Docs

| File | Topic |
|------|-------|
| `docs/research/2026-04-03-comfyui-live2d-art-pipeline.md` | ComfyUI → PSD layer decomposition (See-through, CartoonAlive overview) |
| `docs/research/2026-04-03-realtime-portrait-animation-vtubing.md` | Neural warp vs parametric paradigms, full ecosystem map |
| `docs/research/2026-04-03-comfyui-claude-code-integration.md` | ComfyUI REST API + MCP server for automation |
| `docs/research/2026-04-03-live2d-official-samples-vs-haimeng.md` | Detailed texture/parameter compatibility analysis |
| `docs/research/2026-04-03-vtube-studio-parameter-injection.md` | VTS WebSocket API, custom parameter injection |
| `docs/research/2026-04-03-vtube-studio-lip-sync.md` | Lip sync options in VTS |
| `docs/muse-vtuber-vts-setup-guide.md` | Muse 2 → VTS integration (existing, working) |
