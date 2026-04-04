# Pipeline Architecture Design

**Date:** 2026-04-04  
**Status:** Design complete  
**Scope:** Full portrait-to-live2d system — five subsystems, one design doc, five implementation plans

---

## Goal

Single portrait image → animatable Live2D `.moc3` model with:
- Style-matched textures (inherits input portrait style via Flux Kontext)
- BCI parameter slots pre-injected
- Face tracking via CartoonAlive MLP at 60fps
- Output loadable in VTube Studio without manual rigging

---

## Source Material

**CartoonAlive (Human3DAIGC, July 2025, arXiv:2507.17327)** — 4 stages:
1. Face alignment (horizontal eye alignment, PnP keypoint detection)
2. Parameter estimation (MLP: keypoints → Live2D position/scale params, range ±30)
3. Face repainting (mask-guided inpaint to eliminate animation artifacts)
4. Hair extraction (segmentation, handle bangs occluding eyebrows)

Training: 100,000 synthetic pairs rendered with PyGame + Cubism SDK. Code **unreleased** — reimplemented here from the paper.

**Textoon (Human3DAIGC, Apache 2.0)** — reused as submodule:
- `utils/transfer_part_texture.py` — UV crop/paste with edge case handling (thigh rotation, sleeve interpolation)
- `assets/model_configuration.json` — exact UV coords for 20 components across haimeng (female) and shenxinhui (male)
- `scripts/mediapipe_live2d.py` — direct blendshape lookup (kept as fallback; MLP is primary)

---

## Approach: Hybrid (Textoon as submodule, new layers around it)

Textoon's UV transfer logic and coordinate table are correct and production-tested. No reason to rewrite them. Everything else (generation, MLP, BCI, runtime bridge) is new.

Textoon is cloned as a git submodule (`textoon/`). The new pipeline imports from it.

---

## Project Layout

```
portrait-to-live2d/
  comfyui/                    ← Subsystem 1: ComfyUI HTTP client
    client.py                 ← ComfyUIClient (health, upload, submit, wait, download)
    workflows/                ← API-format JSON workflow files
    test_connection.py

  pipeline/                   ← Subsystem 2: Portrait → texture pipeline
    __init__.py
    run.py                    ← CLI: python -m pipeline.run portrait.jpg [--style anime]
    stages/
      detect.py               ← Face detection + alignment (MediaPipe)
      segment.py              ← 20-component masks (SAM2 + HumanParser)
      generate.py             ← Flux Kontext generation per component (→ comfyui/)
      transfer.py             ← UV paste (wraps textoon/utils/transfer_part_texture.py)
      assemble.py             ← .model3.json from HaiMeng template
    config.py                 ← Paths, model IDs, generation params

  mlp/                        ← Subsystem 3: CartoonAlive MLP
    model.py                  ← 4-layer MLP (478×2 landmarks → 107 params)
    train.py                  ← Synthetic data gen + training loop
    infer.py                  ← Runtime inference (<16ms)
    data/
      generate_samples.py     ← PyGame + live2d-py renders → (landmarks, params) pairs

  rig/                        ← Subsystem 4: Template management + BCI injection
    template.py               ← Copy HaiMeng .moc3 + .model3.json, update texture refs
    bci_inject.py             ← Add ParamJawClench/FocusLevel/Relaxation/Heartbeat slots

  runtime/                    ← Subsystem 5: Live face tracking + param bridge
    mediapipe_stream.py       ← Webcam → 478 landmarks (async)
    param_bridge.py           ← landmarks → mlp/infer → VTube Studio WebSocket
    bci_bridge.py             ← Muse VTuber Bridge → BCI param smooth-send
    fallback.py               ← Direct blendshape mapping (if MLP unavailable)

  textoon/                    ← Git submodule: github.com/Human3DAIGC/Textoon
  pyproject.toml              ← uv project, deps: httpx, mediapipe, torch, live2d-py, etc.
```

---

## Subsystem 1: ComfyUI Client

**Already designed** (`docs/superpowers/specs/2026-04-04-project-init-design.md`).

`ComfyUIClient` wraps six endpoints: `health`, `list_models`, `upload_image`, `submit`, `wait` (WebSocket completion), `download`. Used exclusively by `pipeline/stages/generate.py`.

---

## Subsystem 2: Portrait Pipeline

### Stage: detect.py

Input: portrait path. Output: aligned face crop (PIL Image, square, eyes horizontal).

Uses MediaPipe FaceMesh for 5 keypoint groups (eyes, nose, mouth, eyebrows, contour). Applies affine transform to align left/right eye horizontally at a fixed y-offset. Rejects images with 0 faces or >1 face with a clear error and instructions to crop.

Minimum input resolution: 512px. Recommended: 1024px+.

### Stage: segment.py

Input: aligned face crop. Output: dict of 20 component masks (one per entry in `model_configuration.json`).

Uses SAM2 for initial region proposals, HumanParser for semantic labels (face, hair, clothing, legs, shoes). Maps semantic regions to the 20 Textoon component names. If a component mask is empty (e.g. boots not visible), marks as `missing` — the transfer stage skips it and the template's default texture shows.

### Stage: generate.py

Input: aligned face crop + 20 component masks + optional style override. Output: generated texture image per component.

Uses Flux Kontext [dev] via `comfyui/client.py`. Workflow: upload portrait as reference image → generate each component with a text prompt describing the component and referencing the portrait's style → download result. Style override (`--style anime`) activates the Diving-Illustrious flat-anime LoRA pass instead.

All components are generated in a single ComfyUI queue batch where possible to maximise GPU utilisation. Target: <25 min total for all 20 components on RTX 5090.

### Stage: transfer.py

Input: generated component images + component masks. Output: 9 assembled texture sheets (4096×4096 PNG each).

Thin wrapper around `textoon/utils/transfer_part_texture.py` with `model_configuration.json` coordinates. Handles thigh rotation and sleeve interpolation exactly as Textoon does.

### Stage: assemble.py

Input: 9 texture sheets + HaiMeng template path. Output: output directory with `.moc3` (copied), `textures/` (placed), `.model3.json` (texture refs updated). HaiMeng uses 9 sheets indexed texture_00–texture_08; `texture_06` is an accessories/prop sheet that is present but may not be visibly altered for all character configurations.

Copies the HaiMeng template bundle, rewrites the `Textures` array in `.model3.json` to point at the new texture files. Then calls `rig/bci_inject.py` to add BCI parameter slots.

**Error handling:** any stage failure writes a `pipeline_error.json` to the output dir with the stage name, error message, and any partial files — then cleans up partial texture files. No silent failures.

---

## Subsystem 3: CartoonAlive MLP

### Architecture

The paper's baseline is a 4-layer MLP (956→512→256→128→107 with ReLU). We improve on it in three ways that don't violate the <16ms CPU constraint:

**Model:**
```
InputNorm(956)                        ← learned per-coordinate mean/std, baked in
Linear(956, 512) → LayerNorm(512) → GELU
Linear(512, 256) → LayerNorm(256) → GELU + skip(Linear(512,256))   ← residual
Linear(256, 128) → LayerNorm(128) → GELU
Linear(128, 107)
OutputDenorm(107)                     ← per-param mean/std, baked in
```

- **LayerNorm over BatchNorm**: no running-statistics divergence at inference; more stable for variable-length input sequences.
- **GELU over ReLU**: smoother gradients for regression targets.
- **One residual skip** (512→256 level): empirically stabilises training without meaningful inference overhead. No skip at 256→128 — the input dimensionality drop is large enough that a direct skip adds noise.
- **Baked normalisation**: `InputNorm` and `OutputDenorm` are non-trainable layers whose weights are computed from the training set and frozen into the model before export. Inference requires no external stats files.

Total parameters: ~600k. CPU inference target: <8ms on a mid-range laptop (headroom over the 16ms budget).

**Parameter groups**: the 107 Live2D parameters split into four groups by symmetry and range characteristics:
- Face angle/position (6 params, range ±30): `ParamAngleX/Y/Z`, `ParamBodyAngleX/Y/Z`
- Bilateral eye/brow (32 params, left+right symmetric): eyes, brows, wink, glasses
- Mouth (20 params): open, form, smile, etc.
- Body/misc (49 params): chest, hair physics outputs, accessory toggles

The single output head handles all 107 jointly. A multi-head design (one head per group) is listed as a future experiment — skip for now, adds complexity without proven benefit.

### Training Data

#### Sampling strategy

Pure uniform random sampling across the ±30 parameter space produces a distribution of faces that are heavily weighted toward bizarre edge cases (all params at extremes simultaneously). We use **stratified sampling**:

1. **Base poses** (20%): 8 canonical expressions (neutral, smile, surprised, sad, blink-L, blink-R, mouth-open, head-left) × 12,500 = 100,000 samples, each with Gaussian perturbation σ=3 per param around the base.
2. **Natural random** (50%): all params sampled from `Normal(0, 8)` clipped to valid range — centred on neutral, tail-heavy enough to reach extremes.
3. **Extreme exploration** (20%): uniform random over full range to ensure the MLP learns boundary behaviour.
4. **Symmetric pairs** (10%): for each bilateral param group, sample one side and mirror to the other — enforces symmetric training pressure.

Total: 100,000 samples before filtering.

#### Rendering pipeline (`mlp/data/generate_samples.py`)

```
for each sampled param set:
  1. Apply params to HaiMeng rig via live2d-py
  2. Render to RGBA at 512×512 (matches MediaPipe's optimal range)
  3. Composite over random background (uniform colour, sampled from 20 pastel values)
     — MediaPipe is sensitive to low-contrast face-on-white images
  4. Apply mild augmentation:
       - Gaussian blur σ ∈ [0, 0.8px]  (simulates lens softness)
       - JPEG compression quality ∈ [85, 100]  (simulates webcam compression)
       - Brightness jitter ±10%
  5. Run MediaPipe FaceMesh on the augmented frame
  6. If detection confidence < 0.8 OR detected landmarks < 468: discard, resample
  7. Extract 478 landmarks (x, y) in normalised image coords [0,1]
  8. Save (landmarks[478×2], params[107]) pair
```

Discarded samples are resampled immediately — the dataset always hits exactly 100,000 valid pairs. Log the discard rate per param region (high discard = that region produces undetectable faces; flag for review).

#### Normalisation statistics

After generating all 100,000 samples, compute:
- Per-landmark-coordinate: mean, std (across all samples) → used by `InputNorm`
- Per-param: mean, std (across all samples) → used by `OutputDenorm`

These are computed once and frozen. The model exports with these as constants (e.g. via `torch.nn.functional` in the forward pass, or as registered buffers).

#### Train/validation split

90k / 10k (stratified: validation set contains samples from all four sampling buckets proportionally). No test set — held-out validation is sufficient; the real test is live performance on a webcam stream.

#### Training

- Optimizer: AdamW, lr=3e-4, weight decay=1e-4
- Schedule: cosine annealing over 100 epochs, no warmup
- Loss: MSE on normalised outputs (so all 107 params contribute equally regardless of scale)
- Batch size: 512 (GPU training on RTX 5090; inference is CPU-only)
- Early stop: if validation RMSE hasn't improved for 10 epochs, stop
- Target: validation RMSE <0.05 in normalised space (≈ <2 param units in original scale for the ±30-range params)
- Log per-group RMSE separately to identify weak spots

### Runtime

`mlp/infer.py` exposes `predict(landmarks: np.ndarray) -> np.ndarray`. Loads model once at startup with `torch.jit.script` for optimised CPU execution. Benchmarks inference at startup (median of 50 cold calls) — falls back to `runtime/fallback.py` direct mapping if median >16ms. The model runs on CPU; GPU is reserved for ComfyUI.

---

## Subsystem 4: BCI Injection

### bci_inject.py

Reads `.model3.json`, checks if BCI params already exist (idempotent), appends to `Parameters` array:

```json
{"Id": "ParamJawClench",  "Min": 0, "Max": 1, "Default": 0},
{"Id": "ParamFocusLevel", "Min": 0, "Max": 1, "Default": 0},
{"Id": "ParamRelaxation", "Min": 0, "Max": 1, "Default": 0},
{"Id": "ParamHeartbeat",  "Min": 0, "Max": 1, "Default": 0}
```

The `.moc3` binary is **never modified** — BCI params are runtime-injected via VTube Studio's custom parameter API. Deformers for these parameters are wired in Cubism Editor separately (future work).

---

## Subsystem 5: Runtime Bridge

### mediapipe_stream.py

Async loop: webcam frame → MediaPipe FaceMesh → 478 landmarks as `np.ndarray`. Emits via asyncio queue at ~60fps. Drops frames if downstream is slower (no queue buildup). Logs confidence; if confidence <0.5 for >100ms, emits `LOW_CONFIDENCE` signal but continues.

### param_bridge.py

Consumes landmark queue. Calls `mlp/infer.py` → 107 params. Sends to VTube Studio via WebSocket (`vtube-studio` Python library). On `LOW_CONFIDENCE`: holds last values. On MLP timeout: falls back to `runtime/fallback.py`.

### bci_bridge.py

Receives BCI parameter values from the Muse VTuber Bridge (existing system, ZyphraExps). Smooth-sends to VTube Studio WebSocket. On disconnect: holds last values 2s, then interpolates to 0.0 over 1s. Reconnect loop runs in background.

---

## Implementation Order

These subsystems have the following dependencies:

```
Subsystem 1 (ComfyUI client)   → no deps, implement first
Subsystem 2 (Portrait pipeline) → depends on 1 (ComfyUI) + textoon submodule
Subsystem 3 (MLP)              → depends on live2d-py + HaiMeng template access
Subsystem 4 (BCI injection)    → depends on 2 (assemble.py output)
Subsystem 5 (Runtime bridge)   → depends on 3 (MLP infer.py)
```

Each subsystem gets its own implementation plan. The suggested order:

1. **Subsystem 1** — ComfyUI client + smoke test (already partially designed)
2. **Subsystem 2** — Portrait pipeline (core value, unblocks everything visible)
3. **Subsystem 4** — BCI injection (fast win, independent once Subsystem 2 assembles)
4. **Subsystem 3** — MLP (needs HaiMeng EULA access for training data gen)
5. **Subsystem 5** — Runtime bridge (needs MLP + VTube Studio integration test)

---

## Open Questions / Deferred

- **HaiMeng EULA access**: Subsystems 2 (assemble) and 3 (MLP training) need the `.cmo3`/`.moc3` runtime files. Until access is granted, Subsystem 2 can be tested with the public `assets/haimeng/` sprite data and a placeholder rig.
- **Flux ControlNet for structure guidance**: ControlNet for Flux is immature — deferred. Use img2img (denoise 0.35–0.45) for structure preservation instead.
- **Style detection**: Automatic style detection from input portrait (to pick Flux prompt style) is deferred. For now, default = "match input style via Kontext reference"; explicit `--style anime` override available.
- **Male rig (shenxinhui)**: `model_configuration.json` has both haimeng and shenxinhui UV tables. Male rig support is deferred until female rig pipeline is working end-to-end.
