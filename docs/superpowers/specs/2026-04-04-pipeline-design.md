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

## Rig Configuration

The pipeline is rig-agnostic via a `RigConfig` dataclass in `rig/config.py`. All rig-specific constants — parameter list, texture count, UV coordinate file — live here. Subsystems 2–5 take a `RigConfig` and do not hard-code model names or counts.

```python
@dataclass
class RigConfig:
    name: str                      # "hiyori" | "haimeng"
    moc3_path: Path                # absolute path to .moc3
    model3_json_path: Path         # absolute path to .model3.json
    textures: list[Path]           # all texture sheet paths (2 for hiyori, 9 for haimeng)
    param_ids: list[str]           # ordered list of param IDs from cdi3.json/model3.json
    uv_config: Path | None         # model_configuration.json equiv; None = skip UV transfer
```

Two pre-built configs are provided:

**`RIG_HIYORI`** — development rig. Available immediately at:
`~/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/hiyori_vts/`
- 74 parameters (same core face params as HaiMeng; no ARKit blendshapes)
- 2 × 2048px texture sheets
- `uv_config = None` (UV transfer stage skipped in dev; Hiyori textures used as-is for assembly testing)
- Licensed for personal/research use (Live2D sample model)

**`RIG_HAIMENG`** — production rig. Gated behind Textoon EULA (submit via university account).
- 107 parameters, 9 × 4096px sheets, full UV config via `textoon/assets/model_configuration.json`

The MLP trained on Hiyori is geometry-specific and will need retraining on HaiMeng. The training *infrastructure* (data generation, training loop, model architecture) transfers without change — only the output dimensionality differs (`len(rig_config.param_ids)`).

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
      segment.py              ← component masks (SAM2 + HumanParser; count from rig config)
      generate.py             ← Flux Kontext generation per component (→ comfyui/)
      transfer.py             ← UV paste (wraps textoon/utils/transfer_part_texture.py; skipped if rig_config.uv_config is None)
      assemble.py             ← output bundle from rig template (rig-agnostic via RigConfig)
    config.py                 ← Paths, model IDs, generation params

  mlp/                        ← Subsystem 3: CartoonAlive MLP
    model.py                  ← 4-layer MLP (478×2 landmarks → N params; N from RigConfig)
    train.py                  ← Synthetic data gen + training loop
    infer.py                  ← Runtime inference (<16ms)
    data/
      generate_samples.py     ← live2d-py renders → (landmarks, params) pairs

  rig/                        ← Subsystem 4: Template management + BCI injection
    config.py                 ← RigConfig dataclass + RIG_HIYORI + RIG_HAIMENG presets
    template.py               ← Copy rig bundle, update texture refs in .model3.json
    bci_inject.py             ← Register BCI custom params via VTube Studio WebSocket API

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

Input: aligned face crop + `RigConfig`. Output: dict of component masks keyed by component name.

Component names come from `rig_config.uv_config` (when present) or from a fixed semantic set when `uv_config is None` (Hiyori dev mode: face, hair, body, clothing). Uses SAM2 for initial region proposals, HumanParser for semantic labels. If a component mask is empty, marks as `missing` — the transfer stage skips it and the template's default texture shows.

### Stage: generate.py

Input: aligned face crop + component masks + optional style override. Output: generated texture image per component.

Uses Flux Kontext [dev] via `comfyui/client.py`. Workflow: upload portrait as reference image → generate each component with a text prompt describing the component and referencing the portrait's style → download result. Style override (`--style anime`) activates the Diving-Illustrious flat-anime LoRA pass instead.

All components generated in a single ComfyUI queue batch where possible. Target: <25 min total on RTX 5090.

### Stage: transfer.py

Input: generated component images + component masks + `RigConfig`. Output: assembled texture sheets.

**When `rig_config.uv_config` is not None**: thin wrapper around `textoon/utils/transfer_part_texture.py` with the UV coordinates from the config file. Handles thigh rotation and sleeve interpolation exactly as Textoon does.

**When `rig_config.uv_config is None`** (Hiyori dev mode): skips UV transfer entirely. Generated component images are resized to the rig's texture dimensions and placed as-is. This produces a visually rough output sufficient for pipeline integration testing.

### Stage: assemble.py

Input: texture sheets + `RigConfig`. Output: output directory with `.moc3` (copied), `textures/` (placed), `.model3.json` (texture refs updated).

Copies the rig bundle from `rig_config.moc3_path` / `rig_config.model3_json_path`, rewrites the `Textures` array in `.model3.json` to point at the new texture files. Does not call `bci_inject.py` — BCI params are registered at runtime via the VTS API, not baked into the model file.

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

**Output dimensionality** is `len(rig_config.param_ids)` — **74 for Hiyori, 107 for HaiMeng**. The model architecture is identical; only the final `Linear` layer's output size changes. The trained weights are rig-specific and not transferable.

**Parameter groups** (Hiyori / HaiMeng share the same logical groups, Hiyori just has fewer in each):
- Face angle/position: `ParamAngleX/Y/Z`, `ParamBodyAngleX/Y/Z`
- Bilateral eye/brow: eyes, brows (symmetric left/right)
- Mouth: open, form, smile, etc.
- Body/hair/misc: physics outputs, accessory toggles

The single output head handles all params jointly. A multi-head design is a future experiment.

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
  1. Apply params to rig via live2d-py (rig_config.moc3_path)
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
  8. Save (landmarks[478×2], params[N]) pair  ← N = len(rig_config.param_ids)
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
- Loss: MSE on normalised outputs (so all N params contribute equally regardless of scale)
- Batch size: 512 (GPU training on RTX 5090; inference is CPU-only)
- Early stop: if validation RMSE hasn't improved for 10 epochs, stop
- Target: validation RMSE <0.05 in normalised space (≈ <2 param units in original scale for the ±30-range params)
- Log per-group RMSE separately to identify weak spots

### Runtime

`mlp/infer.py` exposes `predict(landmarks: np.ndarray) -> np.ndarray`. Loads model once at startup with `torch.jit.script` for optimised CPU execution. Output length matches the rig the model was trained on (`len(rig_config.param_ids)`). Benchmarks inference at startup (median of 50 cold calls) — falls back to `runtime/fallback.py` direct mapping if median >16ms. The model runs on CPU; GPU is reserved for ComfyUI.

---

## Subsystem 4: BCI Injection

### Design: VTS Custom Parameters API (not model file modification)

The earlier design proposed appending to the `Parameters` array in `.model3.json`. This approach has two problems:
1. Hiyori's `.model3.json` has no `Parameters` array — parameters are in the separate `.cdi3.json`.
2. Custom parameters visible in VTube Studio don't need to be in `.model3.json` at all — VTS's custom parameter API creates them at runtime.

The correct design is: **BCI parameters are registered via the VTube Studio WebSocket API on startup**. No model files are modified. This works with any rig, including models the user didn't create (Hiyori, commissioned rigs, future HaiMeng).

The `.moc3` is never touched. Deformers for BCI params are wired in Cubism Editor by the user (future work) — the pipeline only ensures the parameter slots exist and receive values.

### bci_inject.py

Connects to VTube Studio WebSocket. For each BCI parameter:

1. Calls `InjectParameterDataV2` with `createNewParameters: true` — creates the custom parameter if it doesn't exist, or reuses it if already present (idempotent).
2. Sends an initial value of 0.0.

Parameters registered:

| ID | Description |
|---|---|
| `ParamJawClench` | EMG jaw clench (0–1) |
| `ParamFocusLevel` | EEG theta/beta concentration (0–1) |
| `ParamRelaxation` | EEG alpha relaxation (0–1) |
| `ParamHeartbeat` | PPG pulse (0–1) |

On exit: parameters are left in VTube Studio (persist across sessions). Running the tool again is a no-op (VTS deduplicates by ID).

### template.py

Copies the rig bundle (`.moc3`, `.model3.json`, textures) from `rig_config` paths to the output directory. Rewrites the `Textures` array in the output `.model3.json` to reference the new texture file paths. Does not register BCI params — that is `bci_inject.py`'s job.

---

## Subsystem 5: Runtime Bridge

### mediapipe_stream.py

Async loop: webcam frame → MediaPipe FaceMesh → 478 landmarks as `np.ndarray`. Emits via asyncio queue at ~60fps. Drops frames if downstream is slower (no queue buildup). Logs confidence; if confidence <0.5 for >100ms, emits `LOW_CONFIDENCE` signal but continues.

### param_bridge.py

Consumes landmark queue. Calls `mlp/infer.py` → N params (N = rig param count). Maps param array to named VTS parameters using `rig_config.param_ids` as the index→ID lookup. Sends to VTube Studio via WebSocket (`pyvts` library). On `LOW_CONFIDENCE`: holds last values. On MLP timeout: falls back to `runtime/fallback.py`.

### bci_bridge.py

Receives BCI parameter values from the Muse VTuber Bridge (existing system, ZyphraExps). Smooth-sends to VTube Studio WebSocket. On disconnect: holds last values 2s, then interpolates to 0.0 over 1s. Reconnect loop runs in background.

---

## Implementation Order

Dependencies with Hiyori as dev rig:

```
Subsystem 1 (ComfyUI client)   → no deps
Subsystem 4 (BCI injection)    → depends on Hiyori (available) + VTS running
Subsystem 3 (MLP)              → depends on Hiyori (available) + live2d-py
Subsystem 5 (Runtime bridge)   → depends on 3 (MLP infer.py)
Subsystem 2 (Portrait pipeline) → depends on 1 (ComfyUI) + textoon submodule
                                   (UV transfer stage deferred until HaiMeng EULA)
```

Each subsystem gets its own implementation plan. Implementation order with Hiyori:

1. **Subsystem 1** — ComfyUI client (plan written: `plans/2026-04-04-subsystem1-comfyui-client.md`)
2. **Subsystem 4** — BCI injection via VTS custom params API (fast, unblocked, validates end-to-end VTS connection)
3. **Subsystem 3** — MLP training on Hiyori (unblocked; retrain on HaiMeng later)
4. **Subsystem 5** — Runtime bridge (needs MLP + VTS integration)
5. **Subsystem 2** — Portrait pipeline (ComfyUI-dependent; UV transfer deferred until HaiMeng)

---

## Open Questions / Deferred

- **HaiMeng EULA access**: EULA form submitted via university account. Until approved, all development uses Hiyori. MLP trained on Hiyori will need retraining once HaiMeng is available — the infrastructure is identical.
- **User will create their own rig**: The long-term plan is a custom rig. The `RigConfig` abstraction is deliberately designed so that any rig with a `.moc3`, `.model3.json`, and a parameter list can be plugged in. The UV transfer stage (Subsystem 2) is the only piece that requires per-rig UV coordinate mapping work; all other subsystems are rig-agnostic once `RigConfig` is populated.
- **Hiyori UV coordinate table**: If full end-to-end texture generation on Hiyori is wanted before HaiMeng access, someone needs to produce a `hiyori_configuration.json` with UV crop coordinates for its 2 texture sheets. Deferred — not needed for MLP/BCI/runtime development.
- **Flux ControlNet for structure guidance**: ControlNet for Flux is immature — deferred. Use img2img (denoise 0.35–0.45) for structure preservation instead.
- **Style detection**: Automatic style detection from input portrait is deferred. Default = "match input style via Kontext reference"; explicit `--style anime` override available.
- **Male rig (shenxinhui)**: Deferred until female rig pipeline is working end-to-end.
