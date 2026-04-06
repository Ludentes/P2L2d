# portrait-to-live2d — Project Instructions

## Scope

This project is **generation-only**. It produces artifacts (rig + trained MLP checkpoint) and exits. The runtime that consumes these artifacts at stream time is `muse-vtuber` (separate repo at `/home/newub/w/zyphraexps/muse-vtuber`).

See `muse-vtuber/docs/p2l-integration-requests.md` for the integration contract from the consumer side.

## Goal

Automated pipeline from a single portrait image → animatable Live2D `.moc3` model + trained MLP checkpoint. Based on the Textoon/CartoonAlive approach (template rig + generated textures + learned MediaPipe→parameter MLP).

## Output Artifacts

| Artifact | Format | Notes |
|---|---|---|
| `character.model3.json` + `.moc3` + textures | Live2D model | Must include external params (BCI) by name |
| `model.pt` | Self-describing checkpoint dict | `param_names` must match model3.json |
| `rig_manifest.json` | Param manifest sidecar | Optional, enables runtime validation |

Checkpoint format (standard for all future training runs):
```python
{
    "state_dict":  ...,         # model weights
    "input_dim":   58,          # 52 blendshapes + 6 pose
    "n_params":    13,          # number of output params
    "param_names": [...],       # list of param IDs in output order
    "epoch":       ...,
    "val_mse":     ...,
}
```

## Stack

- **Python 3.12**, uv for package management
- **ComfyUI** (programmatic via REST API + MCP server) — texture generation, face generation
- **Live2D Cubism SDK** (Python bindings via `live2d-py`) — headless model rendering
- **LivePortrait** (vendored submodule) — expression rendering for training data
- **MediaPipe** — face landmark + blendshape extraction
- **PyTorch** — CartoonAlive MLP training
- **SDXL/FLUX** — texture generation (via ComfyUI)
- **SAM2** — segmentation

## Key Concepts

- **Template system**: Pre-trained archetypes in `templates/`. Each has `schema.toml` (params), `verbs.toml` (training verbs), `model.pt` (MLP), `curves.toml` (response curves). Currently: `humanoid-anime` (13 params, 58-d input).
- **Manifest system**: Per-rig config in `manifests/` mapping custom param names → template canonical names. e.g. `ParamAngleX` → `AngleX`.
- **Verb-based training**: LivePortrait renders expressions from verbs → MediaPipe extracts 58-d features → MLP learns mapping.
- **HaiMeng dataset**: production rig template. 9 × 4096px, 107 params. EULA-gated. Dev rig: Hiyori (74 params).
- **CartoonAlive MLP**: 4-layer network with skip connection. 58-d input (52 blendshapes + 6 pose) → 13 Live2D params.

## Project Layout

```
templates/       — pre-trained template archetypes (schema + verbs + curves)
manifests/       — per-rig param mappings
mlp/             — MLP model, training, inference
rig/             — rig config, manifest loader, headless rendering
comfyui/         — ComfyUI REST API client + workflow definitions
pipeline/        — end-to-end portrait → model3.json pipeline (planned)
docs/
  research/      — research notes
  integration/   — cross-project integration docs
  runbooks/      — repeatable procedures
  research/2026-04-05-full-pipeline-plan.md — current architecture
```

## External Parameters (Category C)

Generated rigs must include these params for the Muse VTuber Bridge to drive. The MLP holds them at defaults during training; muse-vtuber injects live values at stream time.

| Parameter ID | Range | Signal source | Notes |
|---|---|---|---|
| `MuseBlink` | 0–1 | EMG blink detection | Replaces camera blink |
| `MuseClench` | 0–1 | EMG jaw clench | |
| `MuseFocus` | 0–1 | EEG theta/beta ratio | |
| `MuseRelaxation` | 0–1 | EEG alpha relaxation | |

These should be configurable via a `[external_params]` section in generation config, not hardcoded.

## Conventions

- Research docs → `docs/research/YYYY-MM-DD-<topic>.md`
- ComfyUI workflows → `comfyui/workflows/<name>.json`
- Use `uv` for Python dependency management
- Commit and push after each logical unit of work
- Conventional commits: `docs:`, `feat:`, `fix:`, `chore:`
