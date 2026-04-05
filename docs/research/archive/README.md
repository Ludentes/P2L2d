# Archived Research

Historical research docs superseded by later decisions. Kept for reference — do not rely on their recommendations without cross-checking against current `../` docs.

## Archived 2026-04-05 (LivePortrait pivot)

These documents describe the ComfyUI/SDXL/FLUX-centric approach to training-data generation and texture synthesis. The project pivoted to a LivePortrait-based direct-Python pipeline, removing ComfyUI from the runtime and CI paths.

| File | Original topic | Why archived |
|---|---|---|
| `2026-04-03-pipeline-overview.md` | Overall SDXL+Textoon pipeline | Predates MLP pivot and LivePortrait approach |
| `2026-04-03-comfyui-claude-code-integration.md` | ComfyUI REST/MCP integration for Claude Code | Outdated recommendations (flagged in CLAUDE.md) |
| `2026-04-03-comfyui-live2d-art-pipeline.md` | Texture decomposition via ComfyUI | Texture-swap path on hold (HaiMeng EULA), not in current pipeline |
| `2026-04-04-comfyui-programmatic-driving-libs.md` | Audit of Python libs for driving ComfyUI | ComfyUI dropped from runtime/CI |
| `2026-04-04-generation-model-selection.md` | Flux Kontext vs SDXL for texture generation | Texture generation deferred; not in current pipeline |
| `2026-04-05-comfyui-workflows-for-training-data.md` | ComfyUI style-transfer + verb-edit workflows | Superseded by LivePortrait direct-Python approach; retained as fallback plan if LivePortrait proves inadequate |

## Current architecture

See:
- `../2026-04-05-full-pipeline-plan.md` — current training + runtime pipeline
- `../2026-04-05-liveportrait-deep-dive.md` — primary data-generation mechanism
- `../2026-04-05-template-rig-architecture.md` — template + manifest design
