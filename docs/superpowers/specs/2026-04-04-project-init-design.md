# Project Init Design

**Date:** 2026-04-04  
**Status:** Draft  
**Scope:** Three-track init: ComfyUI housekeeping, shadow-learn setup, programmatic client skeleton

---

## Goals

1. **Learn to drive ComfyUI programmatically and agentically** — understand the API surface, build a reusable client, explore MCP-based agentic use
2. **Generate Live2D models** — the pipeline is Textoon/CartoonAlive: portrait → SDXL texture generation via ComfyUI → HaiMeng rig assembly → `.moc3`

The init phase is the foundation for both goals.

---

## Track 1: ComfyUI Housekeeping

### 1a. Git pull

`git pull` in `/home/newub/w/ComfyUI`. Current version is v0.16.4 — want latest.

### 1b. Duplicate custom node removal

Two nodes implement the same thing:
- `comfyui-controlnetaux` — old name  
- `comfyui_controlnet_aux` — canonical PyPI name (kept)

Action: remove `comfyui-controlnetaux` directory. Keep `comfyui_controlnet_aux`.

### 1c. Model audit

Produce `docs/comfyui-model-audit.md` cataloging each model file with a pipeline-relevance tag:

| Tag | Meaning |
|---|---|
| `keep` | Needed for Live2D pipeline: ControlNet, SDXL checkpoint, SAM2, insightface, facexlib, clip, vae |
| `maybe` | Potentially useful for future experiments |
| `unused` | Unrelated to this pipeline (video gen, style LoRAs, etc.) |

**User decides on deletions** — no automatic removal of model files (large, irreversible).

---

## Track 2: Shadow-Learn Init

Run `shadow-learn.sh init -y` from `/home/newub/w/portrait-to-live2d`.

This creates:
- `docs/playbooks/` directory in the project
- Copies `session-knowledge-extract` and `memory-consolidate` skills to `~/.claude/skills/`
- Adds shadow learning bootstrap to project `CLAUDE.md`
- Creates `AGENTS.md`

Then run `shadow-learn.sh install-hooks` to wire the session-end auto-extract hook.

---

## Track 3: ComfyUI Programmatic Client Skeleton

### Library landscape (post-research)

Research (`docs/research/2026-04-04-comfyui-programmatic-driving-libs.md`) identified the following:

| Project | Stars | Status | Role |
|---|---|---|---|
| `Chaoses-Ib/ComfyScript` | 661 | Alive (Feb 2026) | Best for agentic/LLM workflow generation |
| `joenorton/comfyui-mcp-server` | 274 | Alive (Feb 2026) | Best Python MCP server |
| `deimos-deimos/comfy_api_simplified` | 109 | Alive (Mar 2026) | Solid thin HTTP client |
| `artokun/comfyui-mcp` | 24 | Active (Apr 2026) | Feature-rich but small community, Node.js |
| `realazthat/comfy-catapult` | 27 | **DEAD** (Jul 2024) | Drop — previously recommended, now gone |

The ComfyUI HTTP API is simple enough (~6 endpoints) that a custom thin client is ~200 lines. Worth building in Phase 1 for the learning goal; ComfyScript adopted in Phase 2 for agentic use.

### Two-phase approach

**Phase 1 (this init):** Build a thin custom `httpx` client (~200 lines). Rationale:
- Direct learning value — see every API call, understand the WebSocket completion pattern
- Zero library lock-in during exploration
- Reference `joenorton/comfyui-mcp-server`'s `comfyui_client.py` for robustness patterns (multiple fallbacks, logging strategy)

**Phase 2 (later):** Adopt `ComfyScript` for workflow generation. When generating varied texture workflows (different ControlNet configs, etc.), its Python DSL is far more maintainable than editing JSON dicts. The transpiler converts existing workflow JSONs to Python automatically.

**MCP layer:** `joenorton/comfyui-mcp-server` — Python-native, has tests, well-structured. Adopt or reference rather than build from scratch. `artokun/comfyui-mcp` is worth monitoring but the 24-star community is a risk.

For interactive Claude Code sessions: install `joenorton/comfyui-mcp-server` (MCP layer, separate from the Python pipeline client).

### File structure

```
comfyui/
  __init__.py
  client.py          ← ComfyUIClient class
  workflows/         ← workflow JSON files (API format)
  test_connection.py ← smoke test script
```

### `ComfyUIClient` API

```python
class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188")
    
    async def health(self) -> dict                          # GET /system_stats
    async def list_models(self, folder: str) -> list[str]  # GET /models/{folder}
    async def upload_image(self, path: Path) -> str         # POST /upload/image → filename
    async def submit(self, workflow: dict) -> str           # POST /prompt → prompt_id
    async def wait(self, prompt_id: str, timeout: float = 120.0) -> dict  # poll GET /history
    async def download(self, filename: str, dest: Path)    # GET /view → file
```

### `test_connection.py`

Smoke test: connect, call `/system_stats`, call `/models/checkpoints`, print result. Run with `uv run comfyui/test_connection.py`.

### Workflow format

Workflows stored as `comfyui/workflows/<name>.json` in ComfyUI API format (flat node dict with `class_type` + `inputs`). A `README.md` in the workflows directory documents how to export from GUI (Dev Mode → Save API Format).

---

## Success Criteria

- [ ] ComfyUI is on latest commit
- [ ] No duplicate custom nodes
- [ ] Model audit report exists at `docs/comfyui-model-audit.md`
- [ ] Shadow-learn init complete, skills installed, hooks wired
- [ ] `comfyui/client.py` exists with the six methods above
- [ ] `pyproject.toml` initialized at project root (`uv init`) with `httpx` dependency
- [ ] `uv run comfyui/test_connection.py` returns ComfyUI system stats when ComfyUI is running

---

## Out of Scope

- Actual Live2D pipeline implementation (separate phase)
- MCP server installation (can be done ad-hoc; use joenorton/comfyui-mcp-server when needed)
- ComfyScript integration (Phase 2, after baseline client works)
- Custom node audit/cleanup beyond the clear duplicate
