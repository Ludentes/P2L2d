# ComfyUI Workflow Files

Workflow JSON files in this directory are in **API format** — not the default UI save format.

## How to export API format from the ComfyUI GUI

1. Open ComfyUI in your browser (`http://127.0.0.1:8188`)
2. Click the gear icon ⚙️ → **Dev Mode Options** → enable **"Enable Dev mode Options"**
3. Load or build the workflow you want to save
4. Click **"Save (API Format)"** — downloads a JSON with a flat node dict

## API format overview

Keys are node IDs (numeric strings). Links between nodes use `["source_node_id", output_index]`:

```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 20,
      "model": ["4", 0]
    }
  },
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {"ckpt_name": "flux1-dev-kontext_fp8_scaled.safetensors"}
  }
}
```

## Naming convention

`<purpose>[-<model-hint>].json` — examples:
- `flux-kontext-portrait.json`
- `flux-kontext-component-inpaint.json`
- `sdxl-anime-flat.json`
