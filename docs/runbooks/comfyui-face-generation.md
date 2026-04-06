# Runbook: ComfyUI Face Generation for Training Data

## Prerequisites

- ComfyUI at `/home/newub/w/ComfyUI`
- Conda env: `comfyui`
- Checkpoint: `SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors`

## 1. Start ComfyUI Server

```bash
conda activate comfyui
cd /home/newub/w/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --preview-method auto
```

Wait for "To see the GUI go to: http://..." message.

## 2. Verify Server

```bash
curl -s http://127.0.0.1:8188/system_stats | python -m json.tool | head -10
```

## 3. Generate Faces

```bash
cd /home/newub/w/portrait-to-live2d
uv run python -m mlp.data.live_portrait.generate_faces \
    --n 100 \
    --out assets/generated-faces \
    --checkpoint "SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors"
```

This generates 100 diverse realistic face photos (neutral expression,
varied identity/lighting/angle) and saves them as PNGs. Each face is
then usable as a LivePortrait source image.

## 4. Generate Training Dataset

```bash
uv run python -m mlp.data.live_portrait.generate_verb_samples \
    --reference assets/generated-faces \
    --n 10000 \
    --out mlp/data/live_portrait/datasets/train_10k_comfyui.npz
```

## Notes

- JuggernautXL Lightning: ~1-2s/image on RTX 5090 (4-step sampler)
- 100 faces × 100 verbs/face = 10k samples with strong identity diversity
- MediaPipe quality filter built into the generator drops undetectable faces
