# Runbook: MLP Model Training & Export

## Quick Regeneration

The trained model checkpoint (`model.pt`) is gitignored. To regenerate from scratch:

```bash
# 1. Generate 100 diverse face photos (requires ComfyUI running on :8188)
uv run python -m mlp.data.live_portrait.generate_faces --n 100 --out assets/generated-faces

# 2. Generate 20k training samples (blendshapes+pose, 58-d features, ~14 min on RTX 5090)
uv run python -m mlp.data.live_portrait.generate_verb_samples \
    --reference assets/generated-faces \
    --n 20000 \
    --out mlp/data/live_portrait/datasets/train_20k_bs58.npz \
    --bs-only

# 3. Train MLP (~2 min on GPU)
uv run python -m mlp.train_verb_mlp \
    --data mlp/data/live_portrait/datasets/train_20k_bs58.npz \
    --out mlp/checkpoints/humanoid-anime-bs58 \
    --epochs 200
```

## Model Checkpoint Format

Output: `mlp/checkpoints/humanoid-anime-bs58/model.pt`

```python
checkpoint = torch.load("model.pt", weights_only=False)
# Keys:
#   state_dict   — CartoonAliveMLP weights (includes baked norm stats)
#   input_dim    — 58 (blendshapes + pose)
#   n_params     — 13
#   param_names  — ["AngleX", "AngleY", "AngleZ", "EyeLOpen", "EyeROpen",
#                   "EyeBallX", "EyeLSmile", "EyeRSmile", "BrowLY", "BrowRY",
#                   "MouthOpenY", "MouthForm", "Cheek"]
#   epoch        — best epoch number
#   val_mse      — best validation MSE
```

## Loading in Another Project (e.g. Muse VTuber Bridge)

```python
import torch
from portrait_to_live2d.mlp.model import CartoonAliveMLP
# Or copy mlp/model.py — it's self-contained (only needs torch + numpy)

ckpt = torch.load("path/to/model.pt", weights_only=False)
model = CartoonAliveMLP(n_params=ckpt["n_params"], input_dim=ckpt["input_dim"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Input:  58-d float32 tensor = 52 MediaPipe blendshapes + 6 pose values
# Output: 13-d float32 tensor = Live2D parameter values
with torch.no_grad():
    params = model(features_tensor)  # (B, 58) → (B, 13)
```

## Input Feature Layout (58-d)

| Index | Count | Source |
|---|---|---|
| 0–51 | 52 | MediaPipe ARKit blendshapes (in order from `face_landmarker`) |
| 52–57 | 6 | Pose: rx_deg, ry_deg, rz_deg, tx, ty, tz (from facial transform matrix) |

## Output Parameter Layout (13-d)

| Index | Name | Range | Notes |
|---|---|---|---|
| 0 | AngleX | [-30, 30] | Horizontal head rotation (degrees) |
| 1 | AngleY | [-20, 20] | Vertical head rotation (degrees) |
| 2 | AngleZ | [-20, 20] | Head tilt (degrees) |
| 3 | EyeLOpen | [0, 1] | Left eye openness |
| 4 | EyeROpen | [0, 1] | Right eye openness |
| 5 | EyeBallX | [-1, 1] | Horizontal gaze (-1 left, +1 right) |
| 6 | EyeLSmile | [0, 1] | Left eye smile/squint |
| 7 | EyeRSmile | [0, 1] | Right eye smile/squint |
| 8 | BrowLY | [-1, 1] | Left brow height |
| 9 | BrowRY | [-1, 1] | Right brow height |
| 10 | MouthOpenY | [0, 1] | Mouth openness |
| 11 | MouthForm | [-1, 1] | Mouth shape (-1 frown, +1 smile) |
| 12 | Cheek | [0, 1] | Cheek puff |

## Params NOT Predicted by MLP

These are omitted because LivePortrait+MediaPipe can't produce reliable signal:

| Param | Why Dropped | Alternative Source |
|---|---|---|
| EyeBallY | pupil_y too subtle (R²=0.13) | Gaze detector or rule: `≈ -0.03 * AngleY` |
| BrowLAngle | No brow tilt control (R²=0.04) | BCI / manual slider |
| BrowRAngle | Same | BCI / manual slider |

## Performance (Best Run: 10k samples, 100 faces, 54 verbs)

| Param | R² |
|---|---|
| AngleX/Y/Z | 0.989–0.995 |
| MouthOpenY | 0.847 |
| Cheek | 0.847 |
| MouthForm | 0.739 |
| EyeLSmile/EyeRSmile | 0.716 |
| EyeROpen | 0.682 |
| EyeLOpen | 0.654 |
| EyeBallX | 0.471 |
| BrowLY/BrowRY | 0.344–0.348 |
