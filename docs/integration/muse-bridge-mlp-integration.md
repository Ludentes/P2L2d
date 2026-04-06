# MLP Integration Guide for Muse VTuber Bridge

This document describes how to load and use the CartoonAlive MLP model
trained by the `portrait-to-live2d` project.

## What the Model Does

Converts a 58-dimensional feature vector (extracted from a webcam frame
via MediaPipe) into 13 Live2D parameter values. Runs in <1ms on GPU,
<8ms on CPU.

```
Webcam frame → MediaPipe → 58-d features → MLP → 13 Live2D params
```

## Files You Need

1. **`model.pt`** — trained checkpoint (~2.5MB)
   - Location: `portrait-to-live2d/mlp/checkpoints/humanoid-anime-bs58/model.pt`
   - Regenerate: see `portrait-to-live2d/docs/runbooks/model-training-and-export.md`

2. **`mlp/model.py`** — model class definition (self-contained, depends only on torch + numpy)
   - Copy this file into your project, or add `portrait-to-live2d` as a dependency

## Loading the Model

```python
import torch

# model.py is self-contained — copy it or import from portrait-to-live2d
from mlp.model import CartoonAliveMLP

def load_mlp(checkpoint_path: str, device: str = "cpu") -> CartoonAliveMLP:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = CartoonAliveMLP(
        n_params=ckpt["n_params"],   # 13
        input_dim=ckpt["input_dim"], # 58
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model
```

## Checkpoint Contents

```python
{
    "state_dict":   ...,           # model weights (includes baked normalization stats)
    "input_dim":    58,            # blendshapes(52) + pose(6)
    "n_params":     13,            # output Live2D params
    "param_names":  ["AngleX", "AngleY", "AngleZ", "EyeLOpen", "EyeROpen",
                     "EyeBallX", "EyeLSmile", "EyeRSmile", "BrowLY", "BrowRY",
                     "MouthOpenY", "MouthForm", "Cheek"],
    "epoch":        ...,           # training epoch of best model
    "val_mse":      ...,           # validation MSE at best epoch
}
```

## Building the 58-d Input Vector

The MLP expects a specific feature layout. Build it from MediaPipe's
`FaceLandmarker` output:

```python
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- One-time setup ---
def create_landmarker(model_path: str) -> vision.FaceLandmarker:
    """Download model from:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    """
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,                # REQUIRED
        output_facial_transformation_matrixes=True,   # REQUIRED
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)

# --- Per-frame extraction ---
def extract_features(landmarker, frame_rgb: np.ndarray) -> np.ndarray | None:
    """Extract 58-d feature vector from an RGB frame.
    Returns None if no face detected.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks or not result.face_blendshapes:
        return None

    # 52 ARKit blendshapes (order matters — use MediaPipe's order as-is)
    bs = np.array([b.score for b in result.face_blendshapes[0]], dtype=np.float32)
    # Pad/truncate to exactly 52 if needed
    if len(bs) != 52:
        tmp = np.zeros(52, dtype=np.float32)
        tmp[:min(len(bs), 52)] = bs[:52]
        bs = tmp

    # 6-d pose from facial transformation matrix
    mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)
    R = mat[:3, :3]
    t = mat[:3, 3]
    sy = float(np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    pose = np.array([np.rad2deg(rx), np.rad2deg(ry), np.rad2deg(rz),
                      t[0], t[1], t[2]], dtype=np.float32)

    return np.concatenate([bs, pose])  # shape: (58,)
```

## Running Inference

```python
import torch

model = load_mlp("model.pt", device="cuda")  # or "cpu"

# Per frame:
features = extract_features(landmarker, frame_rgb)  # (58,)
if features is not None:
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, 58)
        params = model(x).squeeze(0).cpu().numpy()               # (13,)

    # Map to param names
    param_names = ["AngleX", "AngleY", "AngleZ", "EyeLOpen", "EyeROpen",
                   "EyeBallX", "EyeLSmile", "EyeRSmile", "BrowLY", "BrowRY",
                   "MouthOpenY", "MouthForm", "Cheek"]
    param_dict = dict(zip(param_names, params))
    # → {"AngleX": -5.2, "MouthOpenY": 0.73, ...}
```

## Output Parameters

| Index | Name | Range | Live2D Mapping |
|---|---|---|---|
| 0 | AngleX | [-30, 30] | `ParamAngleX` — horizontal head rotation (deg) |
| 1 | AngleY | [-20, 20] | `ParamAngleY` — vertical head rotation (deg) |
| 2 | AngleZ | [-20, 20] | `ParamAngleZ` — head tilt (deg) |
| 3 | EyeLOpen | [0, 1] | `ParamEyeLOpen` — left eye openness |
| 4 | EyeROpen | [0, 1] | `ParamEyeROpen` — right eye openness |
| 5 | EyeBallX | [-1, 1] | `ParamEyeBallX` — horizontal gaze |
| 6 | EyeLSmile | [0, 1] | `ParamEyeLSmile` — left eye squint |
| 7 | EyeRSmile | [0, 1] | `ParamEyeRSmile` — right eye squint |
| 8 | BrowLY | [-1, 1] | `ParamBrowLY` — left brow height |
| 9 | BrowRY | [-1, 1] | `ParamBrowRY` — right brow height |
| 10 | MouthOpenY | [0, 1] | `ParamMouthOpenY` — mouth openness |
| 11 | MouthForm | [-1, 1] | `ParamMouthForm` — smile/frown |
| 12 | Cheek | [0, 1] | `ParamCheek` — cheek puff |

## Params NOT Predicted — Need Alternative Sources

The MLP does NOT output these params. The bridge should handle them separately:

| Param | Why | Suggested Source |
|---|---|---|
| `EyeBallY` | Vertical gaze undetectable via LivePortrait training | Gaze tracking model, or rule: `EyeBallY ≈ -0.03 × AngleY` |
| `BrowLAngle` | LivePortrait can't control brow tilt | BCI (`MuseFocus`), manual slider, or fixed at 0 |
| `BrowRAngle` | Same | Same |
| `ParamBodyAngleX/Y/Z` | Body pose — not face tracking | IMU, or derive from head: `BodyAngleX ≈ 0.3 × AngleX` |

## Recommended Post-Processing in the Bridge

### 1. Temporal Smoothing (essential)

Raw per-frame output will jitter. Apply exponential moving average:

```python
class ParamSmoother:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha  # 0.0 = no smoothing, 1.0 = no update
        self.prev = None

    def smooth(self, params: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = params.copy()
            return params
        self.prev = self.alpha * self.prev + (1 - self.alpha) * params
        return self.prev.copy()
```

Use different alpha per param group:
- Pose (AngleX/Y/Z): `alpha=0.3` (fast response)
- Eyes (blink, gaze): `alpha=0.3` (fast response)
- Mouth: `alpha=0.4` (moderate)
- Brows: `alpha=0.6` (slow, less jitter)

### 2. Neutral Calibration (recommended)

On startup, capture ~30 frames of the user's neutral face. Compute the
mean MLP output as the "baseline". Subtract it at runtime so the Live2D
model sits at its default pose when the user is neutral.

```python
# Calibration (once on startup):
baseline = np.mean([predict(frame) for frame in neutral_frames], axis=0)

# Runtime:
raw_params = predict(frame)
calibrated = raw_params - baseline + default_params
```

### 3. Response Curves (optional)

The `portrait-to-live2d` project includes `mlp/curves.py` with per-param
gamma/linear response curves. These exaggerate subtle expressions for
more expressive animation. Apply after MLP, before sending to VTS.

## Performance

| Metric | Value |
|---|---|
| MLP inference (GPU) | <1ms |
| MLP inference (CPU) | <8ms |
| MediaPipe extraction | ~5ms |
| Total per-frame budget | ~10ms (100fps capable) |
| Model size | ~2.5MB |
| Input dim | 58 |
| Output dim | 13 |

## Blendshape Index Reference

For debugging, the 52 MediaPipe blendshapes in order:

```
 0: _neutral           13: eyeLookInLeft      26: jawRight
 1: browDownLeft        14: eyeLookInRight     27: mouthClose
 2: browDownRight       15: eyeLookOutLeft     28: mouthDimpleLeft
 3: browInnerUp         16: eyeLookOutRight    29: mouthDimpleRight
 4: browOuterUpLeft     17: eyeLookUpLeft      30: mouthFrownLeft
 5: browOuterUpRight    18: eyeLookUpRight     31: mouthFrownRight
 6: cheekPuff           19: eyeSquintLeft      32: mouthFunnel
 7: cheekSquintLeft     20: eyeSquintRight     33: mouthLeft
 8: cheekSquintRight    21: eyeWideLeft        34: mouthLowerDownLeft
 9: eyeBlinkLeft        22: eyeWideRight       35: mouthLowerDownRight
10: eyeBlinkRight       23: jawForward         36: mouthPressLeft
11: eyeLookDownLeft     24: jawLeft            37: mouthPressRight
12: eyeLookDownRight    25: jawOpen            38: mouthPucker
                                                39: mouthRight
40: mouthRollLower      44: mouthSmileLeft     48: mouthUpperUpLeft
41: mouthRollUpper      45: mouthSmileRight    49: mouthUpperUpRight
42: mouthShrugLower     46: mouthStretchLeft   50: noseSneerLeft
43: mouthShrugUpper     47: mouthStretchRight  51: noseSneerRight
```
