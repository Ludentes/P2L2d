# Anime Eye/Mouth Tracking Failure — Root Cause

**Date:** 2026-04-05

---

## Summary

Geometric heuristics (Eye Aspect Ratio, mouth gap) computed from MediaPipe face landmarks
do **not** correlate with Live2D eye/mouth parameters on anime renders. EAR is ~0.41 ± 0.056
regardless of whether `ParamEyeLOpen = 0.0` (fully closed) or `1.0` (fully open). The MLP
also achieves R² ≈ 0 for these params.

**This is a fundamental limitation of using MediaPipe landmarks on anime characters.**

---

## Evidence

On 20k sample hiyori_v2 dataset:

| Approach | ParamEyeLOpen R² | ParamMouthOpenY R² | ParamAngleX R² |
|---|---|---|---|
| Geometric EAR heuristic | −0.031 | −0.003 | n/a |
| CartoonAlive MLP | −0.034 | +0.106 | +0.690 |

EAR distribution bucketed by EyeLOpen ground truth:
- EyeLOpen 0.00–0.18: EAR = 0.4156 ± 0.058
- EyeLOpen 0.91–1.00: EAR = 0.4119 ± 0.055

The EAR mean is statistically identical across the full range.

Also noted: `raw_mouth_open_y` std = 20594 (extreme outliers), likely from frames where
`_face_h` computes near-zero (face partially off screen or detection failure).

---

## Root Cause

Anime eye openness is expressed through **eyelid texture and mesh deformation** that does
not change the XY positions of the face contour landmarks MediaPipe tracks. Specifically:

1. MediaPipe's 478-point face mesh was designed for real human faces.
2. Anime eyes have large fixed oval pupils/irises — the "open" state shows a full oval, the
   "closed" state shows a horizontal line or crescent, but these are rendered via texture
   blending / deformer weights in the rig, not visible as geometric contour changes.
3. The MediaPipe eye contour landmarks (159, 160, 161, 145, 144, 153) track where
   MediaPipe *thinks* the eye is — on anime renders it likely snaps to the iris outline
   regardless of the open/close state, giving constant EAR.

The head pose params (AngleX/Y/Z) work well (R² ~0.69) because those involve whole-face
geometric shifts that DO show up in landmark positions.

---

## Implication for CartoonAlive

The original CartoonAlive paper trains on pairs of (real webcam frame, blendshape values).
The mapping from real face to blendshapes is provided by ARKit/MediaPipe's own blendshape
estimation — it is NOT computed from raw landmark geometry. When then applied to anime:
the model learned "when ARKit says eye is 30% open → set EyeLOpen=0.3" on real faces.
At runtime, ARKit blendshapes (not raw landmarks) are the input signal.

Our implementation used raw landmark coordinates (956-d vector) instead of pre-computed
blendshapes, which is why eye/mouth params don't train.

---

## Solutions

### Option 1: Use MediaPipe blendshapes as input (recommended)

MediaPipe's `FaceLandmarker` task also outputs 52 ARKit blendshapes (`eyeBlinkLeft`,
`jawOpen`, `mouthSmileLeft`, etc.). These are pre-computed by MediaPipe's own ML model
and directly encode eye/mouth state. Map them linearly to Live2D params:
- `eyeBlinkLeft` → `ParamEyeLOpen` (inverted: blink=1 → open=0)
- `jawOpen` → `ParamMouthOpenY`
- `mouthSmileLeft/Right` → `ParamMouthForm`
- `browInnerUp` / `browDownLeft` → `ParamBrowLY`

This avoids retraining the MLP and gives correct eye/mouth tracking immediately.

### Option 2: Retrain MLP with blendshapes as input (126-d vector: 52 blendshapes + 74 pose)

Full CartoonAlive-style approach. Requires:
- Re-generating training data with blendshape labels
- New MLP architecture (smaller input: 52 instead of 956)

### Option 3: Accept limitation, use response curves for head pose only

Current state. Eye/mouth tracking works poorly but head pose (the most visually important
axis for VTubing) works well. Use `mlp/curves/hiyori.toml` response curves on top of MLP
output. Implement blendshape mapping separately as a post-processing step.

---

## Next Steps

1. Add MediaPipe blendshape extraction to the inference pipeline
2. Create `mlp/blendshape_map.py` — direct linear mapping from 52 blendshapes to
   Live2D eye/mouth/brow params (no training required)
3. Combine: MLP handles pose (AngleX/Y/Z, PositionX/Y/Z, BodyAngle),
   blendshape map handles expression (eyes, mouth, brows)
4. Validate on real webcam (Experiment 3)
