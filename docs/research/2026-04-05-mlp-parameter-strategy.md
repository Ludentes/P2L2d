# MLP Parameter Strategy — Hiyori Sampling & Response Curves

**Date:** 2026-04-05

---

## 1. The Core Problem: We Were Sampling Wrong

Our current `PARAM_RANGE = (-30.0, 30.0)` and `Normal(0, 8)` sampling treats all 74 params as if
they live on a ±30 scale. They don't. The actual rig-defined ranges from `live2d-py` are:

| Group | Params | Native range | What we sampled |
|---|---|---|---|
| Head angles | AngleX/Y/Z | −30 → 30 | ✅ correct |
| Head position | PositionX/Y/Z | −10 → 10 | ❌ sampled up to ±30 |
| Body angles | BodyAngleX/Y/Z | −10 → 10 | ❌ sampled up to ±30 |
| Eyes open | EyeLOpen, EyeROpen | **0 → 1** | ❌ sampled up to ±30 |
| Eye gaze | EyeBallX/Y | **−1 → 1** | ❌ sampled up to ±30 |
| Eye smile | EyeLSmile, EyeRSmile | **0 → 1** | ❌ sampled up to ±30 |
| Brows | BrowL/RY, BrowL/RX, BrowL/RAngle, BrowL/RForm | **−1 → 1** | ❌ sampled up to ±30 |
| Mouth open | MouthOpenY | **0 → 1** | ❌ sampled up to ±30 |
| Mouth shape | MouthForm | **−2 → 1** | ❌ sampled up to ±30 |
| Mouth position | MouthX | **−1 → 1** | ❌ sampled up to ±30 |
| Cheek | ParamCheek | **−1 → 1** | ❌ sampled up to ±30 |
| Body/hair secondary | Skirt, Hair, Ribbon, Bust | −1 → 1 / −10 → 10 | ❌ |
| Arms/hands | ArmLA/RA/LB/RB, HandL/R etc | −10 → 10 | ❌ |
| ArtMesh deformers | 28 × Param_Angle_Rotation_* | −45 → 45 | ❌ |

**What this means in practice:** When we sampled EyeLOpen=15.0, the Live2D renderer clamped it
to 1.0 (fully open). EyeLOpen=8.0 also rendered as 1.0. These are different training targets
for identical renders — introducing label noise that directly tanks R² for eye/mouth/brow params.

This is why Position/Angle params trained well (we happened to use the right scale) and
eye/mouth/brow params trained poorly (scale mismatch → label noise → R² ≈ 0.3).

---

## 2. Param Classification for Generation

### Category A — Face-Tracked (vary during generation, realistic ranges)

These params are what the MLP is actually learning to predict from face landmarks.
Sample within the *realistic* sub-range — slightly narrower than the rig max to
represent what a real human face can produce.

| Param | Rig range | Realistic sample range | Notes |
|---|---|---|---|
| ParamAngleX | −30 → 30 | −30 → 30 | Head yaw, real faces reach ±30° |
| ParamAngleY | −30 → 30 | −20 → 20 | Head pitch, comfortable range ±20° |
| ParamAngleZ | −30 → 30 | −20 → 20 | Head roll |
| ParamPositionX | −10 → 10 | −5 → 5 | Head lateral shift in frame |
| ParamPositionY | −10 → 10 | −5 → 5 | Head vertical shift |
| ParamPositionZ | −10 → 10 | −2 → 2 | Depth (minor for 2D) |
| ParamBodyAngleX | −10 → 10 | −5 → 5 | Body sway follows head (smaller) |
| ParamBodyAngleY | −10 → 10 | −3 → 3 | |
| ParamBodyAngleZ | −10 → 10 | −5 → 5 | |
| ParamEyeLOpen | 0 → 1 | 0 → 1 | 0=closed, 1=open (default), allow blinking |
| ParamEyeROpen | 0 → 1 | 0 → 1 | Mirror of left |
| ParamEyeBallX | −1 → 1 | −0.8 → 0.8 | Gaze left/right |
| ParamEyeBallY | −1 → 1 | −0.8 → 0.8 | Gaze up/down |
| ParamEyeLSmile | 0 → 1 | 0 → 1 | Squint when smiling |
| ParamEyeRSmile | 0 → 1 | 0 → 1 | |
| ParamBrowLY | −1 → 1 | −1 → 1 | Brow height |
| ParamBrowRY | −1 → 1 | −1 → 1 | |
| ParamBrowLX | −1 → 1 | −0.5 → 0.5 | Brow horizontal |
| ParamBrowRX | −1 → 1 | −0.5 → 0.5 | |
| ParamBrowLAngle | −1 → 1 | −1 → 1 | Brow tilt |
| ParamBrowRAngle | −1 → 1 | −1 → 1 | |
| ParamBrowLForm | −1 → 1 | −1 → 1 | Brow shape |
| ParamBrowRForm | −1 → 1 | −1 → 1 | |
| ParamMouthOpenY | 0 → 1 | 0 → 1 | 0=closed, 1=max open |
| ParamMouthForm | −2 → 1 | −2 → 1 | −2=sad, 1=smile |
| ParamMouthX | −1 → 1 | −0.5 → 0.5 | Mouth lateral |
| ParamCheek | −1 → 1 | 0 → 1 | Blush (puff cheek on smile) |

### Category B — Physics-Driven (hold at default, let physics sim handle)

These params are animated by the Live2D physics simulation based on head movement.
Setting them to anything other than default during data generation would fight the physics sim.
Fix at 0 (their default) and let physics run naturally.

- ParamHairAhoge, ParamHairFront, ParamHairBack
- ParamSkirt, ParamSkirt2
- ParamRibbon, ParamSideupRibbon
- ParamBustY

### Category C — Manually Controlled / BCI (fix at default)

Not driven by face landmarks. Hold at their default values during generation.
In production, BCI bridge injects these independently.

- ParamArmLA, ParamArmRA, ParamArmLB, ParamArmRB
- ParamHandLB, ParamHandRB (default=10!), ParamHandL, ParamHandR
- ParamShoulder, ParamStep
- ParamBreath (animated by VTS breathing oscillator, not face)

### Category D — Internal Deformer Params (always 0)

28 params of the form `Param_Angle_Rotation_N_ArtMeshXX`. These control internal mesh
deformers and should never be set externally. Always 0.

---

## 3. Why Low R² for Mouth/Eyes Now Makes Sense

With the old sampling (all params N(0,8) clipped to ±30):

- EyeLOpen=1.5 → rendered as 1.0 (eye fully open)
- EyeLOpen=8.0 → rendered as 1.0 (eye fully open)
- EyeLOpen=0.3 → rendered as 0.3 (eye 30% open)

The training set contained many samples with different `params.npy` values but identical
rendered frames → identical landmarks. The MLP saw inconsistent (landmark, target) pairs
and learned to predict the mean (~0) as a safe low-loss strategy.

With correct sampling in [0, 1], every EyeLOpen value produces a visually distinct frame
and a consistent (landmark, target) pair. The MLP can actually learn the mapping.

---

## 4. Response Curves — The Exaggeration Layer

**Concept:** A *response curve* (also called *transfer function*, *gain curve*, or *drive curve*)
maps MLP output → final param value sent to VTube Studio. It is applied post-inference,
not baked into the model.

**Why needed:** Real human faces move in smaller physical ranges than anime characters
are designed to express. A real eye blink covers maybe 8–12mm of eyelid travel.
An anime eye going 0→1 on EyeLOpen produces a much more dramatic visual shift.
Without a curve, the animation looks muted.

**Curve shapes for VTubing:**

```
Linear (no exaggeration):       S-curve (natural feel):         Gamma (lift midtones):
out |  /                        out |    ___                     out |   __/
    | /                             |   /                            |  /
    |/_____ in                      |__/_______ in                   |_/________ in
```

For most face params, a **gamma curve** (`out = in^gamma`, gamma < 1) works well:
- `gamma=0.7` on EyeLOpen: a slightly-open real eye → clearly visible anime opening
- `gamma=0.6` on MouthOpenY: normal speech mouth → expressive anime speech
- `gamma=1.0` on AngleX/Y/Z: head rotation maps linearly (already 1:1 scale)

**Implementation plan** (post-MLP, pre-VTS send):
```python
# Per-param curve applied after MLP predict()
RESPONSE_CURVES = {
    "ParamEyeLOpen":   ("gamma", 0.7),
    "ParamEyeROpen":   ("gamma", 0.7),
    "ParamMouthOpenY": ("gamma", 0.6),
    "ParamEyeLSmile":  ("gamma", 0.8),
    "ParamEyeRSmile":  ("gamma", 0.8),
    # Head pose: linear, no curve needed
}

def apply_curve(value, curve_type, param):
    if curve_type == "gamma":
        # value in [0,1], gamma < 1 lifts midtones
        return value ** param
    elif curve_type == "linear_scale":
        return value * param
    return value
```

The curves should be **tunable at runtime** without retraining — they're a personality
layer on top of the learned mapping.

---

## 5. Recommended Next Steps

1. **Fix `RigConfig`**: add `param_ranges: dict[str, tuple[float, float]]` to `RigConfig`,
   populated from rig metadata (or hardcoded from the table above for Hiyori).

2. **Fix sampling**: use per-param ranges instead of global `PARAM_RANGE = (-30, 30)`.
   Category B/C/D params get fixed at their default values (not sampled at all).

3. **Regenerate dataset**: with correct ranges, mouth/eye/brow R² should improve
   substantially. Expect eye/mouth params to reach R² > 0.7 with clean label correspondence.

4. **Implement response curves**: `mlp/curves.py` — a `ResponseCurveSet` that wraps
   `MLPInference.predict()` output and applies per-param curves before sending to VTS.
   Curves are tunable JSON/TOML config, not hardcoded.

5. **Validate on real webcam**: synthetic→real generalization check. Run inference on
   a webcam feed, compare subjective animation quality before/after curve tuning.
