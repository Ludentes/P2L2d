# Phase 3 Results: Verb Dataset Generation Pipeline

**Date:** 2026-04-05
**Branch:** `worktree-liveportrait-phase1`
**Status:** ✅ 200-sample dev dataset generated, pipeline validated

## What Was Built

- `templates/humanoid-anime/schema.toml` — 16 canonical params (3 pose,
  4 eye, 2 gaze, 2 brow-height, 2 brow-angle, 2 mouth, 1 cheek) with
  ranges/defaults/curves.
- `mlp/data/live_portrait/template_schema.py` — schema loader +
  `TemplateSchema.apply_verb_params()` for label construction.
- `mlp/data/live_portrait/generate_verb_samples.py` — generation loop:
  verb pick → slider + pose jitter → render → MediaPipe extract →
  (1014-d features, P-dim labels).

## Output Format

NPZ with:
- `features` `(N, 1014)` — 478×2 landmarks + 52 blendshapes + 6 pose (rx/ry/rz deg, tx/ty/tz)
- `labels`   `(N, 16)`   — template params, pose filled from MediaPipe
- `verb_names`, `param_names` for reference

## Throughput

**23 samples/sec on RTX 5090** (render + MediaPipe extract). 10k samples ≈ 7 min.

## 200-sample Dev Results

- 0 MediaPipe rejections (every render found a face)
- Verb distribution: 2–11 samples per verb (uniform random pick)
- Feature sanity: landmarks ∈ [0.08, 0.92], blendshapes ∈ [0, 0.98], pose degrees reasonable

Label spread:

| Param | min | max | std | Notes |
|---|---|---|---|---|
| AngleX (yaw) | -19.0 | 18.7 | 9.6 | good |
| AngleY (pitch) | 2.7 | 28.9 | 5.4 | **biased** — source has ~16° baseline pitch |
| AngleZ (roll) | -13.0 | 10.3 | 5.2 | good |
| EyeLOpen | 0.0 | 1.0 | 0.28 | good |
| EyeLSmile | 0.0 | 0.0 | 0.0 | **never activated** — no verb uses it |
| MouthOpenY | 0.0 | 0.9 | 0.26 | good |
| MouthForm | -0.6 | 1.0 | 0.32 | good |

## Known Limitations (to address before 10k generation)

1. **Pitch bias**: single-reference dataset inherits the reference's
   baseline head pose. Fix = either (a) render from multiple references
   with diverse baseline poses, or (b) subtract source baseline pose
   from labels and train on relative deltas.
2. **EyeLSmile/EyeRSmile never set**: no verb currently targets these.
   Either extend verbs (e.g., squint-smile combinations) or drop them
   from schema.
3. **Uniform verb sampling** gives unbalanced counts (2..11 for N=200).
   Stratified sampling would give even coverage but uniform is fine at
   scale (at N=10k each verb gets ~310 samples).

## Next

- Resolve pitch bias (decision needed: multi-reference vs relative labels)
- Either extend verbs for EyeLSmile or prune from schema
- Generate the full 10k dataset
- Phase 4: MLP retraining on 1014-d input

---

## Update (same day): Multi-reference + baseline-bias correction

**Resolved both known limitations.**

1. **4 new squint-smile verbs** (`squint_smile`, `happy_squint`,
   `soft_smile_eyes`, `laugh`) now activate EyeLSmile/EyeRSmile.
   At N=500, 101 samples (20%) have nonzero eye-smile values.

2. **Multi-reference support** with per-source pose baseline correction:
   - CLI accepts a directory of reference images
   - Neutral-render pose is measured per reference; jitter is biased by
     `-baseline` so that MediaPipe-measured labels center around 0°
   - Sources that fail neutral detection are dropped automatically

3. **N=500 run with 6 references** (after dropping 2 that failed):
   - AngleY pitch: mean **2.3°** (was 14.6° single-ref), std 8.3, range [-23, 22]
   - All 16 template params now have nonzero variance
   - All 36 verbs represented (8–20 samples each, uniform random)
   - Throughput: 23.2 samples/s, rejection rate 1.6%

Pipeline is ready to scale to 10k.

