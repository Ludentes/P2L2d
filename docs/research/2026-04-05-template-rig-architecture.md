# Template Rig Architecture

**Date:** 2026-04-05
**Status:** Design, pre-implementation

---

## Problem

VTuber rigs are radically diverse — humanoid anime characters, kemonomimi (human + cat ears), anthro-furries (muzzled faces), robots, slimes, and abstract designs. Each rig has:

- A custom parameter set (names, counts, ranges chosen by the rigger)
- A unique mapping from facial expressions to rig deformers
- Different anatomical assumptions (a dragon's "mouth" opens differently from a human's)

A per-user calibration session could solve this, but it puts friction on every user. Since most VTubing rigs fall into a small number of **archetypes**, we can amortize calibration over the archetype: calibrate once per template, users map their custom rig to the nearest template via a manifest.

---

## Solution: Template Archetypes + Manifests + Verb Libraries

Three-layer architecture:

1. **Template** — a canonical parameter schema + trained MLP + response curves + verb library, authored once per archetype
2. **Manifest** — a per-rig config mapping user's custom parameter names to the template's standard names
3. **Verb library** — a semantic action vocabulary ("smile", "close eyes", "look left") mapping to template parameters, used for generating training data from reference images

### Flow

```
User's rig (custom param names like "ParamMyCatEars")
    + rig.manifest.toml (maps to template "kemonomimi" + param name mapping)
          ↓
Template: kemonomimi
    ├── standard_param_schema.toml   — canonical names, ranges, signal sources
    ├── verb_library.toml            — verb → param values
    ├── model.pt                     — MLP trained on verb-generated data
    └── curves.toml                  — response curve defaults
          ↓
Runtime:
    MediaPipe(real face) → 1014-d → template MLP → standard params
                        → manifest maps → user's custom param names → VTS
```

---

## Template Structure

Each template ships as a directory:

```
templates/humanoid-anime/
├── README.md                 — archetype characteristics, coverage
├── schema.toml               — standard parameter schema
├── verbs.toml                — verb library for training data generation
├── reference_workflow.json   — ComfyUI workflow (Qwen Edit) for verb rendering
├── model.pt                  — MLP checkpoint (1014-d → N params)
├── curves.toml               — default response curves
└── validation/               — test cases, expected outputs
```

### Schema format (per template)

```toml
# templates/humanoid-anime/schema.toml
template_name = "humanoid-anime"
version = "1.0"

[param.HeadYaw]
range = [-30.0, 30.0]
default = 0.0
source = "pose_matrix.yaw"
description = "Head rotation around vertical axis"

[param.EyeLOpenLeft]
range = [0.0, 1.0]
default = 1.0
source = "blendshape.eyeBlinkLeft"
transform = "invert"   # eyeBlinkLeft=1 → EyeLOpen=0
description = "Left eye openness"

[param.MouthFormSmile]
range = [-2.0, 1.0]
default = 0.0
source = "blendshape.mouthSmileLeft,mouthSmileRight"
transform = "average_scale(from=[0,1], to=[-2,1])"

[param.BodyAngleX]
range = [-10.0, 10.0]
default = 0.0
source = "derived(HeadYaw * 0.3)"
description = "Body follows head yaw at reduced amplitude"
```

The `source` field documents where the value comes from — this is a hint for MLP training targets, not runtime behavior.

### Manifest format (per user rig)

```toml
# user_rigs/my_cat_girl/manifest.toml
template = "kemonomimi"
template_version = "1.0"

# Map user's custom param names → template's standard param names
[param_map]
"ParamAngleX"     = "HeadYaw"
"ParamAngleY"     = "HeadPitch"
"ParamEyeLOpen"   = "EyeLOpenLeft"
"ParamEyeROpen"   = "EyeLOpenRight"
"ParamMouthForm"  = "MouthFormSmile"
"ParamMouthOpenY" = "JawOpen"
"ParamMyCatEarsL" = "EarLPosition"      # kemonomimi-specific
"ParamMyCatEarsR" = "EarRPosition"
"ParamMyTailWag"  = "TailWag"

# Params in the template that user's rig doesn't have → ignored
# Params in user's rig not mapped → pass through unchanged (not MLP-driven)
```

Simple name mapping. No retraining, no calibration.

### Verb library format

Verbs are semantic expression actions. Each verb defines target values for template parameters:

```toml
# templates/humanoid-anime/verbs.toml

[verbs.neutral]
description = "Default expression, relaxed"
# No param overrides = default values

[verbs.smile_slight]
description = "Gentle closed-mouth smile"
MouthFormSmile = 0.3
EyeLSmile = 0.2
EyeRSmile = 0.2
Cheek = 0.2

[verbs.smile_wide]
description = "Open-mouth grin"
MouthFormSmile = 1.0
MouthOpenY = 0.3
EyeLSmile = 0.5
EyeRSmile = 0.5
Cheek = 0.5

[verbs.close_eyes_both]
description = "Both eyes fully closed"
EyeLOpenLeft = 0.0
EyeLOpenRight = 0.0

[verbs.wink_left]
description = "Left eye closed, right open"
EyeLOpenLeft = 0.0
EyeLOpenRight = 1.0

[verbs.surprised]
description = "Eyes wide, mouth open, brows up"
EyeLOpenLeft = 1.0
EyeLOpenRight = 1.0
MouthOpenY = 0.7
BrowLY = 1.0
BrowRY = 1.0

[verbs.look_left]
description = "Gaze shifted left"
EyeBallX = -0.8

[verbs.angry]
description = "Furrowed brows, frown"
BrowLY = -1.0
BrowRY = -1.0
BrowLAngle = 1.0
BrowRAngle = 1.0
MouthFormSmile = -1.5

# Composed verbs (combinations)
[verbs."smile_slight + look_left"]
MouthFormSmile = 0.3
EyeLSmile = 0.2
EyeRSmile = 0.2
EyeBallX = -0.8
```

A template needs roughly **30–60 base verbs + combinations** to cover the parameter space adequately.

---

## Verb-Based Training Data Pipeline

Replaces synthetic rig rendering for expression params. Keeps rendering for pose params.

```
1. User provides reference image (single portrait of character in neutral pose)

2. For each verb in template.verbs.toml:
   a. Build prompt: "<reference character> <verb.description>"
   b. Run ComfyUI workflow (Qwen Edit or similar) → N variations
   c. For each generated image:
      - Run MediaPipe → (landmarks, blendshapes, pose) = 1014-d vector
      - Label with verb's template parameter values
      - Save as training sample
   d. Quality filter: discard samples where MediaPipe signal diverges significantly from expected
      (e.g., verb says "eyes closed" but eyeBlinkLeft < 0.3 → discard)

3. For head pose params: keep synthetic rig rendering (which we know works)
   - Vary AngleX, AngleY, etc. on the reference rig
   - Render frames, extract MediaPipe signal, label with known param values

4. Merge: full training set = verb-generated expression samples + rig-rendered pose samples

5. Train MLP on merged set: 1014-d input → template standard params
```

### Advantages over current synthetic-only pipeline

| Aspect | Current (rig render) | Verb-based (Qwen Edit) |
|---|---|---|
| Expression param labels | Set at render time (correct) but MediaPipe on anime ≈ noise | Set from verb dictionary, MediaPipe on Qwen-edited images tends to work |
| Character consistency | One rig only | Any reference image |
| Generalization | Rig-specific | Template-specific (1 template → many characters) |
| Domain gap | Large (anime render ≠ real face) | Smaller (Qwen outputs tend to have readable face geometry) |
| Compute cost | Fast (Live2D render) | Slow (image generation per sample) |

### Open questions to validate

1. **Character consistency**: Does Qwen Edit preserve character identity across verb applications? Likely depends on prompt engineering and strength parameter.
2. **Verb intensity control**: Can we generate "slight smile" vs "wide smile" with controllable intensity?
3. **MediaPipe reliability on Qwen output**: Does eyeBlinkLeft actually change with "close eyes" verb vs "neutral"?
4. **Coverage**: How many verbs are needed to densely sample the parameter space?
5. **Training data volume**: How many samples per verb × verb count × augmentation = total dataset size?

---

## Initial Template Catalog

Five templates cover an estimated 95%+ of VTuber rigs:

### 1. `humanoid-anime` (PRIMARY — ~60% of rigs)

Characteristics:
- Anime/manga-styled human face
- Large expressive eyes with iris tracking
- Standard 52 ARKit-compatible expressions
- 70–150 total params typical

Reference rig: Hiyori (74 params, already on disk)

Parameter groups:
- Head pose (AngleX/Y/Z, PositionX/Y/Z, BodyAngle*)
- Eye (EyeLOpenLeft/Right, EyeBallX/Y, EyeLSmile/RSmile)
- Mouth (JawOpen, MouthFormSmile, MouthX)
- Brow (BrowLY/RY, BrowLAngle/RAngle, BrowLX/RX)
- Cheek
- Breath, HairFront/Side (physics-driven, not face-tracked)
- Arm/Hand poses (manual control or idle animation)

### 2. `humanoid-realistic` (~15% of rigs)

Characteristics:
- More realistic human proportions
- Smaller eyes, more subtle expressions
- Tighter response curves (less exaggeration)
- Shares schema with humanoid-anime but different curves + training data

Reference rig: TBD (need candidate, possibly Cubism sample model)

### 3. `kemonomimi` (~15% of rigs)

Characteristics:
- Humanoid face with animal extras (cat/fox/dog ears, tail)
- Extends humanoid-anime schema with ear and tail params
- Ears often driven by emotional state (raised = alert, back = uncertain)

Reference rig: TBD

Additional params beyond humanoid-anime:
- EarLPosition, EarRPosition
- EarLTwitch, EarRTwitch
- TailWag, TailPosition

### 4. `anthro-furry` (~8% of rigs)

Characteristics:
- Non-human face geometry (muzzle, snout)
- Mouth opens differently from humanoid
- Ears in different positions
- Standard MediaPipe landmarks may be unreliable — may need custom facial landmarks

Reference rig: TBD (challenging — need a well-rigged anthro Live2D model)

Challenge: MediaPipe is trained on human faces. Landmark detection on anthro characters may be unreliable. This template may require a custom landmark extractor or a larger dependency on blendshape inference from head pose + verb library.

### 5. `non-humanoid` (~2% of rigs, FALLBACK)

Characteristics:
- Robots, slimes, abstract creatures
- No face anatomy assumption
- Parameters are rig-specific (no standard set)
- Requires user calibration session OR manual verb library authoring

This template is a **meta-template**: it provides the calibration tooling and verb-library scaffolding so users can author their own template variant.

---

## Template Authoring Process (for new templates)

Adding a new template:

1. **Define schema** — what are the canonical parameters for this archetype?
2. **Select reference rig** — a well-rigged Live2D model representative of the archetype
3. **Define verb library** — 30–60 base verbs covering the parameter space
4. **Generate training data**:
   - Rig-rendered data for head pose params
   - Verb-based Qwen Edit data for expression params
5. **Train MLP** on merged dataset
6. **Tune response curves** — visual QA on test expressions
7. **Validate** — test with real webcam on the reference rig
8. **Document** — write README, known limitations, coverage estimate

Target: a new template should take ~1 week of authoring for a well-understood archetype (humanoid-realistic), longer for challenging ones (anthro-furry).

---

## Runtime Pipeline (with templates)

```
Webcam frame
    ↓
MediaPipe FaceLandmarker
    → landmarks (956) + blendshapes (52) + transform_matrix → extract pose (6)
    → concat = 1014-d vector
    ↓
Load user's rig.manifest.toml → identifies template
    ↓
Load template_{name}/model.pt
    ↓
MLP(1014-d) → standard params (N_template params in template schema)
    ↓
Remap via manifest param_map → user's custom param names (N_user params)
    ↓
Apply template_{name}/curves.toml response curves
    ↓
Send to VTS via WebSocket API
```

All components are swappable per rig:
- `template` changes for different archetypes
- `param_map` in manifest adapts to any custom rig
- `curves.toml` can be overridden per user for personal feel

---

## Changes to Existing Architecture

| Component | Change |
|---|---|
| `rig/config.py` | Add `template: str` field; templates become first-class |
| `mlp/model.py` | Input size: 956 → 1014 (landmarks + blendshapes + pose) |
| `mlp/data/generate_samples.py` | Split into pose renderer + verb renderer (new) |
| `mlp/data/render_verbs.py` | NEW: runs ComfyUI Qwen Edit workflow per verb |
| `mlp/infer.py` | Accept 1014-d input; load template checkpoint |
| `mlp/curves.py` | Already supports TOML loading — no change |
| NEW: `templates/` directory | Ship pre-trained templates with the project |
| NEW: `rig/manifest.py` | Load + apply user manifests |
| NEW: `mlp/templates.py` | Template discovery, loading, validation |

---

## Migration Path

**Phase 1** — Formalize current Hiyori work as the first template:
- Move existing `RigConfig(hiyori)` into `templates/humanoid-anime/`
- Define schema.toml from current hiyori params
- Keep current MLP (landmarks-only) as baseline

**Phase 2** — Extend input to 1014-d:
- Update data generator to save blendshapes + pose alongside landmarks
- Retrain humanoid-anime MLP on 1014-d input (pose should improve)

**Phase 3** — Verb-based training data:
- Author 30–60 verbs for humanoid-anime
- Build Qwen Edit ComfyUI workflow
- Generate verb-based training data from Hiyori reference image
- Retrain MLP on combined (pose-render + verb-edit) dataset

**Phase 4** — Manifest system:
- Write manifest loader + param remapper
- Validate with a second rig (non-Hiyori humanoid)

**Phase 5** — Additional templates:
- Author humanoid-realistic, kemonomimi

---

## Known Risks

- **Qwen Edit character drift**: edited images may not preserve character identity, producing training samples of different-looking characters per verb → label noise. Mitigation: low-strength edits, ControlNet with reference image, quality filter.
- **MediaPipe reliability on stylized outputs**: Qwen may still produce stylized faces that confuse MediaPipe. Mitigation: prefer templates that produce more realistic renderings, reject samples with low MediaPipe confidence.
- **Verb coverage gaps**: verb library may miss combinations, producing training samples that don't generalize. Mitigation: include combination verbs, use augmentation.
- **Template schema drift**: changing schema breaks user manifests. Mitigation: semantic versioning, migration tooling.
- **Non-humanoid faces**: MediaPipe fundamentally can't process dragon faces, abstract shapes. Mitigation: the `non-humanoid` template is honest about this and provides a calibration tool fallback.

---

## Success Metrics

A template is considered "working" when:
- Trained MLP achieves R² ≥ 0.5 on held-out verb-rendered samples for all face-tracked params
- Real webcam demo produces visually plausible animation on the reference rig
- A second rig (different character, same archetype) animates acceptably with only a manifest (no retraining)
