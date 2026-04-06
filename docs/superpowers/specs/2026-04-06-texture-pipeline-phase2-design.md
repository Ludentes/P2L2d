# Texture Pipeline — Phase 2 Design

**Date:** 2026-04-06
**Scope:** Portrait extraction → replacement images + mask-guided blending + new swap strategies
**Depends on:** Phase 1 (`atlas_config.py`, `texture_swap.py`, `hiyori_atlas.toml`)

---

## Goal

Take a single portrait photograph and produce the `replacements: dict[str, Image.Image]`
dict that Phase 1's `swap_regions()` consumes. Implement the three new swap strategies
(`face_paste`, `landmark_paste`, `cloth_paste`) that Phase 1 stubs out.

Success criterion: portrait → modified Hiyori textures → headless render shows the
portrait's face, eye colour, and skin tone in the correct rig positions, with no
feature ghost-through during expression animation.

---

## What CartoonAlive Taught Us

1. **Texture editability is an atlas design problem.** The UV crop table (our
   `atlas_config.toml`) is the only runtime contract. No UV parsing at swap time.

2. **Face skin uses affine warp, not SAM2.** Detect portrait landmarks → compute affine
   transform to template coordinate frame → warp and paste. SAM2 is only needed for hair
   (extends outside the facial landmark convex hull).

3. **Inpainting the face base is load-bearing.** After placing feature textures, the
   face_skin UV region must have eye sockets, brow ridges, and lip area inpainted with
   surrounding skin. Without this, those features ghost through during animation when
   the overlying feature drawable moves.

4. **Feature extraction order matters.** Parameter estimation runs on the
   feature-stripped rendering, not the raw portrait. For us: estimate placement *after*
   features are removed/inpainted, not before.

5. **Clothing requires inpainting at seam edges.** Adjacent drawable bboxes share
   boundary pixels. A hard paste creates visible seams. Feathering + inpaint blends them.

6. **Masks are first-class artifacts.** Per-feature binary masks (rendered at neutral
   pose from the rig) drive both the face inpainting region and the blending alpha for
   each feature paste. Masks must be generated once per rig and stored alongside the
   atlas config.

---

## New Swap Strategies

### `face_paste`

Used for: `face_skin`

Input to `swap_region()`:
```python
replacement = portrait_image          # full portrait, PIL RGBA
kwargs = {
    "landmarks": dict[str, tuple],   # MediaPipe face mesh keypoints
    "inpaint_mask": Image.Image,     # binary mask: white = inpaint (eye/brow/mouth areas)
    "template_landmarks": dict,      # canonical template landmark positions
}
```

Operation:
1. Compute affine transform: portrait landmark positions → template landmark positions
2. Warp portrait to match template face geometry (thin-plate spline or affine)
3. Crop the face_skin UV region from the warped portrait
4. Paste into each drawable's UV bbox (using existing per-drawable atlas data)
5. Load `inpaint_mask` — generated from rig render, covers eye sockets, brow ridges,
   lip area in face_skin UV coordinates
6. Inpaint masked areas using surrounding skin pixels (Telea inpainting or diffusion)

Why inpaint_mask lives in UV space: the mask is generated once per rig by projecting
each feature drawable's screen-space footprint back to the face_skin UV mesh.
See `pipeline/mask_generator.py`.

---

### `landmark_paste`

Used for: `left_eye`, `right_eye`, `left_eyebrow`, `right_eyebrow`, `mouth`

Input:
```python
replacement = portrait_image
kwargs = {
    "landmarks": dict[str, tuple],
    "template_landmarks": dict,
    "region_name": str,              # "left_eye", "right_eye", etc.
}
```

Operation:
1. Using landmarks, locate the feature region in the portrait (eye corners, brow arch,
   lip line)
2. Extract a padded crop centered on the feature
3. Apply local affine warp to align feature to template geometry
4. For each drawable in the region: scale aligned crop to drawable's bbox, paste

Note: feature paste must happen AFTER face_paste. The feature drawables render on top
of face_skin in the rig, so paste order doesn't affect the final atlas (they're on
separate drawables), but the face inpainting step needs to have already cleaned the
face_skin texture.

---

### `cloth_paste`

Used for: `clothing`, `body`

Input:
```python
replacement = generated_clothing_image   # SDXL/Flux generated, or palette-shifted
kwargs = {
    "seam_mask": Image.Image | None,     # optional: white = seam pixels to feather
    "blend_width": int,                  # feather width in pixels (default 8)
}
```

Operation:
1. For each drawable in the region: scale replacement to drawable's bbox, paste
2. If `seam_mask` provided: at seam pixels, blend new and old with a feathered alpha
   (`blend_width` px Gaussian-feathered edge)
3. Seam detection: adjacent drawables sharing a boundary (their bboxes touch or overlap
   by < 4px) have their shared edge marked in the seam_mask

The seam feathering prevents the hard horizontal/vertical lines that appear when
multiple adjacent drawables are pasted independently with no blending at boundaries.

For `body`: same strategy as `cloth_paste`. The skin-tone replacement image (derived
from portrait) is pasted per-drawable with feathering.

---

## Mask System

### What masks are

Per-rig binary PNG masks stored in `manifests/<rig>_masks/`. Each mask covers one
region and is in **UV coordinates of a specific texture** (not screen coordinates).

| Mask file | UV texture | Covers |
|---|---|---|
| `face_skin_inpaint.png` | texture_00, same size | White = eye socket + brow ridge + lip area to inpaint on face_skin |
| `left_eye_blend.png` | texture_00 | White = eye feature footprint (for alpha blending) |
| `right_eye_blend.png` | texture_00 | Same, right eye |
| `clothing_seam.png` | texture_01 | White = seam pixels between adjacent clothing drawables |

### How face_skin inpaint mask is generated

`pipeline/mask_generator.py`:
1. Load the rig and render at neutral pose (full model, no modifications)
2. For each feature drawable group (left_eye + right_eye, brows, mouth):
   a. Render only those drawables in solid white on black → screen-space mask
   b. Use the face_skin drawable's UV mesh vertices to build a UV→screen transform
   c. Invert the transform: project screen-space mask back to face_skin UV space
   d. Accumulate into `face_skin_inpaint.png`
3. Dilate the result by 4px (ensures no hard edges remain at feature boundaries)

This is a one-time-per-rig operation. Output is committed alongside the atlas TOML.

### How clothing seam mask is generated

1. For each pair of adjacent clothing drawables (bboxes share a boundary):
   a. Mark all pixels within `blend_width` px of the shared edge
   b. Write to `clothing_seam.png`

Adjacency detection: two drawables are adjacent if their bboxes overlap or touch on
any axis by < 4px.

### Where masks are referenced

`AtlasRegion` gains two optional fields:
```python
@dataclass
class AtlasRegion:
    name: str
    swap_strategy: str
    drawables: list[DrawableRegion]
    inpaint_mask_path: str | None = None   # relative to manifests/ dir
    blend_mask_path: str | None = None     # relative to manifests/ dir
```

These are populated by `mask_generator.py` and written into the atlas TOML:
```toml
[[regions]]
name              = "face_skin"
swap_strategy     = "face_paste"
inpaint_mask_path = "hiyori_masks/face_skin_inpaint.png"
```

---

## Portrait Extraction Pipeline

```
portrait.png
  │
  ├─ MediaPipe FaceMesh → 468 landmarks in portrait space
  │
  ├─ Affine transform (portrait landmarks → template landmarks)
  │
  ├─► face_paste(face_skin)
  │     - warp portrait → face UV space
  │     - paste per drawable
  │     - inpaint(face_skin_inpaint.png)
  │
  ├─► landmark_paste(left_eye, right_eye, left_eyebrow, right_eyebrow, mouth)
  │     - extract landmark-bounded crops
  │     - paste per drawable
  │
  ├─► Hair segmentation (SAM2 or dedicated hair model)
  │     - extract hair region from portrait
  │     - compute average hue from portrait hair pixels
  │     → hue_shift(hair_front, hair_back, hair_side_left, hair_side_right)
  │         (Phase 1 strategy, already implemented)
  │
  └─► Clothing (one of two modes):
        Mode A — Recolor: hue_shift from portrait palette (Phase 1)
        Mode B — Replace: generate with SDXL/Flux → cloth_paste(clothing)
```

### Bang handling

If portrait has bangs over eyebrows (detected by checking landmark visibility score
for brow keypoints < 0.3):
1. Run hair segmentation to get hair mask
2. Inpaint hair mask on portrait → brow-exposed version
3. Run `landmark_paste` for eyebrows on brow-exposed version
4. Use original portrait for all other regions

This matches CartoonAlive Stage 4 handling.

---

## Components

### New files

| File | Responsibility |
|---|---|
| `pipeline/portrait_extractor.py` | MediaPipe landmark detection, affine warp, feature crop extraction |
| `pipeline/mask_generator.py` | Generate inpaint + blend masks from rig render; write to `manifests/<rig>_masks/` |
| `pipeline/inpainter.py` | Telea inpainting (cv2) for face_skin base; optional diffusion fallback |
| `pipeline/hair_segmenter.py` | SAM2 or lightweight hair segmentation model |
| `manifests/hiyori_masks/` | Per-rig mask PNGs |

### Modified files

| File | Change |
|---|---|
| `pipeline/texture_swap.py` | Add `face_paste`, `landmark_paste`, `cloth_paste` dispatch + implementations |
| `pipeline/atlas_config.py` | Add `inpaint_mask_path`, `blend_mask_path` to `AtlasRegion`; update TOML loader |
| `templates/humanoid-anime/atlas_schema.toml` | Update strategies: face_skin → `face_paste`, eyes/brows/mouth → `landmark_paste`, clothing → `cloth_paste` |

---

## Data Flow (Phase 2 → Phase 1)

```
portrait.png
  │
  [Phase 2] pipeline/portrait_extractor.py
  │   → replacements: dict[str, Image.Image]
  │       "face_skin"     → warped portrait crop (pre-inpainted)
  │       "left_eye"      → landmark-bounded crop
  │       "right_eye"     → landmark-bounded crop
  │       "left_eyebrow"  → landmark-bounded crop
  │       "right_eyebrow" → landmark-bounded crop
  │       "mouth"         → landmark-bounded crop
  │       "hair_front"    → target hue swatch (for hue_shift)
  │       "clothing"      → generated outfit image (if Mode B)
  │
  [Phase 1] pipeline/texture_swap.swap_regions(atlases, config, replacements)
  │   → modified atlases: dict[int, Image.Image]
  │
  [Phase 1] pipeline/validate.validate_textures(rig_config, modified_atlases)
      → rendered frame for verification
```

---

## What Phase 2 Does NOT Include

- Full anime style transfer (portrait → anime face). Assumes portrait is already
  in a compatible style or that landmark warp is sufficient.
- Mesh deformation to match portrait face proportions (e.g. wide jaw, narrow eyes).
  Phase 2 uses affine/TPS warp only; the mesh stays fixed.
- Multi-character support (one portrait per rig instance).
- Generative outfit design UI (text prompt → clothing is a separate concern).
- Phase 3: full rig generation (new mesh + new textures from portrait).

---

## Testing Plan

| Test | Type | What it checks |
|---|---|---|
| `test_landmark_detection` | unit | MediaPipe returns expected keypoint count |
| `test_affine_warp` | unit | Known landmark pair → warp produces correct pixel at known position |
| `test_face_paste_inpaint` | unit | face_paste: post-inpaint pixels in mask area match surrounding skin |
| `test_landmark_paste_eye` | unit | landmark_paste: eye region pasted into correct drawable bbox |
| `test_cloth_paste_seam` | unit | cloth_paste: seam pixels are blended, not hard cut |
| `test_mask_generator_coverage` | unit | inpaint mask covers ≥ 80% of left_eye bbox projected to face_skin UV |
| `test_full_extraction_pipeline` | integration | portrait.png → modified atlas → render shows portrait face, no ghost features |

Integration test uses Hiyori + a synthetic test portrait (solid-color face, known hue).
