# Atlas Region Verification — Lessons Learned

**Date:** 2026-04-06
**Status:** Process complete (best-effort). Atlas is correct enough to proceed.

---

## What We Were Doing

Manually verifying and correcting semantic region assignments for all 133 drawables
in Hiyori's Live2D model, stored in `manifests/hiyori_atlas.toml`. Goal: correctly
classify every drawable into named regions (face_skin, body, clothing, hair_*, eyes,
mouth, cheeks, eyebrows) so the texture-swap pipeline knows which atlas region to
recolor.

---

## Live2D Python API Findings

### SetDrawableMultiplyColor is broken for suppression

Using `SetDrawableMultiplyColor(idx, 0, 0, 0, 0)` on all non-target drawables does
NOT fully suppress them. Even with multiply=(0,0,0,0) on ALL drawables, ~7,000 pixels
(~2.7% of a 1024² render) remain non-black. Root cause unknown — confirmed not an
additive-blend issue (csmGetDrawableConstantFlags shows all 133 drawables are normal
blend). This approach produces verification images with ghost rendering that obscures
which drawables actually belong to a region.

### SetPartOpacity works perfectly for suppression

`SetPartOpacity(part_index, 0.0)` on all parts gives exactly zero non-black pixels.
This is the correct suppression primitive. See `scripts/verify_atlas_v2.py` for the
working approach:

```python
# Zero out everything
for pi in range(n_parts):
    model.SetPartOpacity(pi, 0.0)
# Re-enable parts that contain any target drawable
active_parts = {d2p[did] for did in target_set if did in d2p}
for pi in active_parts:
    model.SetPartOpacity(pi, 1.0)
# Highlight targets with screen color
for idx, did in enumerate(all_ids):
    if did in target_set:
        model.SetDrawableScreenColor(idx, *color)
```

### Part structure limits per-drawable isolation

Hiyori has 25 parts. Critically:
- **PartArmA** (index 12) and **PartArmB** (index 13) contain BOTH body-skin drawables
  and sleeve/clothing drawables. Enabling these parts for body verification also makes
  clothing visible, and vice versa.
- **PartEye** (index 3) contains ALL eye drawables (left and right combined). There is
  no separate left-eye / right-eye part, so left_eye and right_eye verification renders
  both show both eyes.

The part→drawable map is in `manifests/hiyori_part_map.json` (generated via ctypes
calls to `csmGetDrawableParentPartIndices` and `csmGetPartIds`).

### ctypes is the only way to get blend mode and part membership

live2d-py exposes no `GetDrawableBlendMode()` or `GetDrawableParentPartIndex()` API.
Both must be accessed via ctypes directly on the Cubism Core shared library. See
`docs/research/2026-04-06-ctypes-uv-extraction-lessons.md` for the ctypes pattern.

---

## Atlas File Management

### Regex `[^\[]*` corrupts TOML when blocks are extracted and re-inserted

A block-extraction regex ending in `[^\[]*` (match until next `[`) strips leading
whitespace and partial tokens from subsequent `[[regions.drawables]]` headers, because
the 2-space indent before `[[` is consumed as "non-bracket characters".

Corruption patterns produced:
- `egions.drawables]]` — `  [[r` consumed
- `.drawables]]` — `  [[regions` consumed  
- `[[regions.drawables]]` (no indent) — `  ` consumed

**Fix:** After any scripted block manipulation, validate with `python3 -c "import tomllib; tomllib.loads(open('file.toml').read())"` immediately. If corrupt, grep for `egions` and `.drawables]]` to find all instances.

**Better approach:** Manipulate atlas entries via a Python script that parses with tomllib,
modifies the data structure, and re-serializes from scratch rather than doing regex
surgery on the raw text.

---

## Final Atlas State

**File:** `manifests/hiyori_atlas.toml`  
**Regions (15):** face_skin, left_eyebrow, right_eyebrow, left_eye, right_eye, mouth,
left_cheek, right_cheek, hair_front, hair_back, hair_side_left, hair_side_right, body,
clothing, cloth_and_body  
**Drawables assigned:** 132 of 133 (1 unaccounted for — not identified)

### Notable classification decisions

| Drawable(s) | Region | Notes |
|---|---|---|
| ArtMesh52, 53 | body | Neck/chin skin, tex=0, order=400 |
| ArtMesh29, 41, 42 | hair_front | Hair wisps near face, order=700. Initially mis-classified as face_skin. |
| ArtMesh82, 84, 87, 89, 90 | body | Hand/wrist skin, order=800, tex=1 |
| ArtMesh68, 69, 71, 72, 74, 75, 77 | clothing | Sleeve fabric in PartArmA, order=300 |
| ArtMesh80, 81 | clothing | Sleeve fabric in PartArmA, order=400 |
| ArtMesh83, 86, 88, 91, 92 | clothing | Sleeve/cuff overlays, order=800/400 |
| ArtMesh85 | body | Large arm skin piece, PartArmB |
| ArtMesh101 | body | Arm skin, tex=1, order=500 |
| ArtMesh102, 104 | cloth_and_body | Texture contains both bare skin and clothing (sock+shin). Cannot be cleanly separated. |
| ArtMesh103 | body | Leg skin, tex=1 |

### cloth_and_body region

ArtMesh102 and ArtMesh104 span both skin and clothing within a single texture region.
Created `cloth_and_body` as a 15th region with `swap_strategy = "hue_shift"`. The swap
pipeline will need to apply a skin mask before hue-shifting to avoid recoloring bare
skin. This is deferred until the swap pipeline is implemented.

---

## What Was Not Resolved

- **1 missing drawable**: 132 assigned out of 133 in the model. Identity unknown.
- **Eye render isolation**: left_eye and right_eye verification renders both show both
  eyes (PartEye contains all eye drawables). Workaround: trust the drawable ID list in
  the TOML; the rendered verification is ambiguous for this region only.
- **Body/clothing overlap in arm parts**: verification renders for body and clothing in
  the arm region inevitably show both. Classifications were verified by texture crop
  (UV bbox from `manifests/hiyori_drawables_raw.json`) rather than rendered isolation.
- **cloth_and_body swap strategy**: placeholder `hue_shift`. Actual implementation
  needs a skin-preserve mask.

---

## Recommendation for Future Rigs

1. Build a proper Python atlas editor that parses/serializes TOML structurally (not regex).
2. Use UV bbox crops from the raw drawable data as the primary classification tool —
   faster and more reliable than rendered isolation.
3. Note the part structure up front (`hiyori_part_map.json` equivalent for the new rig)
   before planning verification renders, to predict which regions will be ambiguous.
