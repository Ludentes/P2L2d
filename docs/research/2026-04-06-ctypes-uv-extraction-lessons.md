# Lessons Learned: ctypes UV Extraction + Atlas Design

**Date:** 2026-04-06
**Rig:** Hiyori (74 params, 2 × 2048px textures, 133 drawables)

Raw extraction output: `manifests/hiyori_drawables_raw.json`
Atlas config: `manifests/hiyori_atlas.toml`

---

## 1. Draw order is the primary semantic signal

Hiyori's 133 drawables cluster into tight draw_order bands that map directly to semantic layers. No VLM needed when you have draw_order:

| draw_order range | semantic layer |
|---|---|
| 200 | hair background (rear volume) |
| 400 | ears |
| 550 | neck / body skin |
| 600 | face skin + hair front (highlights) |
| 640 | eye outer shell (whites/irises) |
| 650 | iris details, cheeks, mouth base |
| 700 | sclera highlights, specular dots |
| 750 | hair front (dark frame) + right side strands |
| 800 | mouth overlay |
| 900 | eyebrows |

**Implication for `measure_regions.py`**: render isolation + VLM labeling can be guided/validated by draw_order bands, reducing ambiguous cases.

---

## 2. Rectangular union bboxes are misleading

The original atlas design computed a single bounding box per semantic region by taking the union of all drawables' UVs. For hair_front this produced a 1310px-wide bbox spanning two completely separate texture patches (x=549 and x=1376) — 827px apart with nothing in between. Sampling from that union bbox would have captured large amounts of irrelevant texture.

**Fix**: Store per-drawable UV bboxes. `AtlasRegion` holds a list of `DrawableRegion`, each with exact pixel bounds for one ArtMesh. The swap operation is applied independently per drawable.

---

## 3. Hair requires hue rotation, not paste

Pasting a replacement image over hair destroys the shading, highlights, and shadow detail baked into the texture by the artist. The desired behavior is "change the hair color" — the existing value (lightness) and saturation relationships should be preserved.

**Solution**: `hue_shift` swap strategy. Convert the drawable's texture patch to HSV, replace the H channel with the target color's hue, convert back. Transparent pixels (alpha < 128) are skipped.

This is distinct from face/eye regions where paste is correct — those regions need full pixel replacement because portrait skin and eye textures are not pre-shaded in the same way.

---

## 4. Physics strand chains = one semantic region across many drawables

Live2D physics strands are segmented into 5–8 individual ArtMesh drawables per strand, each covering a short section of the strand length. All segments share the same UV region (they sample the same texture patch in sequence). Treating each segment as a separate region is wrong — they must be grouped under one `AtlasRegion` and hue-shifted together.

Hiyori counts:
- `hair_side_left`: 10 drawables (ArtMesh57, 58, 65, 109–116)
- `hair_side_right`: 24 drawables (ArtMesh59, 60, 66, 117–137)

**Implication**: an atlas config with only the "anchor" drawable per strand would leave most of the strand un-shifted on color swap.

---

## 5. Cheek blush uses two identical-UV drawables for additive blending

`left_cheek` has ArtMesh4 and ArtMesh6 at exactly the same UV coordinates `(23,1170,226,164)`. Similarly `right_cheek` has ArtMesh3 and ArtMesh5 at `(257,1170,251,155)`. This is not a measurement error — Live2D rigs commonly render the same blush patch twice with an additive blend mode to achieve a soft glow effect.

Both drawables must be included in the atlas region. Swap targets both identically.

---

## 6. Mouth is small and positioned between eye columns in atlas space

The mouth region (ArtMesh1 + ArtMesh2) occupies only ~97×115px total on a 2048px texture. Its atlas x-position (213–310px) places it horizontally between the left and right eye columns, not below them as one might expect. The spatial layout of face components in the UV atlas does not correspond to their screen positions.

UV atlas x-positions for reference:
- left_eye region: x ≈ 15–200px
- mouth: x ≈ 213–310px
- right_eye region: x ≈ 300–550px
- face_skin: x ≈ 15–547px

**Implication**: do not infer atlas position from screen position. Always use measured UV coords.

---

## 7. Left/right in atlas x-position is consistent but needs render verification

For Hiyori, lower x-values in the atlas correspond to the character's left side (left_eye, left_eyebrow, left_cheek all have lower x than their right counterparts). This is a coincidence of how the rig was authored — not guaranteed for other rigs. The character's left/right is the mirror of the viewer's left/right (standard Live2D convention).

**Implication**: `measure_regions.py` must produce a Panel B (full render with overlay) to verify left/right assignments before committing an atlas config.

---

## 8. ctypes extraction gives ground truth in seconds

Visual analysis of texture files produces estimates ~20px off from actual UV coordinates. For a 97px-wide mouth region a 20px error is a 20% bbox sizing error. The ctypes path (`csmGetDrawableVertexUvs`, `csmGetDrawableTextureIndices`, `csmGetDrawableDrawOrders`, `csmGetDrawableIds`) runs in under 1 second and gives exact integer-pixel bboxes for all 133 drawables simultaneously.

Key ctypes symbol notes:
- `csmGetSizeofMoc` does **not** exist in live2d.so. Use `csmReviveMocInPlace` directly with moc3 buffer.
- moc3 buffer must be **64-byte aligned** — use `ctypes.create_string_buffer` with alignment or allocate a larger buffer and offset into it.
- UV coordinates are returned as floats in [0,1]. Multiply by texture_width/texture_height to get pixel coords.
- Vertex UVs are per-vertex, not per-drawable bbox. Must compute `(min_u, min_v, max_u - min_u, max_v - min_v)` per drawable.

---

## 9. atlas_schema.toml defines the contract; per-rig atlas TOML is the implementation

`templates/humanoid-anime/atlas_schema.toml` declares which region names exist and their swap strategies. `manifests/hiyori_atlas.toml` fills in the actual ArtMesh IDs and UV coords for one specific rig. This separation means:
- A new rig only needs a new manifest, not a schema change.
- The schema can evolve (add `body`, `clothing` etc.) independently of rig manifests.
- `pipeline/validate.py` checks a manifest against its template's schema to catch missing required regions.

---

## Key Numbers (Hiyori)

| region | drawables | texture | approx area |
|---|---|---|---|
| face_skin | 7 | tex 0 | 532×603 px (main) |
| left_eye | 12 | tex 0 | cluster ~200×250 px |
| right_eye | 15 | tex 0 | cluster ~200×250 px |
| left_eyebrow | 2 | tex 0 | 148×71 px |
| right_eyebrow | 2 | tex 0 | 133×67 px |
| mouth | 4 | tex 0 | ~97×115 px |
| left_cheek | 2 | tex 0 | 226×164 px |
| right_cheek | 2 | tex 0 | 251×155 px |
| hair_front | 3 | tex 0 | two patches: 500×438 + 507×476 |
| hair_back | 1 | tex 0 | 869×1166 px |
| hair_side_left | 10 | tex 0 | strand segments |
| hair_side_right | 24 | tex 0 | multiple strands |
