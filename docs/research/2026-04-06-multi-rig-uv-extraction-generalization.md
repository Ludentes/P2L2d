# Multi-Rig UV Extraction: Findings & Generalization

**Date:** 2026-04-06
**Rigs analyzed:** Hiyori (dev, 2×2048px), Akari (4096px), Wanko (1024px)
**Raw data:** `manifests/hiyori_drawables_raw.json`, `manifests/akari_drawables_raw.json`, `manifests/wanko_drawables_raw.json`

---

## Rig Comparison Table

| Property | Hiyori | Akari | Wanko |
|---|---|---|---|
| Drawables | 133 | 283 | 33 |
| Textures | 2 × 2048px | 1 × 4096px | 1 × 1024px |
| Naming | `ArtMeshN` (anonymous) | semantic English | `D_CATEGORY_NN` |
| Draw order spread | 200–900 (semantic) | all 500 | 100–900 |
| Character type | humanoid | humanoid | pet mascot |

---

## Finding 1: Draw order is rig-specific, not a universal signal

Hiyori's draw_order bands map cleanly to semantic layers (200=hair_back, 600=face, 900=eyebrows). This was useful for Hiyori because its drawables are anonymous — draw_order was the only structural signal available.

Akari collapses all 283 drawables to draw_order 500. The signal contains zero information. Any atlas mapping tool that relies on draw_order bands will fail silently on Akari — it will not crash, it will just put everything in the same bucket.

Wanko uses draw_order but for non-humanoid semantics (bowl layers, background, face). The bands don't map to humanoid regions at all.

**Implication:** draw_order is a fallback heuristic for anonymous rigs only. Primary identification strategy must be drawable name matching.

---

## Finding 2: Three distinct naming conventions in the wild

### Type A — Anonymous (Hiyori)
`ArtMesh0`, `ArtMesh1`, ..., `ArtMesh137`
- Default Cubism Editor behavior when rigger doesn't rename drawables
- Extremely common in freely-distributed VTS models
- Requires draw_order + UV position heuristics or VLM to assign semantic labels
- Most labor-intensive to label

### Type B — Semantic English (Akari)
`eye_iris_left`, `hair_back`, `leg_left_main`, `shirt_skirt_shade_right`, `tie_bottom_left2`, ...
- Full semantic names directly in drawable IDs
- Regex/keyword matching trivially identifies face/hair/body regions
- Example patterns: `eye_*` → eye region, `hair_*` → hair region, `*_left`/`*_right` → side disambiguation
- ArtMesh count suffixes (`left2`, `left3`) indicate physics chain position
- This is the ideal naming convention for automated atlas config generation

### Type C — Prefixed category code (Wanko)
`D_BACKGROUND_00`, `D_BOWL_02`, `D_FACE_00`, `D_BODY_00`
- Cubism convention for mascot/prop rigs
- Category is human-readable but uses custom categories unrelated to humanoid regions
- Regex works but requires custom category→region mapping per rig type

---

## Finding 3: Physics strand naming encodes chain position

In Akari, `tie_bottom_left2` through `tie_bottom_left6` are a single physics strand with 5 segments. Similarly `hair_left2` through `hair_left8` is a 7-segment strand. The number suffix is the segment index within the chain (starting from 2, with the anchor being the unsuffixed drawable).

Pattern: `<semantic_name>` = anchor + `<semantic_name>N` = segment N of chain.

For atlas labeling, all chain segments should be grouped under the same `AtlasRegion` regardless of suffix. The grouping rule: strip trailing digits, group by base name.

---

## Finding 4: Texture count and resolution encode complexity

| Rigs | Texture count | Resolution |
|---|---|---|
| Simple mascots (Wanko) | 1 | 1024px |
| Standard humanoid (Hiyori) | 2 | 2048px |
| Detailed humanoid (Akari) | 1 | 4096px |
| Production (HaiMeng Textoon) | 9 | 4096px |

More textures = semantic partitioning (face vs body). Higher resolution = more detail per drawable. A single 4096px texture (Akari) has the same pixel budget as four 2048px textures.

For generated rigs: prefer 2 textures at 2048px (face+hair/body split) over 1 texture at 4096px. The semantic split makes atlas config generation easier and allows independent face and body swap.

---

## Finding 5: Body texture (tex1) assignment is uncertain without rendering

For Hiyori's tex1, draw_order bands gave a reasonable (but unverified) split: order 300+400 = skin background, order 500 left-side = clothing, order 500 right-side = skin arm pieces, order 800 right-side = skin overlays. However:

- ArtMesh97 at x=1001 was assigned to body based on position heuristic — could be clothing
- ArtMesh101 at x=617 was assigned to clothing — could be an arm skin panel  
- The skirt vs legs boundary is unclear without rendering

**Rule:** Face regions (tex0, humanoid) can be reliably labeled from draw_order + UV position. Body regions (tex1) require per-drawable rendering (Panel A output from `measure_regions.py`) to confirm skin vs clothing.

---

## Generalization: Atlas Config Generation Strategy by Rig Type

### For Type A (anonymous, draw_order spread):
1. Extract all drawables
2. Group by draw_order band
3. For each group, use UV position heuristics to assign left/right:
   - Lower x → character's left side
   - Higher x → character's right side
4. Assign canonical region by band: 900=eyebrows, 700/640=eyes, 650=iris+mouth+cheeks, 600=face, 200/750=hair
5. Flag tex1 drawables for manual/VLM verification
6. Output confidence scores; flag anything confidence < 0.7 for `# REVIEW`

### For Type B (semantic names):
1. Extract all drawables
2. Apply keyword matching to IDs:
   - `eye_iris_*`, `eye_white_*`, `eye_closed_*`, `eye_top_*` → left_eye/right_eye (via `_left`/`_right` suffix)
   - `eye_brows_*` → left_eyebrow/right_eyebrow
   - `mouth_*`, `lip_*`, `tongue`, `fang` → mouth
   - `hair_back*`, `back_hair_*` → hair_back
   - `hair_left*`, `hair_mid_left*`, `hair_over_shade_left*`, `uh_left*` → hair_side_left
   - `face_*` (excluding face_shock_effect) → face_skin
   - `blush_*`, `cheek_*` → left_cheek/right_cheek
   - `leg_*`, `neck_*`, `arm_*`, `finger_*`, `thumb_*`, `belly_*`, `oppai_*` → body
   - `shirt_*`, `skirt_*`, `sleeve_*`, `collar_*`, `tie_*`, `shoe_*` → clothing
3. Group physics chain segments by base name (strip trailing digit)
4. Verify left/right with UV x-position as sanity check

### For Type C (prefixed category):
1. Build custom category→region mapping for rig type
2. Apply prefix matching
3. Same chain-grouping logic as Type B

---

## Implications for Generated Rig Design

When we generate our own template rigs for the portrait-to-live2d pipeline, the naming choice has a major downstream impact on how easily we can auto-generate atlas configs. **Use Type B (semantic English) naming exclusively.**

See `docs/runbooks/rig-authoring-guide.md` for full rigging conventions.
