# Rig Authoring Guide — Template Compatibility

**Purpose:** Conventions for authoring Live2D rigs intended to be used as `portrait-to-live2d` template rigs. Rigs built to these conventions enable fully automated atlas config generation.

---

## Core Principle: Name Everything Semantically

Cubism Editor defaults to `ArtMesh0`, `ArtMesh1`, etc. **Never ship a template rig with default names.** Rename every drawable before exporting. Semantic names allow regex-based atlas auto-generation without any VLM or manual labeling.

---

## Drawable Naming Convention

Format: `<region>_<side>_<detail>_<N>`

- `region` — one of the canonical region names below
- `side` — `left`, `right`, or omit if center/symmetric
- `detail` — optional descriptor (`iris`, `lash`, `shadow`, `highlight`, `inner`)
- `N` — integer suffix for physics chain segments (2, 3, 4…); the anchor has no suffix

Examples:
```
face_skin
eye_iris_left
eye_white_right
eye_lash_upper_left
hair_front
hair_side_left          ← anchor
hair_side_left2         ← chain segment 2
hair_side_left3         ← chain segment 3
mouth_inner
left_cheek
clothing_top
clothing_bottom
body_arm_left
```

---

## Canonical Region Names

These are the names the atlas auto-generator will search for via keyword matching. Use the full name as a prefix.

| Prefix | Atlas region | Texture |
|---|---|---|
| `face_skin` | face_skin | tex0 |
| `eye_*_left` / `eye_left_*` | left_eye | tex0 |
| `eye_*_right` / `eye_right_*` | right_eye | tex0 |
| `eyebrow_left` / `brow_left` | left_eyebrow | tex0 |
| `eyebrow_right` / `brow_right` | right_eyebrow | tex0 |
| `mouth_*`, `lip_*`, `tongue`, `fang` | mouth | tex0 |
| `cheek_left` / `blush_left` | left_cheek | tex0 |
| `cheek_right` / `blush_right` | right_cheek | tex0 |
| `hair_front` | hair_front | tex0 |
| `hair_back` | hair_back | tex0 |
| `hair_side_left` | hair_side_left | tex0 |
| `hair_side_right` | hair_side_right | tex0 |
| `body_*`, `arm_*`, `leg_*`, `neck_*` | body | tex1 |
| `clothing_*`, `shirt_*`, `skirt_*`, `sleeve_*` | clothing | tex1 |

---

## Texture Layout

### Two-texture split (recommended)

Use 2 × 2048px textures:

**texture_00.png** — face + hair (everything above the shoulders visible in the typical VTuber frame):
- All face_skin drawables
- All eye, eyebrow, mouth, cheek drawables  
- All hair drawables (front, back, side chains)

**texture_01.png** — body + clothing:
- All body (skin) drawables
- All clothing drawables
- Any accessories

Keeping face/hair on tex0 and body/clothing on tex1 means:
- A portrait-based face swap only touches tex0
- A palette/outfit change only touches tex1
- The two passes are independent and composable

### Pack face regions in the top-left quadrant of tex0

Place face drawables (face_skin, eyes, eyebrows, mouth, cheeks) in the upper-left quadrant of tex0 (x < 1024, y < 1024). Hair can fill the rest. This makes it easy to visually inspect the face region without scrolling.

### Keep related physics chains contiguous in UV space

All segments of a physics strand should sample from an adjacent strip of texture. This means:
- Color changes apply uniformly across the strand
- If you run hue_shift on the chain, all segments shift together
- No gaps in the sampled region that would leave unstyled pixels

### UV packing guidelines

- Face oval: at least 500×500px at 2048px texture for adequate swap resolution
- Each eye cluster: at least 200×200px
- Hair strand segments: at least 80px wide (narrower gets blocky after hue_shift)
- No region should be smaller than 30×30px — too small to meaningfully swap

---

## Draw Order Conventions

Use meaningful draw_order bands (not all-500). These help the atlas auto-generator when names are ambiguous:

| Draw order range | Layer |
|---|---|
| 100–199 | Far background (scene elements, non-character) |
| 200–299 | Hair background (back hair, rear physics) |
| 300–399 | Body background (arms behind body, leg rears) |
| 400–499 | Mid-layer (ears, side hair touching body) |
| 500–599 | Main body / main clothing |
| 600–649 | Face skin |
| 640–649 | Eye outer shell |
| 650–699 | Eye iris + mouth + cheek overlays |
| 700–749 | Eye whites, sclera, specular |
| 750–799 | Hair front (overlapping face frame) |
| 800–849 | Clothing/body overlays, mouth detail |
| 900–999 | Eyebrows (topmost face element) |

---

## What Automated Atlas Generation Does

Given a rig with semantic names, `pipeline/measure_regions.py` will:
1. Run ctypes UV extraction (< 1 second for any size rig)
2. Match drawable IDs against the canonical prefix table
3. Group physics chain segments by base name
4. Assign left/right using UV x-position as tiebreaker
5. Output `manifests/<rig>_atlas.toml` with per-drawable entries
6. Flag any drawable that doesn't match any prefix as `# NEEDS_LABEL`
7. Produce a 3-panel render for flagged drawables (Panel A: isolated, Panel B: in context, Panel C: atlas region)

For rigs using semantic names, step 7 (VLM fallback) should rarely be needed. For anonymous `ArtMeshN` rigs, every drawable goes through step 7.

---

## Hairstyle Variants

A template rig supports texture-only style changes (color/pattern) via the atlas swap pipeline. For full hairstyle shape changes, you need mesh variants:

**Option A — UV remapping**: Author multiple hair meshes with different UV coordinates pointing to different atlas regions. The pipeline selects which UV mapping to use per-character. Requires planning atlas layout at rig-creation time.

**Option B — Mesh variant library**: Ship N alternative hair meshes as separate drawables (e.g., `hair_front_long`, `hair_front_short`, `hair_front_twin_tails`). At generation time, activate the right mesh and deactivate the others. Requires switching blend shapes or deformer visibility.

**Option C — Texture-only (current)**: Different colors/patterns via hue_shift or paste. No shape change. Simplest to implement, adequate for palette personalization.

For the portrait-to-live2d pipeline, Option C is the current implementation. Plan for Option A by reserving atlas space for 2–3 hair style variants at rig creation time.

---

## Checklist Before Exporting Template Rig

- [ ] All drawables renamed from `ArtMeshN` to semantic names
- [ ] Physics chain segments named with sequential integer suffixes
- [ ] Draw_order values set to semantic bands (not default)
- [ ] Face drawables on tex0, body on tex1
- [ ] Face region clustered in top-left quadrant of tex0
- [ ] No hair strand segment narrower than 80px
- [ ] Export `.moc3` + `.model3.json` + textures
- [ ] Run `python -m pipeline.extract_uvs <model.moc3>` — verify all drawables extract without error
- [ ] Run atlas auto-generator — target: zero `# NEEDS_LABEL` flags
