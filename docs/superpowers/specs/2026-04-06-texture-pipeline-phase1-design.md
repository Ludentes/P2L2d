# Texture Pipeline — Phase 1 Design

**Date:** 2026-04-06 (revised after texture inspection + CartoonAlive analysis)
**Scope:** Atlas coordinate config + strategy-aware texture swap + headless validation
**Phase:** 1 of 3 — see `2026-04-06-texture-pipeline-phase2-design.md` for portrait extraction

---

## Goal

Build the mechanical foundation for texture personalization: define where semantic regions
live in a Live2D texture atlas (per-drawable precision), swap those regions using the right
strategy for each region type, and validate the result with a headless render.

Success criterion: apply a solid-color replacement to Hiyori's face_skin → headless render
shows the replacement colour with no UV misalignment.

---

## Why Rectangular Bbox Swap Is Insufficient

Inspection of Hiyori's texture_00.png revealed three structural problems:

**1. Scattered multi-component regions**
Each semantic region is made of 3–12 separate drawables laid out non-contiguously in the
atlas. A union bbox captures mostly empty space:
- `left_eye`: sclera (y≈709) and iris (y≈864) are 155px apart → 650px tall union bbox
- `hair_front`: highlight piece at x=563, dark frame at x=1385 → 1310px wide union bbox

Pasting into a union bbox wastes the paste and distorts proportions.

**2. Hair color = many scattered patches**
Hiyori's hair spans hair_front (two non-adjacent pieces), hair_back, hair_side_left,
hair_side_right. All must shift to the same new hue for consistent color. But the pieces
have different shapes, shading gradients, and highlights — replacing pixels destroys the
existing render quality.

**3. Mouth components are < 60px each**
The mouth in the atlas consists of tiny scattered blobs. A rectangular paste is
meaningless; per-drawable UV precision is required.

---

## Revised Design: Per-Drawable Precision + Swap Strategies

### Core Insight

A semantic region is a **group of drawables** that share a semantic role. The atlas config
stores the UV bounding box of each drawable individually. Swapping operates per-drawable
using the right strategy for the region type.

### Swap Strategies — Complete Registry

| Strategy | Phase | Regions | How |
|---|---|---|---|
| `paste` | 1 | clothing, cheeks | Scale replacement to each drawable's bbox; alpha-composite. Caller provides pre-cropped image. |
| `hue_shift` | 1 | all hair regions | HSV hue rotation on existing pixels; preserves highlights and shading. Caller provides target color swatch. |
| `face_paste` | 2 | face_skin | Affine-warp portrait to UV region; paste; inpaint the feature mask areas (eye sockets, brow ridges, lip line) with surrounding skin. Prevents feature ghost-through during animation. |
| `landmark_paste` | 2 | left_eye, right_eye, left_eyebrow, right_eyebrow, mouth | Landmark-guided crop from portrait; paste per-drawable. Caller provides portrait + detected landmarks. |
| `cloth_paste` | 2 | clothing, body | Paste a generated outfit/skin image into drawables; feather seam edges between adjacent drawables. Includes inpainting at seam boundaries. |

**Phase 1 implements:** `paste`, `hue_shift`

**Phase 2 implements:** `face_paste`, `landmark_paste`, `cloth_paste`

The `atlas_schema.toml` stores the target strategy per region. Phase 1's `swap_regions()`
raises `NotImplementedError` for Phase 2 strategies, allowing gradual rollout.

`hue_shift` is the key Phase 1 upgrade: instead of replacing hair pixels, we rotate their
hue in HSV space. This preserves the existing shading, gradient, and highlight structure.

---

## Architectural Principle: Atlas Config Is a Rig Artifact

Atlas config parallels the manifest system (param name mapping):

| Layer | Params | Atlas |
|---|---|---|
| Template | `schema.toml` — canonical param names | `atlas_schema.toml` — canonical region names |
| Rig | `manifests/hiyori.toml` — coord mapping | `manifests/hiyori_atlas.toml` — per-drawable UV bboxes |

**For existing rigs (Hiyori):** `measure_regions.py` extracts per-drawable UV bboxes via
ctypes, labels them with VLM, writes `manifests/hiyori_atlas.toml`. Committed once.

**For generated rigs (future):** rig generator outputs `atlas_config.toml` as part of
the artifact bundle. UV layout is designed upfront. No measurement needed.

`pipeline/texture_swap.py` is completely generic — it dispatches on `swap_strategy`.

---

## Style Constraint Note

The atlas swap pipeline is style-agnostic. What constrains viable art styles is the
**mesh**, not the textures:

- Hiyori mesh: anime-proportioned. Anime, stylized, watercolour textures look natural.
  Photorealistic textures will look uncanny on exaggerated anime deformers.
- For generated rigs, we choose proportions that match the intended style range.

---

## Components

### 1. `templates/humanoid-anime/atlas_schema.toml`

Unchanged. Defines canonical region names + required/optional flag.

```toml
[[regions]]
name        = "face_skin"
description = "Face skin base (forehead, cheeks, chin)"
required    = true
swap_strategy = "paste"

[[regions]]
name        = "left_eye"
required    = true
swap_strategy = "paste"

[[regions]]
name        = "right_eye"
required    = true
swap_strategy = "paste"

[[regions]]
name        = "left_eyebrow"
required    = true
swap_strategy = "paste"

[[regions]]
name        = "right_eyebrow"
required    = true
swap_strategy = "paste"

[[regions]]
name        = "mouth"
required    = true
swap_strategy = "paste"

[[regions]]
name        = "left_cheek"
required    = false
swap_strategy = "paste"

[[regions]]
name        = "right_cheek"
required    = false
swap_strategy = "paste"

[[regions]]
name        = "hair_front"
required    = true
swap_strategy = "hue_shift"

[[regions]]
name        = "hair_back"
required    = false
swap_strategy = "hue_shift"

[[regions]]
name        = "hair_side_left"
required    = false
swap_strategy = "hue_shift"

[[regions]]
name        = "hair_side_right"
required    = false
swap_strategy = "hue_shift"
```

### 2. `pipeline/atlas_config.py`

```python
@dataclass
class DrawableRegion:
    drawable_id: str        # e.g. "ArtMesh023"
    texture_index: int      # 0 or 1
    x: int                  # left edge of UV bbox in pixels
    y: int                  # top edge
    w: int                  # width
    h: int                  # height
    draw_order: int         # from csmGetDrawableDrawOrders

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"DrawableRegion {self.drawable_id!r}: w and h must be positive")

@dataclass
class AtlasRegion:
    name: str                           # canonical region name
    swap_strategy: str                  # "paste" | "hue_shift"
    drawables: list[DrawableRegion]     # per-drawable UV bboxes

    def bbox(self, texture_index: int | None = None) -> tuple[int, int, int, int]:
        """Union bbox of all drawables (optionally filtered by texture_index)."""
        items = self.drawables
        if texture_index is not None:
            items = [d for d in items if d.texture_index == texture_index]
        if not items:
            raise ValueError(f"No drawables for texture_index={texture_index}")
        x = min(d.x for d in items)
        y = min(d.y for d in items)
        x2 = max(d.x + d.w for d in items)
        y2 = max(d.y + d.h for d in items)
        return x, y, x2 - x, y2 - y

    def texture_indices(self) -> set[int]:
        return {d.texture_index for d in self.drawables}

@dataclass
class AtlasConfig:
    rig_name: str
    template_name: str
    texture_size: int           # assumed square
    regions: list[AtlasRegion]

    def get(self, name: str) -> AtlasRegion: ...   # raises KeyError if not found
    def has(self, name: str) -> bool: ...

def load_atlas_config(path: Path) -> AtlasConfig: ...
```

TOML format — each region contains a list of per-drawable entries:

```toml
rig          = "hiyori"
template     = "humanoid-anime"
texture_size = 2048

[[regions]]
name           = "face_skin"
swap_strategy  = "paste"

  [[regions.drawables]]
  id           = "ArtMesh001"
  texture_index = 0
  x = 29
  y = 20
  w = 500
  h = 575
  draw_order   = 5

[[regions]]
name           = "hair_front"
swap_strategy  = "hue_shift"

  [[regions.drawables]]
  id           = "ArtMesh045"
  texture_index = 0
  x = 563
  y = 40
  w = 464
  h = 402
  draw_order   = 80

  [[regions.drawables]]
  id           = "ArtMesh046"
  texture_index = 0
  x = 1385
  y = 62
  w = 488
  h = 446
  draw_order   = 81
```

### 3. `manifests/hiyori_atlas.toml`

Populated by `measure_regions.py`. Contains one `[[regions.drawables]]` entry per
drawable with exact UV bbox from ctypes extraction and semantic label from VLM.

The current version (manually measured union bboxes) is a placeholder until
`measure_regions.py` runs and generates per-drawable entries.

### 4. `pipeline/texture_swap.py`

Strategy dispatch. No rig-specific logic.

```python
def swap_region(
    atlases: dict[int, Image.Image],
    region: AtlasRegion,
    replacement: Image.Image,          # for "paste": the replacement image
                                        # for "hue_shift": target color (used to extract hue)
) -> dict[int, Image.Image]:
    """Apply strategy-aware swap for one region. Returns modified atlases dict."""
    if region.swap_strategy == "paste":
        return _paste_region(atlases, region, replacement)
    elif region.swap_strategy == "hue_shift":
        return _hue_shift_region(atlases, region, replacement)
    else:
        raise ValueError(f"Unknown swap_strategy: {region.swap_strategy!r}")


def _paste_region(
    atlases: dict[int, Image.Image],
    region: AtlasRegion,
    replacement: Image.Image,
) -> dict[int, Image.Image]:
    """Scale replacement to each drawable's bbox and alpha-composite."""
    out = {k: v.copy() for k, v in atlases.items()}
    # Union bbox of all drawables on each texture
    for tex_idx in region.texture_indices():
        x, y, w, h = region.bbox(texture_index=tex_idx)
        src = replacement.resize((w, h), Image.Resampling.LANCZOS)
        if src.mode != "RGBA":
            src = src.convert("RGBA")
        # Paste each drawable's sub-region
        for d in region.drawables:
            if d.texture_index != tex_idx:
                continue
            # Crop the sub-region of the replacement that corresponds to this drawable
            # within the union bbox
            rx = d.x - x
            ry = d.y - y
            sub = src.crop((rx, ry, rx + d.w, ry + d.h))
            out[tex_idx].paste(sub, (d.x, d.y), sub)
    return out


def _hue_shift_region(
    atlases: dict[int, Image.Image],
    region: AtlasRegion,
    target_color: Image.Image,
) -> dict[int, Image.Image]:
    """Shift hue of existing pixels to match target_color's hue.

    Preserves the original lightness and saturation structure (highlights,
    shading gradients). Only the hue channel rotates.

    target_color: any image; its average hue is extracted and used as target.
    """
    import colorsys
    out = {k: v.copy() for k, v in atlases.items()}

    # Extract target hue from replacement image
    arr = np.array(target_color.convert("RGB")).reshape(-1, 3) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in arr])
    target_h = float(hsv[:, 0].mean())

    for d in region.drawables:
        atlas_arr = np.array(out[d.texture_index]).astype(float)
        patch = atlas_arr[d.y:d.y + d.h, d.x:d.x + d.w]
        alpha = patch[:, :, 3]
        rgb = patch[:, :, :3] / 255.0

        # Shift hue pixel by pixel where alpha > 0
        result = np.zeros_like(patch)
        for row in range(d.h):
            for col in range(d.w):
                if alpha[row, col] < 10:
                    result[row, col] = patch[row, col]
                    continue
                h, s, v = colorsys.rgb_to_hsv(*rgb[row, col])
                r2, g2, b2 = colorsys.hsv_to_rgb(target_h, s, v)
                result[row, col] = [r2 * 255, g2 * 255, b2 * 255, alpha[row, col]]

        atlas_arr[d.y:d.y + d.h, d.x:d.x + d.w] = result
        out[d.texture_index] = Image.fromarray(atlas_arr.astype(np.uint8))

    return out


def swap_regions(
    atlases: dict[int, Image.Image],
    config: AtlasConfig,
    replacements: dict[str, Image.Image],
) -> dict[int, Image.Image]:
    """Batch replacement. Applies each region's strategy."""
    out = {k: v.copy() for k, v in atlases.items()}
    for name, replacement in replacements.items():
        region = config.get(name)
        out = swap_region(out, region, replacement)
    return out
```

Note: The pixel-loop hue shift in `_hue_shift_region` is shown for clarity. The
implementation should use numpy vectorized HSV conversion for performance.

### 5. `pipeline/validate.py`

Unchanged from previous spec. Uses `rig/render.py` headless renderer.

```python
def validate_textures(
    config: RigConfig,
    modified_atlases: dict[int, Image.Image],
) -> np.ndarray:
    """Render with modified textures. Returns (H, W, 4) RGBA frame."""
    ...

def check_region_color(
    frame: np.ndarray,
    expected_color: tuple[int, int, int],
    tolerance: int = 20,
) -> bool:
    """Check that some pixels in the render match expected_color (within tolerance)."""
    ...
```

### 6. `pipeline/measure_regions.py`

Three-phase automated tool. Produces `manifests/<rig>_atlas.toml` with per-drawable
entries (not union bboxes).

**Phase 1 — UV extraction (ctypes)**
Calls Cubism Core C functions via ctypes (exported symbols in `live2d.so`):
- `csmReviveMocInPlace` + `csmInitializeModelInPlace` to load the moc3
- `csmGetDrawableIds`, `csmGetDrawableTextureIndices`, `csmGetDrawableVertexUvs`,
  `csmGetDrawableVertexCounts`, `csmGetDrawableDrawOrders` per drawable
- Output: list of `{id, texture_index, uv_bbox, draw_order}` for all drawables

**Phase 2 — Render isolation (live2d-py)**
For each drawable `i`:
- Set ALL drawables' multiply color to `(0, 0, 0, 1)` → renders black
- Set drawable `i`'s screen color to `(0, 255, 0, 255)` → renders solid green
- Render neutral pose → only drawable `i` appears
- Save the isolated render frame for VLM input

Also render the full model at neutral pose once for context.

**Phase 3 — VLM labeling (local VLM or Claude API)**
Group drawables by UV bbox proximity. For each group, compose a 3-panel image and call
the VLM (see `docs/runbooks/drawable-labeling-playbook.md` for the prompt).

Output format: `manifests/<rig>_atlas.toml` with:
- Per-drawable UV bbox (from ctypes, not estimated)
- Semantic region name (from VLM) → `name` on parent `[[regions]]`
- `swap_strategy` inherited from `atlas_schema.toml`
- Low-confidence entries flagged `# REVIEW: <note>`

CLI:
```
uv run python -m pipeline.measure_regions \
    --model3 /path/to/hiyori.model3.json \
    --rig hiyori \
    --template humanoid-anime \
    --out manifests/hiyori_atlas.toml
```

**Dependency:** No external VLM API required — the playbook prompt is designed to be
run by the operator (Claude Code session) rather than via an autonomous API call. The
tool outputs the 3-panel images; the operator runs the playbook and edits the TOML.
(Fully automated API mode can be added later.)

---

## Data Flow

```
Input: replacement images (one per region name)
       manifests/hiyori_atlas.toml  (per-drawable UV bboxes + strategies)
         ↓
load_atlas_config() → AtlasConfig
         ↓
swap_regions(atlases, config, replacements)
  → paste regions: per-drawable scaled paste
  → hue_shift regions: HSV hue rotation preserving shading
         ↓
validate.validate_textures(rig_config, modified_atlases) → frame
         ↓
Visual confirmation / automated pixel check
```

---

## Integration Point: Phase 1 → Phase 2

Phase 2 (see `2026-04-06-texture-pipeline-phase2-design.md`) sits above Phase 1.
Its output is the `replacements: dict[str, Image.Image]` dict that `swap_regions()` accepts,
plus the new strategy implementations added to `texture_swap.py`.

| Region | Phase 2 provides | Phase 1 strategy used |
|---|---|---|
| `face_skin` | warped+inpainted portrait crop | `face_paste` (Phase 2) |
| `left_eye`, `right_eye`, brows, mouth | landmark-guided crops | `landmark_paste` (Phase 2) |
| `hair_*` | average hue from portrait hair | `hue_shift` (Phase 1) ✓ |
| `clothing` | generated outfit image | `cloth_paste` (Phase 2) |
| `body` | warped skin-tone image | `face_paste` or `paste` (Phase 2) |

Phase 1 does not depend on Phase 2. Phase 2 requires Phase 1's atlas config to be in place.

---

## What Phase 1 Does NOT Include

- Portrait ingestion of any kind (landmark detection, affine warp, segmentation)
- `face_paste`, `landmark_paste`, `cloth_paste` strategy implementations
- Mask generation or inpainting
- AI/diffusion generation for clothing
- HaiMeng support (EULA-pending)
- UV polygon warp (non-rectangular paste). Phase 1 paste uses drawable bbox;
  true polygon warp uses vertex mesh. Phase 2 enhancement.

These are specified in `2026-04-06-texture-pipeline-phase2-design.md`.

---

## Testing Plan

| Test | Type | What it checks |
|---|---|---|
| `test_load_atlas_config` | unit | TOML loads to `AtlasConfig`, drawables parsed |
| `test_atlas_config_get` | unit | `get()` returns correct region, raises `KeyError` |
| `test_atlas_region_bbox` | unit | Union bbox computed correctly from drawables |
| `test_swap_paste_pixels` | unit | paste strategy: replacement pixels appear at drawable bbox |
| `test_swap_paste_preserves_outside` | unit | Pixels outside drawables unchanged |
| `test_swap_hue_shift_changes_hue` | unit | hue_shift: hue changes, lightness preserved |
| `test_swap_hue_shift_alpha` | unit | Fully transparent pixels not modified |
| `test_swap_regions_batch` | unit | Multiple regions swapped, correct strategies applied |
| `test_validate_textures` | integration | Modified atlas → headless render → non-null frame |
| `test_validate_region_color` | integration | Red-filled face region appears red in render |

Integration tests skip if Hiyori model files not present.

---

## File Layout After Phase 1

```
templates/
  humanoid-anime/
    atlas_schema.toml          ← updated (adds swap_strategy per region)
manifests/
  hiyori.toml                  ← existing (param mapping)
  hiyori_atlas.toml            ← per-drawable UV bboxes (from measure_regions.py)
pipeline/
  __init__.py
  atlas_config.py              ← updated (DrawableRegion, AtlasRegion.drawables)
  texture_swap.py              ← updated (strategy dispatch, hue_shift)
  validate.py                  ← new
  measure_regions.py           ← new (produces per-drawable TOML)
tests/
  pipeline/
    test_atlas_config.py       ← updated
    test_texture_swap.py       ← updated
    test_validate.py           ← new (integration, skips if no model)
docs/
  runbooks/
    drawable-labeling-playbook.md  ← existing
```
