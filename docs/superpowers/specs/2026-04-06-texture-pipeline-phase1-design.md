# Texture Pipeline — Phase 1 Design

**Date:** 2026-04-06  
**Scope:** Atlas coordinate config + texture region swap + headless validation  
**Phase:** 1 of 2 (Phase 2 = AI generation from portrait, separate brainstorm)

---

## Goal

Build the mechanical foundation for texture personalization: define where semantic regions
live in a Live2D texture atlas, swap those regions programmatically, and validate the result
with a headless render. No AI generation in Phase 1.

Success criterion: paste a solid-color block into Hiyori's face skin region → headless render
shows the replacement colour on the model's face with no UV misalignment.

---

## Architectural Principle: Atlas Config Is a Rig Artifact

Atlas config parallels the manifest system (param name mapping):

| Layer | Params | Atlas |
|---|---|---|
| Template | `schema.toml` — canonical param names | `atlas_schema.toml` — canonical region names |
| Rig | `manifests/hiyori.toml` — coord mapping | `manifests/hiyori_atlas.toml` — pixel coords |

**For existing rigs (Hiyori):** measure once with `measure_regions.py` → commit
`manifests/hiyori_atlas.toml`. Never needs to change unless the rig file changes.

**For generated rigs (future):** the rig generator outputs `atlas_config.toml` as part of the
artifact bundle (spec-driven from `atlas_schema.toml`). No manual measurement needed.
The atlas layout is designed upfront when we design the rig.

This means `pipeline/texture_swap.py` is completely generic — it works with any
`AtlasConfig`, whether measured or generated.

---

## Style Constraint Note

The atlas swap pipeline is style-agnostic (PIL coordinate paste). What constrains viable
art styles is the **mesh**, not the textures:

- Hiyori mesh: anime-proportioned. Anime, stylized, watercolour textures look natural.
  Photorealistic textures will look uncanny on exaggerated anime deformers.
- For generated rigs (when P2L controls the mesh), we choose proportions that match the
  intended style range. No pipeline-level pigeonholing.

---

## Components

### 1. `templates/humanoid-anime/atlas_schema.toml`

Defines canonical region names for this template type. Rig manifests must provide
coordinates for all required regions. Optional regions may be omitted.

```toml
# Canonical texture region names for humanoid-anime rigs.
# Each rig provides pixel coordinates for these in manifests/<rig>_atlas.toml.

[[regions]]
name        = "face_skin"
description = "Face skin base (forehead, cheeks, chin)"
required    = true

[[regions]]
name        = "left_eye"
description = "Left eye including white, iris, pupil, lashes"
required    = true

[[regions]]
name        = "right_eye"
description = "Right eye"
required    = true

[[regions]]
name        = "left_eyebrow"
required    = true

[[regions]]
name        = "right_eyebrow"
required    = true

[[regions]]
name        = "mouth"
description = "Mouth open/closed region"
required    = true

[[regions]]
name        = "left_cheek"
description = "Cheek blush / highlight"
required    = false

[[regions]]
name        = "right_cheek"
required    = false

[[regions]]
name        = "hair_front"
description = "Front hair layer (bangs, frame)"
required    = true

[[regions]]
name        = "hair_back"
description = "Back hair layer"
required    = false

[[regions]]
name        = "hair_side_left"
required    = false

[[regions]]
name        = "hair_side_right"
required    = false
```

### 2. `pipeline/atlas_config.py`

```python
@dataclass
class AtlasRegion:
    name: str           # canonical name, e.g. "face_skin"
    texture_index: int  # 0 or 1 (texture_00 / texture_01)
    x: int              # left edge in pixels
    y: int              # top edge in pixels
    w: int              # width in pixels
    h: int              # height in pixels

@dataclass
class AtlasConfig:
    rig_name: str
    template_name: str
    texture_size: int          # assumed square (2048 for Hiyori)
    regions: list[AtlasRegion]

    def get(self, name: str) -> AtlasRegion: ...   # raises KeyError if not found
    def has(self, name: str) -> bool: ...

def load_atlas_config(path: Path) -> AtlasConfig: ...
```

TOML format:
```toml
rig        = "hiyori"
template   = "humanoid-anime"
texture_size = 2048

[[regions]]
name          = "face_skin"
texture_index = 0
x = 412
y = 88
w = 280
h = 320
# ... etc
```

### 3. `manifests/hiyori_atlas.toml`

Hiyori-specific atlas coordinates. Populated by running `measure_regions.py` once on
Hiyori's `texture_00.png` and `texture_01.png`, then committed.

This file is the one-time manual artifact. All downstream code reads it via
`load_atlas_config()`.

### 4. `pipeline/texture_swap.py`

Generic region paste. No rig-specific logic.

```python
def swap_region(
    atlas: Image.Image,
    region: AtlasRegion,
    replacement: Image.Image,
) -> Image.Image:
    """Paste replacement into atlas at region coordinates. Alpha compositing."""
    out = atlas.copy()
    src = replacement.resize((region.w, region.h), Image.LANCZOS)
    if src.mode != "RGBA":
        src = src.convert("RGBA")
    out.paste(src, (region.x, region.y), src)
    return out

def swap_regions(
    atlases: dict[int, Image.Image],   # {texture_index: PIL image}
    config: AtlasConfig,
    replacements: dict[str, Image.Image],   # {region_name: replacement image}
) -> dict[int, Image.Image]:
    """Batch replacement. Returns modified atlas images."""
    ...
```

### 5. `pipeline/validate.py`

Uses `rig/render.py` headless renderer. Copies modified textures to a temp dir alongside
Hiyori model files, renders neutral pose, returns the frame as `np.ndarray`.

```python
def validate_textures(
    config: RigConfig,
    modified_atlases: dict[int, Image.Image],
) -> np.ndarray:
    """Render with modified textures. Returns (H, W, 4) RGBA frame."""
    ...
```

Also provides a simple pixel-region check:
```python
def check_region_color(
    frame: np.ndarray,
    expected_color: tuple[int, int, int],
    tolerance: int = 20,
) -> bool:
    """Check that some pixels in the render match expected_color (within tolerance)."""
    ...
```

### 6. `pipeline/measure_regions.py`

One-time interactive tool. Opens `texture_00.png` and `texture_01.png` side by side
(using matplotlib or PIL + tkinter), lets user draw bounding boxes, prints TOML entries
to stdout. Not part of the production pipeline — run once per new existing rig, commit
the output.

CLI:
```
uv run python -m pipeline.measure_regions --rig hiyori
```

---

## Data Flow

```
Input: replacement images (one per region name)
       manifests/hiyori_atlas.toml  (coordinates)
         ↓
load_atlas_config() → AtlasConfig
         ↓
swap_regions(atlases, config, replacements) → modified atlas images
         ↓
validate.validate_textures(rig_config, modified_atlases) → frame
         ↓
Visual confirmation / automated pixel check
```

---

## Future Integration Point (Phase 2)

Phase 2 will generate replacement images from a portrait via SDXL img2img + segmentation.
The output of Phase 2 is exactly the `replacements: dict[str, Image.Image]` dict that
Phase 1's `swap_regions()` already accepts. Phase 2 plugs in above Phase 1 with no
changes to Phase 1 code.

---

## What Phase 1 Does NOT Include

- AI/diffusion generation
- Portrait → anime style transfer
- Segmentation of input portrait into components
- HaiMeng support (EULA-pending)
- Body/clothing texture regions (can be added to atlas_schema later)
- Atlas config generation for new rigs (just the measure tool for existing rigs)

---

## Testing Plan

| Test | Type | What it checks |
|---|---|---|
| `test_load_atlas_config` | unit | TOML loads to `AtlasConfig`, types correct |
| `test_atlas_config_get` | unit | `get()` returns correct region, raises `KeyError` |
| `test_swap_region_pixels` | unit | Paste solid red block → verify pixels at region coords |
| `test_swap_preserves_other_regions` | unit | Pixels outside region unchanged |
| `test_swap_alpha_compositing` | unit | RGBA replacement blends correctly |
| `test_validate_textures` | integration | Modified atlas → headless render → non-null frame |
| `test_validate_region_color` | integration | Red-filled face region appears red in render |

Integration tests skip if Hiyori model files not present (same pattern as existing tests).

---

## File Layout After Phase 1

```
templates/
  humanoid-anime/
    atlas_schema.toml          ← new (canonical region names)
manifests/
  hiyori.toml                  ← existing (param mapping)
  hiyori_atlas.toml            ← new (measured pixel coords)
pipeline/
  __init__.py
  atlas_config.py              ← new
  texture_swap.py              ← new
  validate.py                  ← new
  measure_regions.py           ← new (one-time tool)
tests/
  pipeline/
    test_atlas_config.py       ← new
    test_texture_swap.py       ← new
    test_validate.py           ← new (integration, skips if no model)
```
