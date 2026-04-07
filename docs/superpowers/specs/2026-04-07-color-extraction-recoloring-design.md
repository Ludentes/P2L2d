# Color Extraction + Atlas Recoloring — Design

**Date:** 2026-04-07
**Status:** Design
**Replaces:** AI-based per-region editing (FLUX Fill, Kontext, QwenEdit, ControlNet+PuLID) — all failed to preserve anime style when given a real photo reference.

## Problem

All AI-based approaches for transferring portrait identity into Live2D atlas regions produce photorealistic output when given a real photo as reference. The domain gap (photo → anime) cannot be bridged by prompting or parameter tuning. We need a deterministic, non-AI approach.

## Approach

Extract specific colors from the portrait photo (hair, eyes, skin, lips, clothing), then apply those colors to the corresponding atlas regions via color-space remapping. No generation — only recoloring existing anime textures.

This preserves:
- Anime line art and shading
- Alpha channels (pixel-perfect boundaries)
- Artistic style of the original template rig

## Architecture

```
Portrait Image
    │
    ├─ MediaPipe landmarks (478 pts) ─── sampling zones (iris, lips, cheeks)
    │
    ├─ BiSeNet face parsing ──────────── pixel masks (hair, skin, clothing)
    │
    └─ Color extraction ──────────────── ColorPalette dataclass
                                              │
                                              ▼
Atlas Textures (2 × 2048px) ──► Region recoloring ──► Modified atlas
                                  (per-region strategy)
```

### Two-stage pipeline

**Stage 1: `extract_palette(portrait) → ColorPalette`**

Uses MediaPipe landmarks + BiSeNet face parsing to sample colors from the portrait:

| Color          | Sampling method                                  | Color space |
|----------------|--------------------------------------------------|-------------|
| `hair`         | BiSeNet hair mask → median LAB of masked pixels  | LAB         |
| `skin`         | Landmark cheek points (116, 345) → 15px patches  | LAB         |
| `eye_color`    | Iris landmarks (468-472, 473-477) → median hue   | HSV         |
| `lip_color`    | Inner lip landmarks → median within mask         | LAB         |
| `clothing`     | BiSeNet clothing mask → dominant color (k-means k=2, pick largest cluster) | LAB |

Output: `ColorPalette` dataclass with LAB/HSV values for each attribute.

**Stage 2: `recolor_atlas(atlases, palette, atlas_config) → modified atlases`**

Per-region recoloring using the strategy best suited for each region type:

| Region(s)                    | Strategy        | Details |
|------------------------------|-----------------|---------|
| `hair_front/back/side_*`     | Hue rotation    | Compute hue delta from template hair → portrait hair. Rotate hue of saturated pixels (S > 15). Adjust saturation toward target. Preserve V channel (shading). |
| `left_eye`, `right_eye`      | Hue rotation    | Hue-rotate all saturated pixels (S > 15) toward portrait iris color. Iris pixels are colored (shifted), eye whites and line art are desaturated (preserved automatically). No need to identify specific iris drawables. |
| `face_skin`, `body`          | LAB a\*b\* shift | Compute (a\*, b\*) delta from template skin → portrait skin. Apply to all opaque pixels. Preserves L\* (luminance/shading) exactly. |
| `left_cheek`, `right_cheek`  | Tint blend      | Blend cheek blush toward portrait skin's a\* (redness). Light touch — these are additive blush layers. |
| `mouth`                      | LAB a\*b\* shift | Same as skin but with lip color target. Only lip drawables. |
| `clothing`                   | Hue rotation    | Same as hair — rotate hue of saturated pixels toward clothing color. |
| `cloth_and_body`             | Split strategy  | Classify pixels as skin vs cloth by saturation/hue proximity to template values, apply appropriate transform to each. |
| `left/right_eyebrow`         | Hue rotation    | Match to hair color (eyebrows follow hair). |

### Key design decisions

**Why hue rotation for hair/eyes/clothing:** These regions have strong, saturated colors where hue is the dominant perceptual attribute. Rotating hue preserves the anime shading gradients (value channel) and saturation structure.

**Why LAB a\*b\* shift for skin:** Skin tones vary primarily along the a\* (green-red) and b\* (blue-yellow) axes. Shifting in LAB preserves luminance (shading, highlights, shadows) exactly. HSV fails for skin because skin has low saturation and hue is unstable near the red-yellow boundary.

**Why we skip low-saturation pixels:** Line art, outlines, and shadows in anime textures are near-black/near-gray (S < 15). Skipping them preserves the artistic structure. Only "colored" pixels get recolored.

**Saturation threshold:** S > 15 (in 0-255 HSV scale) separates colored areas from line art. This is conservative — even slightly tinted pixels will be shifted.

## Components

### `pipeline/color_extract.py`

```python
@dataclass
class ColorPalette:
    hair: np.ndarray        # LAB [L, a, b]
    skin: np.ndarray        # LAB [L, a, b]
    eye_color: float        # HSV hue (0-180)
    eye_saturation: float   # HSV saturation (0-255)
    lip_color: np.ndarray   # LAB [L, a, b]
    clothing: np.ndarray    # LAB [L, a, b]

def extract_palette(portrait: Image.Image) -> ColorPalette:
    """Extract color palette from portrait using landmarks + face parsing."""
```

Dependencies: `mediapipe`, `numpy`, `opencv-python`, `scikit-learn` (for k-means on clothing).

Face parsing via BiSeNet (`zllrunning/face-parsing.PyTorch`, vendored or via `facer` pip package):
- Label 1: skin
- Label 10: hair
- Label 16: cloth
- Labels 4/5: eyes
- Labels 12/13: upper/lower lip

If BiSeNet is unavailable, fall back to approximate region sampling:
- Hair: top 20% of portrait above face bbox
- Clothing: area below face bbox
- Skin/lips/eyes: MediaPipe landmarks only (no parsing needed)

### `pipeline/color_apply.py`

```python
def recolor_atlas(
    atlases: list[Image.Image],
    palette: ColorPalette,
    atlas_config: AtlasConfig,
    template_palette: ColorPalette | None = None,
) -> list[Image.Image]:
    """Apply extracted colors to atlas regions."""
```

`template_palette` is the baseline colors of the unmodified template. If not provided, extract from the atlas itself (measure dominant color per region).

Internal helpers:
- `_hue_rotate(crop, source_hue, target_hue, sat_threshold=15)` — HSV hue rotation for saturated pixels
- `_lab_shift(crop, source_lab, target_lab, sat_threshold=15)` — LAB a\*b\* channel shift preserving L\*
- `_tint_blend(crop, target_ab, strength=0.3)` — light tint for blush/cheek layers

### `pipeline/template_palette.py`

```python
def extract_template_palette(
    atlases: list[Image.Image],
    atlas_config: AtlasConfig,
) -> ColorPalette:
    """Measure baseline colors from the unmodified template atlas."""
```

Measures the dominant color of each region type in the template to compute deltas. Cached per template (run once, store as JSON alongside the atlas TOML).

### Integration with existing pipeline

The color extraction + recoloring replaces `region_edit.py` as the texture personalization step. The pipeline becomes:

```
portrait
  → face_align.detect_landmarks()     # existing
  → color_extract.extract_palette()    # NEW
  → color_apply.recolor_atlas()        # NEW
  → texture_swap.swap_regions()        # existing (if needed)
  → package.package_output()           # existing
```

No ComfyUI dependency. No GPU required (except optionally for BiSeNet inference, which is ~13MB and runs in <1s on CPU).

## Testing

1. **Unit tests for color transforms:**
   - Hue rotation: known input → expected output (e.g., red → blue = 120 degree shift)
   - LAB shift: verify L\* unchanged, a\*b\* shifted correctly
   - Saturation threshold: verify line art pixels unchanged

2. **Visual regression test:**
   - Extract palette from `assets/data/image1.png`
   - Recolor Hiyori atlas
   - Render via RigRenderer
   - Save comparison grid (original vs recolored)
   - Human inspection (no automated pixel comparison — perceptual quality)

3. **Round-trip test:**
   - Recolored atlas → package → render → verify no rendering artifacts

## Error handling

- **No face detected:** `MediaPipeLandmarkError` (already exists)
- **BiSeNet finds no hair/clothing:** Fall back to sampling from approximate image regions (top 20% = hair, bottom 40% = clothing)
- **Very dark/light portrait (extreme exposure):** LAB shift still works — it's relative, not absolute
- **Grayscale portrait:** All hues will be ~0, saturation ~0. Regions stay at template colors (delta = 0). Acceptable degradation.

## Scope boundaries

**In scope:**
- Color extraction from portrait
- Deterministic recoloring of all 14 atlas region types
- Template palette measurement
- Integration into pipeline/run.py

**Out of scope:**
- Face shape changes (Live2D mesh deformation)
- Hair style changes (different template needed)
- Accessory detection/addition
- Multiple clothing colors (take dominant)
