# UV Remap Design — Clean Rectangular Atlas Regions for AI Editing

**Date:** 2026-04-07
**Status:** Draft

## Problem

Hiyori's texture atlas has heavily overlapping UV bounding boxes between semantic regions (5.14x overlap ratio on tex0). While the actual triangle coverage is non-overlapping (confirmed by pixel-level analysis), the interleaved layout makes it impossible to crop a clean rectangle and get "just the face" or "just the hair" for AI editing.

AI texture generation (FLUX Fill, Kontext, QwenEdit) needs clean rectangular regions per semantic body part to produce coherent edits without cross-region contamination.

## Goal

Modify Hiyori's moc3 UV coordinates so each semantic region occupies a non-overlapping rectangular area on the texture atlas. Produce matching texture images where pixels are rearranged to match the new UV layout. The rendered model must be visually identical to the original.

## Approach: Per-Region Bounding-Box Packing (Approach A)

### Why this approach

- **Zero pixel overlap between regions on tex0** (confirmed by `analyze_uv_overlap.py`). Only tex1 has minor arm_a/arm_b overlap (8,741 pixels, 1.3% of arm area).
- Simple linear UV transform per vertex (translate + scale).
- Bounding-box rectangles are ideal for AI model input — clean crops, predictable sizes.
- Foreign pixels within a region's bbox are harmless: Live2D renders by triangle, not by rectangle.

### Fallback: Approach C (Individual Mesh Packing)

If Approach A produces editing artifacts from foreign pixels in shared bboxes, fall back to per-mesh packing. This is viable when combined with see-through layer rendering (CartoonAlive technique), where each mesh gets its own clean texture render. Each of 133 meshes gets its own atlas rectangle. More complex but gives maximum editorial isolation.

## Architecture

### Semantic Region Groups

Based on Hiyori's part hierarchy (confirmed by analysis):

| Region | Parts | Texture | Triangle Coverage |
|--------|-------|---------|-------------------|
| face | PartFace | 0 | ~3.5% |
| eyes | PartEye, PartEyeBall | 0 | ~4.2% |
| brows | PartBrow | 0 | ~0.8% |
| mouth | PartMouth | 0 | ~1.1% |
| nose | PartNose | 0 | ~0.3% |
| ear | PartEar | 0 | ~0.5% |
| hair_front | PartHairFront | 0 | ~1.2% |
| hair_back | PartHairBack | 0 | ~4.3% |
| hair_side | PartHairSide | 0 | ~0.2% |
| neck | PartNeck | 0 | ~0.5% |
| skinning | PartSkinning | 0 | ~3.4% |
| body | PartBody | 1 | ~10.2% |
| arm_a | PartArmA | 1 | ~3.0% |
| arm_b | PartArmB | 1 | ~2.8% |

Total triangle coverage: ~11.4% of tex0, ~16.0% of tex1. Plenty of room for repacking — even a single 2048 sheet could fit everything with margin.

### Data Flow

```
Input:  hiyori.moc3 + texture_00.png + texture_01.png
                    │
                    ▼
         ┌──────────────────┐
         │  1. Parse moc3   │  Read UVs, triangles, part assignments
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  2. Group meshes │  Assign each art mesh to a semantic region
         │     by region    │  based on parent part index → part name
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  3. Compute UV   │  Per-region union bounding box in UV space
         │     bboxes       │  (tight bbox of all mesh UVs in region)
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  4. Pack regions │  Assign non-overlapping rectangles on
         │     into atlas   │  new atlas sheet(s). Use shelf-packing.
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  5. Transform    │  For each vertex in each region:
         │     UV coords    │  new_uv = (old_uv - old_min) / old_size
         │                  │         * new_size + new_min
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  6. Copy pixels  │  For each region, copy old_bbox pixels
         │     to new atlas │  from old texture to new_bbox in new texture
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  7. Write output │  Modified moc3 + new texture PNGs
         └──────────────────┘
```

### UV Transform

For each art mesh vertex `(u, v)` belonging to region `R`:

```python
# Old region bbox in UV space [0,1]
old_min_u, old_min_v = R.old_bbox.min_u, R.old_bbox.min_v
old_w, old_h = R.old_bbox.width, R.old_bbox.height

# New region bbox in UV space [0,1] (on new atlas)
new_min_u, new_min_v = R.new_bbox.min_u, R.new_bbox.min_v
new_w, new_h = R.new_bbox.width, R.new_bbox.height

# Transform (preserves relative position within region)
new_u = (u - old_min_u) / old_w * new_w + new_min_u
new_v = (v - old_min_v) / old_h * new_h + new_min_v
```

Since we preserve aspect ratio and only translate+scale, the mapping is exact (no interpolation artifacts).

### Texture Consolidation

Current: 2 texture sheets × 2048px, ~13.5% average utilization.
After remap: All regions fit on a **single 2048px sheet** (total coverage ~27.4% = 11.4% + 16.0%).

This means:
- All `art_mesh.texture_indices` change to `0` (single sheet)
- The `model3.json` texture list reduces to one entry
- Less VRAM at runtime

If single-sheet packing is too tight (need margin for AI editing at edges), use 4096px or keep 2 sheets.

### Packing Algorithm

Use simple **shelf packing** (good enough for ~15 rectangles):

1. Sort regions by height (tallest first)
2. Place regions left-to-right on shelves
3. Start new shelf when current one fills up
4. Target: 2048×2048 sheet with ~4px padding between regions

The packing produces a `RegionMap` — a dict mapping region name → `(x, y, w, h)` in pixel coordinates on the new atlas. This map is the key output: it tells AI editing code exactly where to crop each region.

### Region Map Output

Save as `region_map.json` alongside the modified model:

```json
{
  "atlas_size": 2048,
  "regions": {
    "face": {"x": 0, "y": 0, "w": 512, "h": 480},
    "eyes": {"x": 512, "y": 0, "w": 400, "h": 350},
    "hair_back": {"x": 0, "y": 480, "w": 600, "h": 400},
    ...
  }
}
```

AI editing pipeline reads this map to crop/paste regions.

## Components

### `pipeline/uv_remap.py` — Core module

```python
@dataclass
class RegionDef:
    name: str
    part_names: list[str]  # parts that belong to this region

@dataclass  
class RegionBBox:
    name: str
    texture_index: int
    min_u: float; min_v: float
    max_u: float; max_v: float
    mesh_indices: list[int]  # art mesh indices in this region

@dataclass
class PackedRegion:
    name: str
    old_bbox: RegionBBox      # original UV bbox
    new_x: int; new_y: int    # pixel position on new atlas
    new_w: int; new_h: int    # pixel size on new atlas

def define_regions() -> list[RegionDef]:
    """Default Hiyori region groupings."""

def compute_region_bboxes(moc: Moc3, regions: list[RegionDef]) -> list[RegionBBox]:
    """Compute per-region UV bounding boxes from mesh data."""

def pack_regions(bboxes: list[RegionBBox], atlas_size: int = 2048, padding: int = 4) -> list[PackedRegion]:
    """Shelf-pack region bboxes into a new atlas layout."""

def remap_uvs(moc: Moc3, packing: list[PackedRegion]) -> None:
    """Modify moc3 UV coordinates in-place to match new atlas layout."""

def remap_textures(
    old_textures: dict[int, np.ndarray],
    packing: list[PackedRegion],
    atlas_size: int = 2048,
) -> np.ndarray:
    """Copy pixel regions from old textures to new atlas. Returns single RGBA image."""

def remap_model(
    moc3_path: Path,
    texture_paths: list[Path],
    output_dir: Path,
    atlas_size: int = 2048,
    regions: list[RegionDef] | None = None,
) -> dict:
    """End-to-end: read model, remap UVs, write modified model + textures + region_map.json."""
```

### `scripts/test_uv_remap.py` — Validation script

1. Load original Hiyori, render at default pose → `original.png`
2. Run `remap_model()` → modified moc3 + new textures + region_map.json
3. Load modified model, render at default pose → `remapped.png`
4. Compare: pixel diff, PSNR (should be >40dB, ideally identical)
5. Visualize: overlay region boundaries on new atlas, save debug images
6. Save all outputs to `test_output/uv_remap/`

### Modifications to `art_mesh.texture_indices`

If consolidating to a single texture sheet, all texture indices change from their current values (0 or 1) to 0. This is a simple bulk write to the moc3 section.

### Modifications to `model3.json`

The model3.json texture list must match:
```json
"FileReferences": {
  "Moc": "hiyori.moc3",
  "Textures": ["textures/texture_00.png"]
}
```

## Edge Cases

1. **arm_a/arm_b pixel overlap on tex1**: Treat arms as a single region `arms` or accept 1.3% shared pixels (harmless — both arms use the same skin tone).

2. **PartCore (HitArea)**: Tiny mesh (6 vertices), not visually relevant. Include in a "misc" region or drop.

3. **Unmatched parts** (Part, Part2, Part3, Part4): Generic container parts with no art meshes. Ignored.

4. **Padding**: 4px padding between regions prevents bilinear sampling bleed at region edges during rendering.

5. **Keyform UV deltas**: Art mesh keyforms store position deltas (`keyform_position.xys`) that animate vertex positions in screen space. **UVs are static** — keyforms don't modify UVs. So our UV remap is valid across all animation states.

## Testing

- **Round-trip render comparison**: Original vs remapped model at default pose. Must be pixel-identical (or PSNR >40dB if bilinear sampling introduces sub-pixel differences at new region edges).
- **Multi-pose test**: Render at several parameter configurations to verify no UV drift.
- **Region map validation**: Each region in region_map.json must contain exactly the pixels for its semantic group (verify by rasterizing triangles on new atlas).
- **moc3 structural integrity**: `moc3 verify` (from py-moc3 CLI) on the output file must still pass structural checks (section alignment, count consistency).

## Success Criteria

1. Modified moc3 + single new texture renders identically to original across all poses
2. `region_map.json` enables clean rectangular crops of each semantic region
3. Each region crop contains only that region's visual content (no cross-contamination)
4. AI editing a single region (e.g., changing face skin color) affects only that region when rendered
