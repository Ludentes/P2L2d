# Research: Texture Atlas Tools & Live2D UV Extraction

**Date:** 2026-04-06  
**Sources:** 12 sources — GitHub repos, official Live2D docs, binary symbol inspection

---

## Executive Summary

No dedicated "semantic atlas reader" Python library exists — atlas coordinates are just JSON/TOML parsing. Our custom TOML format is the right call. More importantly: **UV coordinates for every drawable are extractable from the moc3 via ctypes**, because the Cubism Core C functions (`csmGetDrawableVertexUvs`, `csmGetDrawableTextureIndices`) are exported symbols in the live2d-py shared library (`live2d.so`). Hiyori's drawables are not semantically named (all `ArtMesh0..133`), so user labeling is still required once, but it can be presented as a visual pick-from-overlay workflow rather than manual box-drawing.

---

## Key Findings

### 1. Python sprite atlas tools: just JSON/TOML parsing

There is no Python library that manages "named semantic regions" of a texture atlas — the ecosystem consists of packers (PyTexturePacker, TexturePacker) that create atlases, not readers that interpret them semantically. The TexturePacker JSON format is the de facto standard for atlas metadata: a file with named frames, each having `x`, `y`, `w`, `h` in pixel coordinates. Python reads this with `json.load()`. Pygame Arcade's `TextureAtlas` class is internal to the Arcade game engine and not usable standalone.

**Conclusion:** our TOML format with `[[regions]]` blocks (name, texture_index, x, y, w, h) is correct and requires no third-party library. The TexturePacker JSON format is an alternative that offers broader toolchain compatibility but adds no semantic value for our use case.

### 2. UV data is accessible via Cubism Core ctypes — no moc3 parser needed

The live2d-py shared library (`live2d.so`) embeds the Cubism Core C library and exports its raw C API symbols. Confirmed via `nm -D live2d.so`:

```
T csmGetDrawableCount
T csmGetDrawableIds
T csmGetDrawableTextureIndices    ← which texture (0 or 1) per drawable
T csmGetDrawableVertexCounts      ← vertex count per drawable
T csmGetDrawableVertexUvs         ← float[drawable][vertex][2], UV in [0,1]
T csmGetDrawableIndexCounts
T csmGetDrawableIndices
T csmReviveMocInPlace             ← load moc3 binary into memory
T csmGetSizeofModel
T csmInitializeModelInPlace       ← create model struct from moc3
T csmUpdateModel
```

This means we can:
1. Read the moc3 binary file
2. Call `csmReviveMocInPlace` + `csmInitializeModelInPlace` via ctypes
3. Call `csmGetDrawableVertexUvs` → per-drawable vertex UV pairs in normalized [0,1] space
4. Convert to pixel coords: `x = u * texture_size`, `y = v * texture_size`
5. Compute bounding box per drawable: `(min_u, min_v, max_u - min_u, max_v - min_v)` in pixels

No third-party moc3 parser needed. The moc3 format spec (rentry.co/moc3spec) confirms UV data is stored in an "UVs" section with `uvSourcesBeginIndex` per art mesh and `textureNo` as the texture index. The ctypes approach reads this directly via the official Cubism Core routines.

**moc3ingbird (CVE-2023-27566)** is a DoS exploit, not a parser. **moc3-reader-re** (Java, archived 2024) does parse the format but is Java only and incomplete. Neither is useful here.

### 3. Hiyori drawable IDs are not semantic

Hiyori has 133 drawables, all named `ArtMesh`, `ArtMesh0`, `ArtMesh1`, ... `ArtMesh134`. No semantic names (no "FaceSkin", "EyeLeft", etc.). User labeling is required to map drawable groups → canonical region names.

**Impact on `measure_regions.py`:** instead of asking the user to manually draw boxes, we:
1. Extract UV bounding boxes for all 133 drawables via ctypes
2. Overlay them on the texture atlas image (colour-coded, labeled ArtMesh0..N)
3. User clicks/assigns which drawable group → which canonical region name
4. Tool outputs `manifests/hiyori_atlas.toml` with exact UV-derived coordinates

This is more accurate than hand-drawn boxes and scales to any rig.

### 4. Generated rigs can use semantic drawable names → auto-atlas-config

When P2L generates a new rig, we name drawables semantically during the Cubism workflow ("FaceSkin", "EyeLeft", etc.). Then `measure_regions.py` in non-interactive mode can auto-map by name. The atlas config is derived without user input.

### 5. live2d-py utils are OpenGL rendering helpers, not UV tools

`live2d/utils/image.py` and `canvas.py` are OpenGL quad renderers for displaying textures on screen. They have nothing to do with atlas region management or UV extraction. The only useful texture-related code in live2d-py for our purposes is the bundled Cubism Core C API accessed via ctypes.

---

## Comparison: atlas coordinate approaches

| Approach | Accuracy | User effort | Works for any rig |
|---|---|---|---|
| Manual box-drawing | Low (eyeball) | High | Yes |
| csm UV extraction + labeling | Exact (from moc3) | Low (label once) | Yes |
| Semantic drawable name matching | Exact, zero labeling | None | Only for rigs we generate |

---

## Open Questions

- **ctypes calling convention**: need to verify that `csmReviveMocInPlace` and `csmInitializeModelInPlace` can be called without the full Cubism Framework init (they're pure Core, not Framework). Likely yes — Core is independent of Framework.
- **Drawable grouping**: multiple ArtMesh IDs may form one semantic region (e.g., face skin has multiple overlapping meshes). The UV extraction gets bboxes per drawable; the user labels groups of them as one region.

---

## Sources

[1] nm -D output on live2d.so (local inspection, 2026-04-06)  
[2] rentry.co/moc3spec — MOC3 file format specification  
[3] github.com/Arkueid/live2d-py/blob/main/package/live2d/v3/live2d.pyi — LAppModel method list  
[4] docs.live2d.com/en/cubism-sdk-manual/cubism-core-api-reference/ — Cubism Core API reference  
[5] github.com/OpenL2D/moc3ingbird — DoS exploit (CVE-2023-27566), not a parser  
[6] github.com/QiE2035/moc3-reader-re — Java moc3 reader, archived March 2024  
[7] github.com/wo1fsea/PyTexturePacker — Python packer, not a reader  
[8] api.arcade.academy — Pygame Arcade TextureAtlas (game-engine specific)  
[9] Live2D drawable IDs from Hiyori (local inspection via GetDrawableIds(), 2026-04-06)  
[10] live2d/utils/image.py, canvas.py (local inspection, 2026-04-06)
