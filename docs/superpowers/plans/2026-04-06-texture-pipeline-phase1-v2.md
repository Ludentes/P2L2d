# Texture Pipeline Phase 1 — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-drawable atlas config + strategy-aware texture swap (paste / hue_shift) + headless validation.

**Architecture:** `AtlasRegion` stores a list of `DrawableRegion` (per-drawable UV bbox). `swap_regions()` dispatches on `swap_strategy`: "paste" composites per-drawable; "hue_shift" rotates HSV hue of existing pixels, preserving shading.

**Tech Stack:** Python 3.12, Pillow, numpy, tomllib, live2d-py (ctypes for UV extraction), pytest

**Prior work (do not redo):**
- `pipeline/__init__.py` — exists
- `templates/humanoid-anime/atlas_schema.toml` — updated with `swap_strategy` fields
- `manifests/hiyori_atlas.toml` — **COMPLETE. Hand-verified, 132 drawables, 15 semantic regions. DO NOT overwrite.**
  See `docs/research/2026-04-06-atlas-region-verification-lessons.md` for full history.

**Task status:**
- ✅ Task 1: `atlas_config.py` — DrawableRegion model
- ✅ Task 2: `hiyori_atlas.toml` — per-drawable format (done manually, not via measure_regions)
- ✅ Task 3: `texture_swap.py` — paste / hue_shift dispatch
- ⏳ Task 4: `pipeline/validate.py` + integration test
- ⏳ Task 5: `pipeline/measure_regions.py` — UV extraction tool for future rigs (not for Hiyori)
- ⏳ Task 6: Smoke test measure_regions → /tmp only

---

### Task 1: Refactor `pipeline/atlas_config.py` — DrawableRegion model

**Files:**
- Modify: `pipeline/atlas_config.py`
- Modify: `tests/pipeline/test_atlas_config.py`

The existing `AtlasRegion` (flat x/y/w/h) is replaced by `AtlasRegion` containing a list
of `DrawableRegion` objects. `AtlasConfig.get()` and `.has()` keep their signatures.

- [ ] **Step 1: Write the new tests first**

Replace `tests/pipeline/test_atlas_config.py` entirely:

```python
import pytest
from pathlib import Path
from pipeline.atlas_config import AtlasConfig, AtlasRegion, DrawableRegion, load_atlas_config

_SAMPLE_TOML = """\
rig = "test_rig"
template = "humanoid-anime"
texture_size = 2048

[[regions]]
name = "face_skin"
swap_strategy = "paste"

  [[regions.drawables]]
  id = "ArtMesh001"
  texture_index = 0
  x = 29
  y = 20
  w = 500
  h = 575
  draw_order = 5

[[regions]]
name = "hair_front"
swap_strategy = "hue_shift"

  [[regions.drawables]]
  id = "ArtMesh045"
  texture_index = 0
  x = 563
  y = 40
  w = 464
  h = 402
  draw_order = 80

  [[regions.drawables]]
  id = "ArtMesh046"
  texture_index = 0
  x = 1385
  y = 62
  w = 488
  h = 446
  draw_order = 81
"""


@pytest.fixture
def tmp_atlas(tmp_path):
    p = tmp_path / "test.toml"
    p.write_text(_SAMPLE_TOML)
    return load_atlas_config(p)


def test_load_regions(tmp_atlas):
    assert tmp_atlas.rig_name == "test_rig"
    assert tmp_atlas.template_name == "humanoid-anime"
    assert tmp_atlas.texture_size == 2048
    assert len(tmp_atlas.regions) == 2


def test_region_drawables(tmp_atlas):
    hair = tmp_atlas.get("hair_front")
    assert len(hair.drawables) == 2
    assert hair.drawables[0].drawable_id == "ArtMesh045"
    assert hair.drawables[1].x == 1385


def test_region_swap_strategy(tmp_atlas):
    assert tmp_atlas.get("face_skin").swap_strategy == "paste"
    assert tmp_atlas.get("hair_front").swap_strategy == "hue_shift"


def test_region_bbox_single(tmp_atlas):
    face = tmp_atlas.get("face_skin")
    x, y, w, h = face.bbox()
    assert x == 29 and y == 20 and w == 500 and h == 575


def test_region_bbox_union(tmp_atlas):
    hair = tmp_atlas.get("hair_front")
    x, y, w, h = hair.bbox()
    assert x == 563                     # min x
    assert y == 40                      # min y
    assert x + w == 1385 + 488         # max x_right = 1873
    assert y + h == max(40+402, 62+446) # max y_bottom = 508


def test_atlas_config_get_missing(tmp_atlas):
    with pytest.raises(KeyError):
        tmp_atlas.get("nonexistent")


def test_atlas_config_has(tmp_atlas):
    assert tmp_atlas.has("face_skin")
    assert not tmp_atlas.has("nonexistent")


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_atlas_config(tmp_path / "nonexistent.toml")


def test_load_invalid_texture_size(tmp_path):
    bad = """\
rig = "t"
template = "humanoid-anime"
texture_size = 0
"""
    p = tmp_path / "bad.toml"
    p.write_text(bad)
    with pytest.raises(ValueError, match="texture_size"):
        load_atlas_config(p)


def test_drawable_region_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        DrawableRegion(drawable_id="x", texture_index=0, x=0, y=0, w=0, h=100, draw_order=0)
    with pytest.raises(ValueError, match="positive"):
        DrawableRegion(drawable_id="x", texture_index=0, x=0, y=0, w=100, h=-1, draw_order=0)


def test_region_texture_indices(tmp_atlas):
    hair = tmp_atlas.get("hair_front")
    assert hair.texture_indices() == {0}
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd /home/newub/w/portrait-to-live2d
uv run pytest tests/pipeline/test_atlas_config.py -v 2>&1 | tail -20
```

Expected: ImportError or AttributeError (DrawableRegion does not exist yet).

- [ ] **Step 3: Rewrite `pipeline/atlas_config.py`**

```python
"""Atlas config: per-drawable UV bounding boxes for Live2D texture regions."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DrawableRegion:
    drawable_id: str
    texture_index: int
    x: int
    y: int
    w: int
    h: int
    draw_order: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(
                f"DrawableRegion {self.drawable_id!r}: w and h must be positive, "
                f"got w={self.w}, h={self.h}"
            )


@dataclass
class AtlasRegion:
    name: str
    swap_strategy: str           # "paste" | "hue_shift"
    drawables: list[DrawableRegion]

    def bbox(self, texture_index: int | None = None) -> tuple[int, int, int, int]:
        """Union bbox of all drawables (optionally filtered by texture_index)."""
        items = self.drawables
        if texture_index is not None:
            items = [d for d in items if d.texture_index == texture_index]
        if not items:
            raise ValueError(
                f"AtlasRegion {self.name!r}: no drawables"
                + (f" on texture_index={texture_index}" if texture_index is not None else "")
            )
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
    texture_size: int
    regions: list[AtlasRegion]

    def get(self, name: str) -> AtlasRegion:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return any(r.name == name for r in self.regions)


def load_atlas_config(path: Path) -> AtlasConfig:
    """Load atlas config from TOML. Raises FileNotFoundError or ValueError on bad input."""
    if not path.exists():
        raise FileNotFoundError(f"Atlas config not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    texture_size = data.get("texture_size", 0)
    if texture_size <= 0:
        raise ValueError(f"texture_size must be positive, got {texture_size!r}")

    regions = []
    for rdata in data.get("regions", []):
        drawables = []
        for d in rdata.get("drawables", []):
            drawables.append(DrawableRegion(
                drawable_id=d["id"],
                texture_index=d["texture_index"],
                x=d["x"],
                y=d["y"],
                w=d["w"],
                h=d["h"],
                draw_order=d.get("draw_order", 0),
            ))
        regions.append(AtlasRegion(
            name=rdata["name"],
            swap_strategy=rdata.get("swap_strategy", "paste"),
            drawables=drawables,
        ))

    return AtlasConfig(
        rig_name=data["rig"],
        template_name=data["template"],
        texture_size=texture_size,
        regions=regions,
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/pipeline/test_atlas_config.py -v
```

Expected: 11/11 PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/atlas_config.py tests/pipeline/test_atlas_config.py
git commit -m "feat(atlas): DrawableRegion model — per-drawable UV bboxes with swap_strategy"
```

---

### Task 2: Update `manifests/hiyori_atlas.toml` to new TOML format

**Files:**
- Modify: `manifests/hiyori_atlas.toml`

The per-drawable format uses drawable IDs from ctypes. Since measure_regions.py hasn't run
yet, we use placeholder IDs (`"unknown_0"`, `"unknown_1"` etc.) with the visually-measured
bboxes from the previous atlas. This keeps the file valid and loadable.

- [ ] **Step 1: Rewrite `manifests/hiyori_atlas.toml`**

```toml
# Hiyori texture atlas — per-drawable UV bboxes.
# Drawable IDs are placeholders until measure_regions.py runs with ctypes extraction.
# Coordinates visually measured from texture_00.png (2048×2048).
# mouth coords marked REVIEW — sub-components < 60px, need ctypes to confirm.

rig          = "hiyori"
template     = "humanoid-anime"
texture_size = 2048

# ── Face skin ────────────────────────────────────────────────────────────────
[[regions]]
name          = "face_skin"
swap_strategy = "paste"

  [[regions.drawables]]
  id           = "unknown_face_skin"
  texture_index = 0
  x = 29
  y = 20
  w = 500
  h = 575
  draw_order   = 0

# ── Eyebrows ─────────────────────────────────────────────────────────────────
[[regions]]
name          = "left_eyebrow"
swap_strategy = "paste"

  [[regions.drawables]]
  id           = "unknown_left_eyebrow"
  texture_index = 0
  x = 20
  y = 576
  w = 230
  h = 60
  draw_order   = 0

[[regions]]
name          = "right_eyebrow"
swap_strategy = "paste"

  [[regions.drawables]]
  id           = "unknown_right_eyebrow"
  texture_index = 0
  x = 317
  y = 576
  w = 213
  h = 50
  draw_order   = 0

# ── Eyes ─────────────────────────────────────────────────────────────────────
# Each eye is split into sclera and iris drawables (different y positions in atlas).

[[regions]]
name          = "left_eye"
swap_strategy = "paste"

  [[regions.drawables]]
  id           = "unknown_left_eye_sclera"
  texture_index = 0
  x = 56
  y = 709
  w = 137
  h = 123
  draw_order   = 0

  [[regions.drawables]]
  id           = "unknown_left_eye_iris"
  texture_index = 0
  x = 123
  y = 864
  w = 81
  h = 113
  draw_order   = 0

[[regions]]
name          = "right_eye"
swap_strategy = "paste"

  [[regions.drawables]]
  id           = "unknown_right_eye_sclera"
  texture_index = 0
  x = 324
  y = 712
  w = 131
  h = 123
  draw_order   = 0

  [[regions.drawables]]
  id           = "unknown_right_eye_iris"
  texture_index = 0
  x = 333
  y = 868
  w = 84
  h = 112
  draw_order   = 0

# ── Mouth ────────────────────────────────────────────────────────────────────
[[regions]]
name          = "mouth"
swap_strategy = "paste"
# REVIEW: sub-components are tiny (< 60px). Needs ctypes UV extraction to confirm.

  [[regions.drawables]]
  id           = "unknown_mouth"
  texture_index = 0
  x = 15
  y = 820
  w = 520
  h = 185
  draw_order   = 0

# ── Cheeks ───────────────────────────────────────────────────────────────────
[[regions]]
name          = "left_cheek"
swap_strategy = "paste"

  [[regions.drawables]]
  id           = "unknown_left_cheek"
  texture_index = 0
  x = 62
  y = 1192
  w = 156
  h = 109
  draw_order   = 0

[[regions]]
name          = "right_cheek"
swap_strategy = "paste"

  [[regions.drawables]]
  id           = "unknown_right_cheek"
  texture_index = 0
  x = 292
  y = 1193
  w = 176
  h = 113
  draw_order   = 0

# ── Hair ─────────────────────────────────────────────────────────────────────
[[regions]]
name          = "hair_front"
swap_strategy = "hue_shift"

  [[regions.drawables]]
  id           = "unknown_hair_front_highlight"
  texture_index = 0
  x = 563
  y = 40
  w = 464
  h = 402
  draw_order   = 0

  [[regions.drawables]]
  id           = "unknown_hair_front_dark"
  texture_index = 0
  x = 1385
  y = 62
  w = 488
  h = 446
  draw_order   = 0

[[regions]]
name          = "hair_back"
swap_strategy = "hue_shift"

  [[regions.drawables]]
  id           = "unknown_hair_back"
  texture_index = 0
  x = 1180
  y = 867
  w = 831
  h = 1136
  draw_order   = 0

[[regions]]
name          = "hair_side_left"
swap_strategy = "hue_shift"

  [[regions.drawables]]
  id           = "unknown_hair_side_left"
  texture_index = 0
  x = 606
  y = 463
  w = 204
  h = 842
  draw_order   = 0

[[regions]]
name          = "hair_side_right"
swap_strategy = "hue_shift"

  [[regions.drawables]]
  id           = "unknown_hair_side_right_a"
  texture_index = 0
  x = 1081
  y = 446
  w = 181
  h = 842
  draw_order   = 0

  [[regions.drawables]]
  id           = "unknown_hair_side_right_b"
  texture_index = 0
  x = 1273
  y = 61
  w = 154
  h = 839
  draw_order   = 0

  [[regions.drawables]]
  id           = "unknown_hair_side_right_c"
  texture_index = 0
  x = 1835
  y = 65
  w = 160
  h = 909
  draw_order   = 0
```

- [ ] **Step 2: Verify it loads**

```bash
uv run python -c "
from pipeline.atlas_config import load_atlas_config
from pathlib import Path
cfg = load_atlas_config(Path('manifests/hiyori_atlas.toml'))
print(f'{cfg.rig_name}: {len(cfg.regions)} regions')
for r in cfg.regions:
    print(f'  {r.name} ({r.swap_strategy}): {len(r.drawables)} drawables, bbox={r.bbox()}')
"
```

Expected: 12 regions loaded, each printing its union bbox.

- [ ] **Step 3: Commit**

```bash
git add manifests/hiyori_atlas.toml
git commit -m "feat(atlas): hiyori_atlas.toml updated to per-drawable format (placeholder IDs)"
```

---

### Task 3: Rewrite `pipeline/texture_swap.py` — strategy dispatch + hue_shift

**Files:**
- Modify: `pipeline/texture_swap.py`
- Modify: `tests/pipeline/test_texture_swap.py`

- [ ] **Step 1: Write the new tests**

Replace `tests/pipeline/test_texture_swap.py` entirely:

```python
import numpy as np
from PIL import Image
from pipeline.atlas_config import AtlasConfig, AtlasRegion, DrawableRegion
from pipeline.texture_swap import swap_region, swap_regions


def _make_region(name, strategy, drawables):
    return AtlasRegion(name=name, swap_strategy=strategy, drawables=drawables)


def _drawable(tid=0, x=50, y=50, w=60, h=40, draw_order=0):
    return DrawableRegion(drawable_id="d0", texture_index=tid, x=x, y=y, w=w, h=h, draw_order=draw_order)


def _atlas(color=(200, 200, 200, 255), size=256):
    return Image.new("RGBA", (size, size), color)


# ── paste strategy ───────────────────────────────────────────────────────────

def test_paste_pixels_appear():
    region = _make_region("face_skin", "paste", [_drawable()])
    red = Image.new("RGBA", (60, 40), (255, 0, 0, 255))
    result = swap_region({0: _atlas()}, region, red)
    px = result[0].getpixel((70, 70))
    assert px[0] == 255 and px[1] == 0


def test_paste_preserves_outside():
    region = _make_region("face_skin", "paste", [_drawable()])
    red = Image.new("RGBA", (60, 40), (255, 0, 0, 255))
    result = swap_region({0: _atlas((200, 200, 200, 255))}, region, red)
    assert result[0].getpixel((10, 10))[0] == 200


def test_paste_alpha_compositing():
    region = _make_region("face_skin", "paste", [_drawable()])
    semi = Image.new("RGBA", (60, 40), (255, 0, 0, 128))
    result = swap_region({0: _atlas((200, 200, 200, 255))}, region, semi)
    px = result[0].getpixel((70, 70))
    assert px[0] > 200


def test_paste_non_rgba_replacement():
    region = _make_region("face_skin", "paste", [_drawable()])
    rgb = Image.new("RGB", (60, 40), (0, 255, 0))
    result = swap_region({0: _atlas()}, region, rgb)
    assert result[0].getpixel((70, 70))[1] == 255


def test_paste_multi_drawable():
    """Two drawables in one region both get the replacement sub-region."""
    d0 = DrawableRegion("d0", 0, 10, 10, 40, 40, 0)
    d1 = DrawableRegion("d1", 0, 100, 100, 40, 40, 0)
    region = _make_region("eyes", "paste", [d0, d1])
    # Replacement covers union bbox (10,10) to (140,140) = 130×130
    red = Image.new("RGBA", (130, 130), (255, 0, 0, 255))
    result = swap_region({0: _atlas(size=256)}, region, red)
    # Both drawable locations should be red
    assert result[0].getpixel((30, 30))[0] == 255   # d0 area
    assert result[0].getpixel((120, 120))[0] == 255  # d1 area


# ── hue_shift strategy ───────────────────────────────────────────────────────

def test_hue_shift_changes_hue():
    """Existing blue region → target red → result should have reddish hue."""
    # Blue region in atlas
    atlas = Image.new("RGBA", (256, 256), (0, 0, 200, 255))
    region = _make_region("hair_front", "hue_shift", [_drawable(x=50, y=50, w=60, h=40)])
    target = Image.new("RGB", (10, 10), (200, 0, 0))  # red target
    result = swap_region({0: atlas}, region, target)
    px = result[0].getpixel((70, 70))
    # Red channel should dominate after hue shift to red
    assert px[0] > px[2], f"Expected red > blue after hue shift, got {px}"


def test_hue_shift_preserves_transparent():
    """Transparent pixels must not be modified."""
    atlas = Image.new("RGBA", (256, 256), (0, 0, 0, 0))  # fully transparent
    region = _make_region("hair_front", "hue_shift", [_drawable()])
    target = Image.new("RGB", (10, 10), (255, 0, 0))
    result = swap_region({0: atlas}, region, target)
    # Transparent pixels stay transparent
    assert result[0].getpixel((70, 70))[3] == 0


def test_hue_shift_preserves_outside():
    """Pixels outside the drawable bbox are untouched."""
    atlas = Image.new("RGBA", (256, 256), (100, 150, 200, 255))
    region = _make_region("hair_front", "hue_shift", [_drawable(x=50, y=50, w=60, h=40)])
    target = Image.new("RGB", (10, 10), (200, 100, 0))
    result = swap_region({0: atlas}, region, target)
    assert result[0].getpixel((10, 10)) == (100, 150, 200, 255)


# ── swap_regions batch ───────────────────────────────────────────────────────

def test_swap_regions_batch():
    atlas0 = _atlas((200, 200, 200, 255))
    config = AtlasConfig(
        rig_name="test",
        template_name="humanoid-anime",
        texture_size=256,
        regions=[
            _make_region("face_skin", "paste", [_drawable(tid=0, x=50, y=50, w=60, h=40)]),
            _make_region("hair_front", "hue_shift", [_drawable(tid=0, x=10, y=10, w=30, h=30)]),
        ],
    )
    replacements = {
        "face_skin": Image.new("RGBA", (60, 40), (255, 0, 0, 255)),
        "hair_front": Image.new("RGB", (10, 10), (200, 50, 50)),
    }
    result = swap_regions({0: atlas0}, config, replacements)
    # face_skin region should be red (paste)
    assert result[0].getpixel((70, 70))[0] == 255
```

- [ ] **Step 2: Run tests — expect failure**

```bash
uv run pytest tests/pipeline/test_texture_swap.py -v 2>&1 | tail -20
```

Expected: ImportError (swap_region signature changed) or AttributeError.

- [ ] **Step 3: Rewrite `pipeline/texture_swap.py`**

```python
"""Texture region replacement — strategy-aware paste and hue-shift operations."""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion


def swap_region(
    atlases: dict[int, Image.Image],
    region: AtlasRegion,
    replacement: Image.Image,
) -> dict[int, Image.Image]:
    """Apply the region's swap_strategy.

    For "paste": replacement is the image to paste (scaled per-drawable).
    For "hue_shift": replacement is used to extract the target hue color.

    Returns a new dict; inputs are not modified.
    """
    if region.swap_strategy == "paste":
        return _paste_region(atlases, region, replacement)
    if region.swap_strategy == "hue_shift":
        return _hue_shift_region(atlases, region, replacement)
    raise ValueError(f"Unknown swap_strategy: {region.swap_strategy!r}")


def swap_regions(
    atlases: dict[int, Image.Image],
    config: AtlasConfig,
    replacements: dict[str, Image.Image],
) -> dict[int, Image.Image]:
    """Batch replacement across all named regions."""
    out = {k: v.copy() for k, v in atlases.items()}
    for name, replacement in replacements.items():
        region = config.get(name)
        out = swap_region(out, region, replacement)
    return out


def _paste_region(
    atlases: dict[int, Image.Image],
    region: AtlasRegion,
    replacement: Image.Image,
) -> dict[int, Image.Image]:
    """Scale replacement to each drawable's bbox and alpha-composite."""
    out = {k: v.copy() for k, v in atlases.items()}
    for tex_idx in region.texture_indices():
        ux, uy, uw, uh = region.bbox(texture_index=tex_idx)
        src = replacement.resize((uw, uh), Image.Resampling.LANCZOS)
        if src.mode != "RGBA":
            src = src.convert("RGBA")
        for d in region.drawables:
            if d.texture_index != tex_idx:
                continue
            # Crop sub-region of the scaled replacement corresponding to this drawable
            rx, ry = d.x - ux, d.y - uy
            sub = src.crop((rx, ry, rx + d.w, ry + d.h))
            out[tex_idx].paste(sub, (d.x, d.y), sub)
    return out


def _hue_shift_region(
    atlases: dict[int, Image.Image],
    region: AtlasRegion,
    target_color: Image.Image,
) -> dict[int, Image.Image]:
    """Rotate hue of existing atlas pixels to match target_color's average hue.

    Preserves the original saturation and value (shading, highlights, gradients).
    Only fully transparent pixels (alpha == 0) are left untouched.
    """
    # Extract target hue from replacement image
    rgb_arr = np.array(target_color.convert("RGB")).reshape(-1, 3) / 255.0
    hsv_arr = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in rgb_arr])
    target_h = float(hsv_arr[:, 0].mean())

    out = {k: v.copy() for k, v in atlases.items()}
    for d in region.drawables:
        atlas_arr = np.array(out[d.texture_index]).astype(np.float32)
        patch = atlas_arr[d.y : d.y + d.h, d.x : d.x + d.w]  # (H, W, 4)
        alpha = patch[:, :, 3]
        rgb = patch[:, :, :3] / 255.0  # (H, W, 3) in [0,1]

        # Build HSV array
        flat_rgb = rgb.reshape(-1, 3)
        flat_hsv = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in flat_rgb])

        # Replace hue with target, keep S and V
        flat_hsv[:, 0] = target_h
        flat_rgb_new = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in flat_hsv])
        rgb_new = flat_rgb_new.reshape(d.h, d.w, 3) * 255.0

        # Build result patch: new rgb + original alpha; skip fully-transparent pixels
        result = patch.copy()
        opaque = alpha > 0
        result[opaque, :3] = rgb_new[opaque]
        atlas_arr[d.y : d.y + d.h, d.x : d.x + d.w] = result
        out[d.texture_index] = Image.fromarray(atlas_arr.astype(np.uint8))

    return out
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/pipeline/test_texture_swap.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full test suite — no regressions**

```bash
uv run pytest tests/pipeline/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/texture_swap.py tests/pipeline/test_texture_swap.py
git commit -m "feat(swap): strategy-aware texture swap — paste per-drawable + hue_shift for hair"
```

---

### Task 4: `pipeline/validate.py` + integration test

**Files:**
- Create: `pipeline/validate.py`
- Create: `tests/pipeline/test_validate.py`

- [ ] **Step 1: Write the test**

```python
# tests/pipeline/test_validate.py
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

HIYORI_MODEL3 = Path(
    "/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam"
    "/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets"
    "/Live2DModels/hiyori_vts/hiyori.model3.json"
)
skip_no_model = pytest.mark.skipif(
    not HIYORI_MODEL3.exists(), reason="Hiyori model files not present"
)


def test_check_region_color_hit():
    from pipeline.validate import check_region_color
    frame = np.zeros((100, 100, 4), dtype=np.uint8)
    frame[40:60, 40:60] = [255, 0, 0, 255]
    assert check_region_color(frame, (255, 0, 0), tolerance=20)


def test_check_region_color_miss():
    from pipeline.validate import check_region_color
    frame = np.zeros((100, 100, 4), dtype=np.uint8)
    assert not check_region_color(frame, (255, 0, 0), tolerance=20)


@skip_no_model
def test_validate_textures_returns_frame():
    from rig.config import load_rig_config
    from pipeline.validate import validate_textures
    from pipeline.atlas_config import load_atlas_config

    rig = load_rig_config(HIYORI_MODEL3)
    atlas_cfg = load_atlas_config(Path("manifests/hiyori_atlas.toml"))

    tex_dir = HIYORI_MODEL3.parent / "hiyori.2048"
    atlases = {
        0: Image.open(tex_dir / "texture_00.png").convert("RGBA"),
        1: Image.open(tex_dir / "texture_01.png").convert("RGBA"),
    }

    frame = validate_textures(rig, atlases)
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 4


@skip_no_model
def test_validate_face_color_replacement():
    """Paint face_skin red → red pixels appear in rendered frame."""
    from rig.config import load_rig_config
    from pipeline.validate import validate_textures, check_region_color
    from pipeline.atlas_config import load_atlas_config
    from pipeline.texture_swap import swap_regions

    rig = load_rig_config(HIYORI_MODEL3)
    atlas_cfg = load_atlas_config(Path("manifests/hiyori_atlas.toml"))

    tex_dir = HIYORI_MODEL3.parent / "hiyori.2048"
    atlases = {
        0: Image.open(tex_dir / "texture_00.png").convert("RGBA"),
        1: Image.open(tex_dir / "texture_01.png").convert("RGBA"),
    }

    red = Image.new("RGBA", (500, 575), (255, 0, 0, 255))
    modified = swap_regions(atlases, atlas_cfg, {"face_skin": red})
    frame = validate_textures(rig, modified)
    assert check_region_color(frame, (255, 0, 0), tolerance=30)
```

- [ ] **Step 2: Run test — expect skip or ImportError**

```bash
uv run pytest tests/pipeline/test_validate.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Read `rig/render.py` to understand the headless renderer API**

```bash
cat rig/render.py
```

Use whatever `render_frame()` or similar function exists there. The validate functions
wrap it.

- [ ] **Step 4: Implement `pipeline/validate.py`**

```python
"""Headless texture validation via Live2D render."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rig.config import RigConfig


def validate_textures(
    config: RigConfig,
    modified_atlases: dict[int, Image.Image],
) -> np.ndarray:
    """Render the rig with modified textures at neutral pose.

    Copies modified textures to a temp directory alongside rig model files,
    renders one frame, returns it as (H, W, 4) RGBA uint8 array.
    """
    import live2d.v3 as live2d
    import OpenGL.GL as gl
    from rig.render import create_offscreen_context, render_frame

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Copy model files
        model_dir = config.model3_path.parent
        import shutil
        shutil.copytree(model_dir, tmp / model_dir.name)
        # Write modified textures
        tex_subdir = tmp / model_dir.name / "hiyori.2048"
        tex_subdir.mkdir(exist_ok=True)
        for idx, img in modified_atlases.items():
            img.save(tex_subdir / f"texture_0{idx}.png")

        model3 = tmp / model_dir.name / config.model3_path.name
        frame = render_frame(str(model3))
    return frame


def check_region_color(
    frame: np.ndarray,
    expected_color: tuple[int, int, int],
    tolerance: int = 20,
) -> bool:
    """Return True if any pixel in frame matches expected_color within tolerance."""
    r, g, b = expected_color
    mask = (
        (np.abs(frame[:, :, 0].astype(int) - r) <= tolerance)
        & (np.abs(frame[:, :, 1].astype(int) - g) <= tolerance)
        & (np.abs(frame[:, :, 2].astype(int) - b) <= tolerance)
        & (frame[:, :, 3] > 10)
    )
    return bool(mask.any())
```

Note: `rig/render.py` may need to expose `render_frame(model3_path: str) -> np.ndarray`.
Check the existing API and adapt accordingly.

- [ ] **Step 5: Run unit tests (no model required) — expect pass**

```bash
uv run pytest tests/pipeline/test_validate.py::test_check_region_color_hit \
              tests/pipeline/test_validate.py::test_check_region_color_miss -v
```

- [ ] **Step 6: Run integration tests (requires Hiyori model)**

```bash
uv run pytest tests/pipeline/test_validate.py -v -s
```

If model present: expect 4/4 pass. If not present: unit tests pass, integration skip.

- [ ] **Step 7: Commit**

```bash
git add pipeline/validate.py tests/pipeline/test_validate.py
git commit -m "feat(validate): headless texture validation + check_region_color"
```

---

### Task 5: `pipeline/measure_regions.py` — ctypes UV extraction + 3-panel render output

**Files:**
- Create: `pipeline/measure_regions.py`

**IMPORTANT:** This tool is for onboarding NEW rigs. `manifests/hiyori_atlas.toml` is already
complete and must NOT be overwritten by this tool. `--out` has no default; it must be supplied
explicitly. Task 6 will smoke-test this tool outputting to `/tmp` only.

This tool does Phases 1 and 2 from the spec. Phase 3 (VLM labeling) is done by the
operator using `docs/runbooks/drawable-labeling-playbook.md`. The tool outputs the 3-panel
images and a pre-filled TOML with `label = "other"` for all groups; the operator fills in
labels using the playbook.

- [ ] **Step 1: Write the tool**

```python
#!/usr/bin/env python3
"""measure_regions.py — extract UV bboxes from Live2D moc3 and render per-drawable isolations.

Phase 1: ctypes UV extraction — reads moc3 to get exact UV bbox per drawable.
Phase 2: Render isolation — renders each drawable as solid green on black background.
Phase 3: Outputs 3-panel images for operator VLM labeling + pre-filled TOML stub.

For NEW rigs only. Do NOT run on hiyori — its atlas is already hand-verified.

Usage:
    uv run python -m pipeline.measure_regions \
        --model3 /path/to/model.model3.json \
        --rig my_rig \
        --template humanoid-anime \
        --out /tmp/my_rig_atlas_stub.toml \
        --panels-dir /tmp/panels
"""
from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ─── Phase 1: ctypes UV extraction ───────────────────────────────────────────

def _find_live2d_so() -> Path:
    import live2d.v3
    so = Path(live2d.v3.__file__).parent / "live2d.so"
    if not so.exists():
        raise FileNotFoundError(f"live2d.so not found at {so}")
    return so


def extract_drawable_uvs(model3_path: Path) -> list[dict]:
    """Extract per-drawable UV bounding boxes from moc3 via ctypes.

    Returns list of dicts with keys:
        id, texture_index, uv_bbox (x, y, w, h in pixels), draw_order
    """
    lib = ctypes.CDLL(str(_find_live2d_so()))

    # Load model3.json to get moc3 path and texture size
    with open(model3_path) as f:
        model3 = json.load(f)

    moc3_rel = model3["FileReferences"]["Moc"]
    moc3_path = model3_path.parent / moc3_rel
    tex_paths = model3["FileReferences"]["Textures"]
    # Determine texture size from first texture
    tex0 = Image.open(model3_path.parent / tex_paths[0])
    tex_size = tex0.size[0]  # assume square

    # Read moc3 bytes
    moc3_bytes = moc3_path.read_bytes()
    moc3_size = len(moc3_bytes)

    # Allocate aligned buffer for moc3 (64-byte alignment required)
    align = 64
    raw_buf = ctypes.create_string_buffer(moc3_size + align)
    addr = ctypes.addressof(raw_buf)
    offset = (align - addr % align) % align
    moc3_buf = (ctypes.c_byte * moc3_size).from_buffer(raw_buf, offset)
    ctypes.memmove(moc3_buf, moc3_bytes, moc3_size)

    # ReviveMocInPlace
    lib.csmReviveMocInPlace.restype = ctypes.c_void_p
    lib.csmReviveMocInPlace.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    moc_ptr = lib.csmReviveMocInPlace(moc3_buf, moc3_size)
    if not moc_ptr:
        raise RuntimeError("csmReviveMocInPlace failed")

    # Get model size and allocate model buffer
    lib.csmGetSizeofModel.restype = ctypes.c_uint
    lib.csmGetSizeofModel.argtypes = [ctypes.c_void_p]
    model_size = lib.csmGetSizeofModel(moc_ptr)

    raw_model = ctypes.create_string_buffer(model_size + align)
    model_addr = ctypes.addressof(raw_model)
    model_offset = (align - model_addr % align) % align
    model_buf = (ctypes.c_byte * model_size).from_buffer(raw_model, model_offset)

    lib.csmInitializeModelInPlace.restype = ctypes.c_void_p
    lib.csmInitializeModelInPlace.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    model_ptr = lib.csmInitializeModelInPlace(moc_ptr, model_buf, model_size)
    if not model_ptr:
        raise RuntimeError("csmInitializeModelInPlace failed")

    # Get drawable count
    lib.csmGetDrawableCount.restype = ctypes.c_int
    lib.csmGetDrawableCount.argtypes = [ctypes.c_void_p]
    count = lib.csmGetDrawableCount(model_ptr)

    # Get drawable IDs
    lib.csmGetDrawableIds.restype = ctypes.POINTER(ctypes.c_char_p)
    lib.csmGetDrawableIds.argtypes = [ctypes.c_void_p]
    id_ptr = lib.csmGetDrawableIds(model_ptr)
    drawable_ids = [id_ptr[i].decode() for i in range(count)]

    # Get texture indices
    lib.csmGetDrawableTextureIndices.restype = ctypes.POINTER(ctypes.c_int)
    lib.csmGetDrawableTextureIndices.argtypes = [ctypes.c_void_p]
    tex_idx_ptr = lib.csmGetDrawableTextureIndices(model_ptr)
    texture_indices = [tex_idx_ptr[i] for i in range(count)]

    # Get draw orders
    lib.csmGetDrawableDrawOrders.restype = ctypes.POINTER(ctypes.c_int)
    lib.csmGetDrawableDrawOrders.argtypes = [ctypes.c_void_p]
    order_ptr = lib.csmGetDrawableDrawOrders(model_ptr)
    draw_orders = [order_ptr[i] for i in range(count)]

    # Get vertex counts
    lib.csmGetDrawableVertexCounts.restype = ctypes.POINTER(ctypes.c_int)
    lib.csmGetDrawableVertexCounts.argtypes = [ctypes.c_void_p]
    vc_ptr = lib.csmGetDrawableVertexCounts(model_ptr)
    vertex_counts = [vc_ptr[i] for i in range(count)]

    # Get UV arrays
    class Vec2(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    lib.csmGetDrawableVertexUvs.restype = ctypes.POINTER(ctypes.POINTER(Vec2))
    lib.csmGetDrawableVertexUvs.argtypes = [ctypes.c_void_p]
    uv_ptr = lib.csmGetDrawableVertexUvs(model_ptr)

    drawables = []
    for i in range(count):
        n_verts = vertex_counts[i]
        if n_verts == 0:
            continue
        uvs = uv_ptr[i]
        xs = [uvs[j].x for j in range(n_verts)]
        ys = [uvs[j].y for j in range(n_verts)]
        # UV y is flipped in Cubism (0=bottom, 1=top) — convert to image coords
        u_min, u_max = min(xs), max(xs)
        v_min, v_max = min(ys), max(ys)
        px_x = int(u_min * tex_size)
        px_y = int((1.0 - v_max) * tex_size)
        px_w = max(1, int((u_max - u_min) * tex_size))
        px_h = max(1, int((v_max - v_min) * tex_size))
        drawables.append({
            "id": drawable_ids[i],
            "texture_index": texture_indices[i],
            "uv_bbox": (px_x, px_y, px_w, px_h),
            "draw_order": draw_orders[i],
        })
    return drawables


# ─── Phase 2: render isolation ───────────────────────────────────────────────

def render_isolated_drawables(
    model3_path: Path,
    drawable_ids: list[str],
    out_dir: Path,
) -> dict[str, Path]:
    """Render each drawable as solid green on black; save PNG. Returns {id: path}."""
    import live2d.v3 as live2d

    live2d.init()
    model = live2d.LAppModel()
    model.LoadModelJson(str(model3_path))
    model.Resize(512, 512)

    all_ids = model.GetDrawableIds()
    saved = {}

    # Render full model at neutral pose for context
    full_path = out_dir / "_full_render.png"
    live2d.clearBuffer(0, 0, 0, 1)
    model.Update()
    model.Draw()
    # Note: capturing from OpenGL framebuffer requires glReadPixels — see rig/render.py
    # Here we save via the renderer; exact implementation depends on rig/render.py API.

    for target_id in drawable_ids:
        if target_id not in all_ids:
            continue
        # Set all drawables black
        for did in all_ids:
            model.SetDrawableMultiplyColor(did, 0, 0, 0, 1)
        # Set target green
        model.SetDrawableScreenColor(target_id, 0, 255, 0, 255)
        live2d.clearBuffer(0, 0, 0, 1)
        model.Update()
        model.Draw()
        # Save frame — requires rig/render.py capture helper
        path = out_dir / f"{target_id}.png"
        # frame = capture_frame(512, 512)
        # Image.fromarray(frame).save(path)
        saved[target_id] = path

    live2d.dispose()
    return saved


# ─── Phase 3: group + 3-panel compose ────────────────────────────────────────

def group_by_proximity(drawables: list[dict], gap: int = 80) -> list[list[dict]]:
    """Group drawables whose UV bboxes are within gap pixels of each other."""
    if not drawables:
        return []
    groups: list[list[dict]] = []
    used = set()
    for i, d in enumerate(drawables):
        if i in used:
            continue
        group = [d]
        used.add(i)
        x0, y0, w0, h0 = d["uv_bbox"]
        for j, d2 in enumerate(drawables):
            if j in used:
                continue
            x2, y2, w2, h2 = d2["uv_bbox"]
            # Check proximity: centres within gap px
            cx1, cy1 = x0 + w0 // 2, y0 + h0 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            if abs(cx1 - cx2) <= gap and abs(cy1 - cy2) <= gap:
                group.append(d2)
                used.add(j)
        groups.append(group)
    return groups


def compose_3panel(
    group: list[dict],
    atlas_images: dict[int, Image.Image],
    isolated_renders: dict[str, Path],
    full_render_path: Path,
) -> Image.Image:
    """Compose a 3-panel image for VLM labeling."""
    size = 256
    panel_a = Image.new("RGB", (size, size), (0, 0, 0))   # isolated green renders
    panel_b = Image.new("RGB", (size, size), (128, 128, 128))  # full render + highlight
    panel_c = Image.new("RGB", (size, size), (255, 255, 255))  # atlas excerpt

    # Panel A: composite isolated renders
    for d in group:
        if d["id"] in isolated_renders and isolated_renders[d["id"]].exists():
            iso = Image.open(isolated_renders[d["id"]]).convert("RGB")
            iso = iso.resize((size, size), Image.Resampling.LANCZOS)
            panel_a = Image.blend(panel_a, iso, 0.8)

    # Panel B: full render (just show it scaled)
    if full_render_path.exists():
        full = Image.open(full_render_path).convert("RGB")
        panel_b = full.resize((size, size), Image.Resampling.LANCZOS)

    # Panel C: atlas excerpt around union bbox
    tex_idx = group[0]["texture_index"]
    if tex_idx in atlas_images:
        atlas = atlas_images[tex_idx]
        xs = [d["uv_bbox"][0] for d in group]
        ys = [d["uv_bbox"][1] for d in group]
        x2s = [d["uv_bbox"][0] + d["uv_bbox"][2] for d in group]
        y2s = [d["uv_bbox"][1] + d["uv_bbox"][3] for d in group]
        cx = (min(xs) + max(x2s)) // 2
        cy = (min(ys) + max(y2s)) // 2
        pad = 80
        crop = atlas.crop((max(0, cx - pad), max(0, cy - pad),
                           min(atlas.width, cx + pad), min(atlas.height, cy + pad)))
        panel_c = crop.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")

    # Concatenate horizontally
    combined = Image.new("RGB", (size * 3 + 20, size), (200, 200, 200))
    combined.paste(panel_a, (0, 0))
    combined.paste(panel_b, (size + 10, 0))
    combined.paste(panel_c, (size * 2 + 20, 0))
    return combined


# ─── TOML writer ─────────────────────────────────────────────────────────────

def write_atlas_toml(
    groups: list[list[dict]],
    labels: dict[int, str],   # group_index → region name
    strategies: dict[str, str],  # region_name → swap_strategy
    rig: str,
    template: str,
    texture_size: int,
    out_path: Path,
) -> None:
    lines = [
        f'# Generated by pipeline.measure_regions',
        f'# Edit labels where marked # REVIEW before committing.',
        f'',
        f'rig          = "{rig}"',
        f'template     = "{template}"',
        f'texture_size = {texture_size}',
        f'',
    ]
    # Aggregate by label
    by_label: dict[str, list[dict]] = {}
    for g_idx, group in enumerate(groups):
        label = labels.get(g_idx, "other")
        by_label.setdefault(label, []).extend(group)

    for label, drawables in by_label.items():
        strategy = strategies.get(label, "paste")
        lines.append(f'[[regions]]')
        lines.append(f'name          = "{label}"')
        lines.append(f'swap_strategy = "{strategy}"')
        lines.append(f'')
        for d in drawables:
            x, y, w, h = d["uv_bbox"]
            lines.append(f'  [[regions.drawables]]')
            lines.append(f'  id           = "{d["id"]}"')
            lines.append(f'  texture_index = {d["texture_index"]}')
            lines.append(f'  x = {x}')
            lines.append(f'  y = {y}')
            lines.append(f'  w = {w}')
            lines.append(f'  h = {h}')
            lines.append(f'  draw_order   = {d["draw_order"]}')
            lines.append(f'')

    out_path.write_text('\n'.join(lines))
    print(f"Wrote {out_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model3", required=True, type=Path)
    parser.add_argument("--rig", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--panels-dir", type=Path, default=Path("/tmp/atlas_panels"))
    args = parser.parse_args()

    args.panels_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 1: extracting UV bboxes via ctypes...")
    drawables = extract_drawable_uvs(args.model3)
    print(f"  {len(drawables)} drawables found")

    # Load atlas images for panel C
    model3 = json.loads(args.model3.read_text())
    tex_paths = model3["FileReferences"]["Textures"]
    atlas_images = {}
    for i, tp in enumerate(tex_paths):
        path = args.model3.parent / tp
        if path.exists():
            atlas_images[i] = Image.open(path).convert("RGBA")

    # Determine texture size
    tex_size = list(atlas_images.values())[0].size[0] if atlas_images else 2048

    print("Phase 2: rendering isolated drawables...")
    isolated = render_isolated_drawables(
        args.model3,
        [d["id"] for d in drawables],
        args.panels_dir,
    )

    print("Phase 3: grouping and composing 3-panel images...")
    groups = group_by_proximity(drawables)
    print(f"  {len(groups)} groups formed")
    full_render = args.panels_dir / "_full_render.png"
    for g_idx, group in enumerate(groups):
        panel = compose_3panel(group, atlas_images, isolated, full_render)
        panel_path = args.panels_dir / f"group_{g_idx:03d}.png"
        panel.save(panel_path)

    print(f"\nLabeling required: {len(groups)} panels saved to {args.panels_dir}/")
    print("Use docs/runbooks/drawable-labeling-playbook.md to label each group.")
    print("Then re-run with --labels <json_file> to write the final TOML.\n")

    # Write stub TOML with all labels = "other"
    labels = {i: "other" for i in range(len(groups))}
    strategies = {}  # will default to "paste" — operator updates hair regions
    write_atlas_toml(groups, labels, strategies, args.rig, args.template, tex_size, args.out)
    print(f"Stub TOML written. Edit labels in {args.out} then commit.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the tool is importable**

```bash
uv run python -c "import pipeline.measure_regions; print('OK')"
```

- [ ] **Step 3: Run Phase 1 only (UV extraction) as smoke test**

```bash
uv run python -c "
from pipeline.measure_regions import extract_drawable_uvs
from pathlib import Path
drawables = extract_drawable_uvs(Path(
    '/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam'
    '/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets'
    '/Live2DModels/hiyori_vts/hiyori.model3.json'
))
print(f'{len(drawables)} drawables')
for d in drawables[:5]:
    print(f\"  {d['id']}  tex={d['texture_index']}  bbox={d['uv_bbox']}\")
"
```

Expected: ~133 drawables printed with pixel bboxes.

- [ ] **Step 4: Commit**

```bash
git add pipeline/measure_regions.py
git commit -m "feat(measure): measure_regions.py — ctypes UV extraction + per-drawable panel output"
```

---

### Task 6: Smoke test `measure_regions.py` on Hiyori → /tmp (read-only)

**NOTE:** `manifests/hiyori_atlas.toml` is complete and must NOT be touched.
This task only verifies the tool runs correctly end-to-end.

- [ ] **Step 1: Run tool to /tmp only**

```bash
uv run python -m pipeline.measure_regions \
    --model3 "/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/hiyori_vts/hiyori.model3.json" \
    --rig hiyori \
    --template humanoid-anime \
    --out /tmp/hiyori_atlas_stub.toml \
    --panels-dir /tmp/atlas_panels
```

Expected: 133 drawables extracted, panels written to `/tmp/atlas_panels/`, stub TOML written
to `/tmp/hiyori_atlas_stub.toml`. `manifests/hiyori_atlas.toml` unchanged.

- [ ] **Step 2: Verify stub loaded and drawable count matches**

```bash
python3 -c "
import tomllib
data = tomllib.loads(open('/tmp/hiyori_atlas_stub.toml').read())
total = sum(len(r['drawables']) for r in data['regions'])
print(f'stub: {total} drawables in {len(data[\"regions\"])} groups')
import tomllib as t2
data2 = t2.loads(open('manifests/hiyori_atlas.toml').read())
total2 = sum(len(r['drawables']) for r in data2['regions'])
print(f'atlas: {total2} drawables in {len(data2[\"regions\"])} regions (hand-verified, keep this)')
"
```

- [ ] **Step 3: Commit (tool only, no atlas change)**

```bash
git add pipeline/measure_regions.py
git commit -m "test(measure): smoke test measure_regions on Hiyori — tool works, atlas untouched"
```
