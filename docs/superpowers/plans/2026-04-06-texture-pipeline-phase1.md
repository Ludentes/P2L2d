# Texture Pipeline Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement atlas coordinate config, texture region swap, headless validation, and a UV-extraction measure tool — the mechanical foundation for texture personalization.

**Architecture:** Atlas config (TOML per rig) maps canonical region names to pixel bounding boxes. `texture_swap.py` pastes replacement images at those coordinates. `validate.py` headlessly renders the result using RigRenderer. `measure_regions.py` extracts UV data from the moc3 via the Cubism Core C API (ctypes) and displays bboxes overlaid on the texture atlas so the user can label semantic regions once.

**Tech Stack:** Python 3.12, tomllib (stdlib), Pillow, matplotlib (measure tool only), ctypes, existing `rig/render.py` + `rig/config.py`, live2d-py Cubism Core functions (csmGetDrawableVertexUvs, csmGetDrawableTextureIndices).

**Spec:** `docs/superpowers/specs/2026-04-06-texture-pipeline-phase1-design.md`

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `pipeline/__init__.py` | Create | Package marker |
| `pipeline/atlas_config.py` | Create | `AtlasRegion`, `AtlasConfig`, `load_atlas_config()` |
| `pipeline/texture_swap.py` | Create | `swap_region()`, `swap_regions()` |
| `pipeline/validate.py` | Create | `validate_textures()`, `check_region_color()` |
| `pipeline/measure_regions.py` | Create | One-time UV extractor + labeling tool |
| `templates/humanoid-anime/atlas_schema.toml` | Create | Canonical region name reference |
| `manifests/hiyori_atlas.toml` | Create | Hiyori-specific pixel coordinates (run measure_regions.py to fill) |
| `tests/pipeline/__init__.py` | Create | Test package marker |
| `tests/pipeline/test_atlas_config.py` | Create | Unit tests for atlas config |
| `tests/pipeline/test_texture_swap.py` | Create | Unit tests for texture swap |
| `tests/pipeline/test_validate.py` | Create | Integration tests (skip if no model) |

---

## Task 1: AtlasConfig data types + loader

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/atlas_config.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_atlas_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/__init__.py` (empty) and `tests/pipeline/test_atlas_config.py`:

```python
from pathlib import Path
import pytest
from pipeline.atlas_config import AtlasRegion, AtlasConfig, load_atlas_config

_SAMPLE_TOML = """\
rig = "test_rig"
template = "humanoid-anime"
texture_size = 2048

[[regions]]
name = "face_skin"
texture_index = 0
x = 400
y = 100
w = 280
h = 320

[[regions]]
name = "left_eye"
texture_index = 0
x = 450
y = 150
w = 80
h = 60
"""

@pytest.fixture
def tmp_atlas(tmp_path):
    p = tmp_path / "test.toml"
    p.write_text(_SAMPLE_TOML)
    return load_atlas_config(p)


def test_load_atlas_config(tmp_atlas):
    assert tmp_atlas.rig_name == "test_rig"
    assert tmp_atlas.template_name == "humanoid-anime"
    assert tmp_atlas.texture_size == 2048
    assert len(tmp_atlas.regions) == 2


def test_atlas_config_get(tmp_atlas):
    r = tmp_atlas.get("face_skin")
    assert r.texture_index == 0
    assert r.x == 400
    assert r.y == 100
    assert r.w == 280
    assert r.h == 320


def test_atlas_config_get_missing(tmp_atlas):
    with pytest.raises(KeyError):
        tmp_atlas.get("nonexistent")


def test_atlas_config_has(tmp_atlas):
    assert tmp_atlas.has("face_skin")
    assert not tmp_atlas.has("nonexistent")


def test_load_atlas_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_atlas_config(Path("nonexistent_atlas.toml"))
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
uv run pytest tests/pipeline/test_atlas_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Create `pipeline/__init__.py` (empty)**

```bash
touch pipeline/__init__.py tests/pipeline/__init__.py
```

- [ ] **Step 4: Write `pipeline/atlas_config.py`**

```python
"""Atlas coordinate config — maps canonical region names to pixel bounding boxes.

Each rig has a corresponding manifests/<rig>_atlas.toml. This file never changes
for a given rig unless the texture atlas layout changes.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AtlasRegion:
    name: str           # canonical name, e.g. "face_skin"
    texture_index: int  # 0-based index into RigConfig.textures
    x: int              # left edge in pixels
    y: int              # top edge in pixels
    w: int              # width in pixels
    h: int              # height in pixels


@dataclass
class AtlasConfig:
    rig_name: str
    template_name: str
    texture_size: int           # square texture side length (e.g. 2048)
    regions: list[AtlasRegion]

    def get(self, name: str) -> AtlasRegion:
        """Return the region with this name. Raises KeyError if not found."""
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"Region {name!r} not found in atlas config for {self.rig_name!r}")

    def has(self, name: str) -> bool:
        return any(r.name == name for r in self.regions)


def load_atlas_config(path: Path) -> AtlasConfig:
    """Load atlas config from a TOML file. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Atlas config not found: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    regions = [
        AtlasRegion(
            name=r["name"],
            texture_index=r["texture_index"],
            x=r["x"],
            y=r["y"],
            w=r["w"],
            h=r["h"],
        )
        for r in data.get("regions", [])
    ]
    return AtlasConfig(
        rig_name=data["rig"],
        template_name=data["template"],
        texture_size=data["texture_size"],
        regions=regions,
    )
```

- [ ] **Step 5: Run tests — confirm they pass**

```bash
uv run pytest tests/pipeline/test_atlas_config.py -v
```
Expected: 5 PASSED

- [ ] **Step 6: Commit**

```bash
git add pipeline/__init__.py pipeline/atlas_config.py tests/pipeline/__init__.py tests/pipeline/test_atlas_config.py
git commit -m "feat(pipeline): AtlasConfig data types and TOML loader"
```

---

## Task 2: atlas_schema.toml + hiyori_atlas.toml stub

**Files:**
- Create: `templates/humanoid-anime/atlas_schema.toml`
- Create: `manifests/hiyori_atlas.toml`

No tests for these — they're data files, not code.

- [ ] **Step 1: Create `templates/humanoid-anime/atlas_schema.toml`**

This is a reference file for riggers and future automation. Lists what regions a
humanoid-anime rig should provide.

```toml
# Canonical texture region names for humanoid-anime rigs.
# Each rig provides pixel coordinates for these in manifests/<rig>_atlas.toml.
# Regions marked required = true must be present; others are optional.

[[regions]]
name        = "face_skin"
description = "Face skin base (forehead, cheeks, chin)"
required    = true

[[regions]]
name        = "left_eye"
description = "Left eye (iris, pupil, white, lashes)"
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
description = "Mouth region (open/closed area)"
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
description = "Front hair layer (bangs, face frame)"
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

- [ ] **Step 2: Create `manifests/hiyori_atlas.toml` stub**

This will be populated by running `measure_regions.py` in Task 5. Create a stub
so the repo has the file with placeholder coordinates. Replace with real values
after running the measurement tool.

```toml
# Hiyori texture atlas region coordinates.
# Generated by: uv run python -m pipeline.measure_regions --rig hiyori
# PLACEHOLDER — replace coordinates by running measure_regions.py

rig          = "hiyori"
template     = "humanoid-anime"
texture_size = 2048

# texture_00 = face, hair, accessories
# texture_01 = body, clothing

[[regions]]
name          = "face_skin"
texture_index = 0
x = 0
y = 0
w = 100
h = 100

[[regions]]
name          = "left_eye"
texture_index = 0
x = 0
y = 0
w = 50
h = 50

[[regions]]
name          = "right_eye"
texture_index = 0
x = 0
y = 0
w = 50
h = 50

[[regions]]
name          = "left_eyebrow"
texture_index = 0
x = 0
y = 0
w = 50
h = 20

[[regions]]
name          = "right_eyebrow"
texture_index = 0
x = 0
y = 0
w = 50
h = 20

[[regions]]
name          = "mouth"
texture_index = 0
x = 0
y = 0
w = 60
h = 40

[[regions]]
name          = "hair_front"
texture_index = 0
x = 0
y = 0
w = 200
h = 200
```

- [ ] **Step 3: Commit**

```bash
git add templates/humanoid-anime/atlas_schema.toml manifests/hiyori_atlas.toml
git commit -m "feat(pipeline): atlas_schema.toml + hiyori_atlas.toml stub"
```

---

## Task 3: Texture region swap

**Files:**
- Create: `pipeline/texture_swap.py`
- Create: `tests/pipeline/test_texture_swap.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_texture_swap.py`:

```python
import pytest
from PIL import Image
from pipeline.atlas_config import AtlasRegion, AtlasConfig
from pipeline.texture_swap import swap_region, swap_regions

_REGION = AtlasRegion(name="face_skin", texture_index=0, x=50, y=50, w=60, h=40)


def _atlas(color=(200, 200, 200, 255)):
    return Image.new("RGBA", (256, 256), color)


def test_swap_region_pixels():
    red = Image.new("RGBA", (60, 40), (255, 0, 0, 255))
    result = swap_region(_atlas(), _REGION, red)
    px = result.getpixel((70, 70))  # inside region
    assert px[0] == 255, "Red channel should be 255"
    assert px[1] == 0, "Green channel should be 0"


def test_swap_preserves_outside():
    red = Image.new("RGBA", (60, 40), (255, 0, 0, 255))
    result = swap_region(_atlas(color=(200, 200, 200, 255)), _REGION, red)
    px = result.getpixel((10, 10))  # outside region
    assert px[0] == 200, "Pixel outside region should be unchanged"


def test_swap_alpha_compositing():
    # Semi-transparent red over grey — result should be between the two
    semi = Image.new("RGBA", (60, 40), (255, 0, 0, 128))
    result = swap_region(_atlas(color=(200, 200, 200, 255)), _REGION, semi)
    px = result.getpixel((70, 70))
    assert px[0] > 200, "Red channel should increase"


def test_swap_non_rgba_replacement():
    # RGB (no alpha) replacement should be treated as fully opaque
    rgb = Image.new("RGB", (60, 40), (0, 255, 0))
    result = swap_region(_atlas(), _REGION, rgb)
    px = result.getpixel((70, 70))
    assert px[1] == 255, "Green channel should be 255"


def test_swap_regions():
    atlas0 = _atlas(color=(200, 200, 200, 255))
    atlas1 = _atlas(color=(150, 150, 150, 255))
    config = AtlasConfig(
        rig_name="test",
        template_name="humanoid-anime",
        texture_size=256,
        regions=[
            _REGION,
            AtlasRegion(name="hair_front", texture_index=1, x=10, y=10, w=50, h=50),
        ],
    )
    replacements = {
        "face_skin": Image.new("RGBA", (60, 40), (255, 0, 0, 255)),
        "hair_front": Image.new("RGBA", (50, 50), (0, 0, 255, 255)),
    }
    result = swap_regions({0: atlas0, 1: atlas1}, config, replacements)
    # face_skin → atlas 0, red
    assert result[0].getpixel((70, 70))[0] == 255
    # hair_front → atlas 1, blue
    assert result[1].getpixel((30, 30))[2] == 255


def test_swap_regions_only_named_are_replaced():
    atlas0 = _atlas(color=(200, 200, 200, 255))
    config = AtlasConfig(
        rig_name="test",
        template_name="humanoid-anime",
        texture_size=256,
        regions=[_REGION],
    )
    # Only swap face_skin — other pixels unchanged
    result = swap_regions({0: atlas0}, config, {"face_skin": Image.new("RGBA", (60, 40), (255, 0, 0, 255))})
    assert result[0].getpixel((10, 10))[0] == 200
```

- [ ] **Step 2: Run — confirm fail**

```bash
uv run pytest tests/pipeline/test_texture_swap.py -v
```
Expected: `ModuleNotFoundError: No module named 'pipeline.texture_swap'`

- [ ] **Step 3: Write `pipeline/texture_swap.py`**

```python
"""Texture region replacement — paste images into a Live2D texture atlas.

All functions operate on PIL Images and AtlasConfig; no rig-specific logic.
"""
from __future__ import annotations

from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion


def swap_region(
    atlas: Image.Image,
    region: AtlasRegion,
    replacement: Image.Image,
) -> Image.Image:
    """Paste replacement into atlas at region coordinates.

    Scales replacement to region size. Alpha-composites if replacement has alpha.

    Args:
        atlas: The full texture atlas image (RGBA).
        region: Target bounding box within atlas.
        replacement: Image to paste. Resized to (region.w, region.h).

    Returns:
        New Image with replacement pasted. Input atlas is not modified.
    """
    out = atlas.copy()
    src = replacement.resize((region.w, region.h), Image.LANCZOS)
    if src.mode != "RGBA":
        src = src.convert("RGBA")
    out.paste(src, (region.x, region.y), src)
    return out


def swap_regions(
    atlases: dict[int, Image.Image],
    config: AtlasConfig,
    replacements: dict[str, Image.Image],
) -> dict[int, Image.Image]:
    """Batch region replacement across multiple texture atlas images.

    Args:
        atlases: Map of texture_index → PIL Image.
        config: Atlas config providing region coordinates.
        replacements: Map of region_name → replacement Image.

    Returns:
        New dict with modified atlas images. Inputs are not modified.
    """
    out: dict[int, Image.Image] = {k: v.copy() for k, v in atlases.items()}
    for name, replacement in replacements.items():
        region = config.get(name)
        out[region.texture_index] = swap_region(
            out[region.texture_index], region, replacement
        )
    return out
```

- [ ] **Step 4: Run — confirm pass**

```bash
uv run pytest tests/pipeline/test_texture_swap.py -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add pipeline/texture_swap.py tests/pipeline/test_texture_swap.py
git commit -m "feat(pipeline): texture region swap (swap_region, swap_regions)"
```

---

## Task 4: Headless render validation

**Files:**
- Create: `pipeline/validate.py`
- Create: `tests/pipeline/test_validate.py`

- [ ] **Step 1: Write tests**

Create `tests/pipeline/test_validate.py`:

```python
"""Integration tests for validate.py — skip if Hiyori model files not present."""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path


_HIYORI_MODEL3 = Path(
    "~/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common"
    "/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels"
    "/hiyori_vts/hiyori.model3.json"
).expanduser()


def _hiyori_available():
    return _HIYORI_MODEL3.exists()


@pytest.mark.skipif(not _hiyori_available(), reason="Hiyori model files not present")
def test_validate_textures_returns_frame():
    from rig.config import RIG_HIYORI
    from pipeline.validate import validate_textures

    frame = validate_textures(RIG_HIYORI, modified_atlases={})
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (512, 512, 4)
    assert frame.dtype == np.uint8


@pytest.mark.skipif(not _hiyori_available(), reason="Hiyori model files not present")
def test_validate_solid_color_region_visible():
    """Swap face_skin with solid red — red pixels should appear in frame."""
    from rig.config import RIG_HIYORI
    from pipeline.atlas_config import load_atlas_config, AtlasRegion
    from pipeline.validate import validate_textures, check_region_color

    # Load atlas 0 and swap a region with solid red
    tex_path = RIG_HIYORI.textures[0]
    original = Image.open(tex_path).convert("RGBA")
    # Use a simple inline region (100x100 block at top-left) — no atlas.toml needed
    from pipeline.atlas_config import AtlasRegion, AtlasConfig
    from pipeline.texture_swap import swap_regions
    
    test_region = AtlasRegion(name="test_block", texture_index=0, x=0, y=0, w=200, h=200)
    config = AtlasConfig(
        rig_name="hiyori", template_name="humanoid-anime",
        texture_size=2048, regions=[test_region],
    )
    modified = swap_regions(
        {0: original},
        config,
        {"test_block": Image.new("RGBA", (200, 200), (255, 0, 0, 255))},
    )
    frame = validate_textures(RIG_HIYORI, modified_atlases=modified)
    assert isinstance(frame, np.ndarray)


def test_check_region_color_match():
    from pipeline.validate import check_region_color

    frame = np.zeros((100, 100, 4), dtype=np.uint8)
    frame[40:60, 40:60] = [255, 0, 0, 255]  # red block
    assert check_region_color(frame, (255, 0, 0), tolerance=20)


def test_check_region_color_no_match():
    from pipeline.validate import check_region_color

    frame = np.zeros((100, 100, 4), dtype=np.uint8)
    frame[:, :] = [0, 255, 0, 255]  # all green
    assert not check_region_color(frame, (255, 0, 0), tolerance=20)
```

- [ ] **Step 2: Run — confirm correct failures**

```bash
uv run pytest tests/pipeline/test_validate.py -v
```
Expected: `test_check_region_color_*` fail with import error; skipif tests skip (or fail if model present).

- [ ] **Step 3: Write `pipeline/validate.py`**

```python
"""Headless render-based texture validation.

Copies model files to a temp directory, substitutes modified textures,
renders a neutral pose via RigRenderer, returns the frame.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rig.config import RigConfig


def validate_textures(
    config: RigConfig,
    modified_atlases: dict[int, Image.Image],
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Render model with modified textures. Returns (H, W, 4) RGBA uint8 array.

    Copies the model directory to a temp dir, writes modified atlases over the
    originals, then renders a neutral-pose frame.

    Args:
        config: RigConfig pointing to the original model files.
        modified_atlases: Map of texture_index → modified PIL Image. Unmodified
            textures are copied as-is from config.textures.
        width: Render width in pixels.
        height: Render height in pixels.
    """
    from rig.render import RigRenderer

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Copy all files from model_dir (flat — hiyori has no subdirs except texture dir)
        for item in config.model_dir.iterdir():
            if item.is_file():
                shutil.copy(item, tmp_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, tmp_path / item.name)

        # Overwrite modified atlases (preserving relative path structure)
        for idx, img in modified_atlases.items():
            rel = config.textures[idx].relative_to(config.model_dir)
            dest = tmp_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(dest))

        # Build a RigConfig pointing to the temp directory
        tmp_config = RigConfig(
            name=config.name,
            model_dir=tmp_path,
            moc3_path=tmp_path / config.moc3_path.name,
            model3_json_path=tmp_path / config.model3_json_path.name,
            textures=[tmp_path / t.relative_to(config.model_dir) for t in config.textures],
            param_ids=config.param_ids,
        )

        with RigRenderer(tmp_config, width=width, height=height) as renderer:
            return renderer.render()


def check_region_color(
    frame: np.ndarray,
    expected_color: tuple[int, int, int],
    tolerance: int = 20,
) -> bool:
    """Return True if any visible pixel in frame is within tolerance of expected_color.

    Args:
        frame: (H, W, 4) RGBA uint8 array.
        expected_color: (R, G, B) target colour.
        tolerance: per-channel absolute tolerance.
    """
    r, g, b = expected_color
    match = (
        (np.abs(frame[:, :, 0].astype(int) - r) < tolerance)
        & (np.abs(frame[:, :, 1].astype(int) - g) < tolerance)
        & (np.abs(frame[:, :, 2].astype(int) - b) < tolerance)
        & (frame[:, :, 3] > 128)  # pixel is not transparent
    )
    return bool(match.any())
```

- [ ] **Step 4: Run — confirm pass**

```bash
uv run pytest tests/pipeline/test_validate.py -v
```
Expected: `test_check_region_color_*` PASS. Integration tests SKIP (or PASS if model present).

- [ ] **Step 5: Commit**

```bash
git add pipeline/validate.py tests/pipeline/test_validate.py
git commit -m "feat(pipeline): validate_textures + check_region_color"
```

---

## Task 5: UV extraction tool (`measure_regions.py`)

**Files:**
- Create: `pipeline/measure_regions.py`

No automated tests — this is an interactive one-time tool. Validate by running it.

**How it works:** loads the moc3 binary via ctypes calling Cubism Core functions
(exported from `live2d.so`), extracts per-drawable UV bounding boxes, overlays them
on the texture atlas with matplotlib, lets the user assign canonical region names.

- [ ] **Step 1: Verify the ctypes approach works**

Run this smoke test to confirm csmGetDrawableCount works:

```bash
uv run python3 -c "
import ctypes, importlib.util
spec = importlib.util.find_spec('live2d.v3.live2d')
core = ctypes.CDLL(spec.origin)
core.csmGetDrawableCount.restype = ctypes.c_int
core.csmGetDrawableCount.argtypes = [ctypes.c_void_p]
print('csmGetDrawableCount loaded OK')
"
```
Expected: `csmGetDrawableCount loaded OK`

- [ ] **Step 2: Write `pipeline/measure_regions.py`**

```python
"""One-time UV extraction tool: moc3 → atlas coordinate config.

Loads a moc3 file, extracts UV bounding boxes per drawable via Cubism Core ctypes,
overlays them on the texture atlas (matplotlib), and outputs a TOML manifest.

Usage:
    uv run python -m pipeline.measure_regions \\
        --moc3 /path/to/hiyori.moc3 \\
        --textures /path/to/texture_00.png /path/to/texture_01.png \\
        --rig hiyori \\
        --template humanoid-anime \\
        --out manifests/hiyori_atlas.toml
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import sys
from pathlib import Path


_ALIGN = 64  # Cubism Core requires 64-byte alignment


def _aligned_buffer(data: bytes) -> ctypes.Array:
    """Copy data into a 64-byte-aligned ctypes buffer."""
    size = len(data)
    buf = ctypes.create_string_buffer(size + _ALIGN)
    offset = (_ALIGN - ctypes.addressof(buf) % _ALIGN) % _ALIGN
    dest = (ctypes.c_char * size).from_buffer(buf, offset)
    ctypes.memmove(dest, data, size)
    return dest, buf  # return both so buf stays alive


def _load_core() -> ctypes.CDLL:
    """Load live2d.so and set up Cubism Core function signatures."""
    spec = importlib.util.find_spec("live2d.v3.live2d")
    if spec is None or spec.origin is None:
        raise RuntimeError("live2d-py not installed — run: uv add live2d-py")
    core = ctypes.CDLL(spec.origin)

    vp = ctypes.c_void_p
    u32 = ctypes.c_uint
    c_int = ctypes.c_int
    c_float2 = ctypes.c_float * 2

    core.csmReviveMocInPlace.restype = vp
    core.csmReviveMocInPlace.argtypes = [vp, u32]

    core.csmGetSizeofModel.restype = u32
    core.csmGetSizeofModel.argtypes = [vp]

    core.csmInitializeModelInPlace.restype = vp
    core.csmInitializeModelInPlace.argtypes = [vp, vp, u32]

    core.csmUpdateModel.restype = None
    core.csmUpdateModel.argtypes = [vp]

    core.csmGetDrawableCount.restype = c_int
    core.csmGetDrawableCount.argtypes = [vp]

    core.csmGetDrawableIds.restype = ctypes.POINTER(ctypes.c_char_p)
    core.csmGetDrawableIds.argtypes = [vp]

    core.csmGetDrawableTextureIndices.restype = ctypes.POINTER(c_int)
    core.csmGetDrawableTextureIndices.argtypes = [vp]

    core.csmGetDrawableVertexCounts.restype = ctypes.POINTER(c_int)
    core.csmGetDrawableVertexCounts.argtypes = [vp]

    core.csmGetDrawableVertexUvs.restype = ctypes.POINTER(ctypes.POINTER(c_float2))
    core.csmGetDrawableVertexUvs.argtypes = [vp]

    return core


def extract_uv_bboxes(moc3_path: Path) -> list[dict]:
    """Return list of {id, texture_index, x, y, w, h} in pixel coords (texture_size=1 normalized)."""
    core = _load_core()
    data = moc3_path.read_bytes()

    moc_data, moc_buf = _aligned_buffer(data)
    moc = core.csmReviveMocInPlace(moc_data, len(data))
    if not moc:
        raise RuntimeError(f"csmReviveMocInPlace failed — invalid moc3: {moc3_path}")

    model_size = core.csmGetSizeofModel(moc)
    model_data, model_buf = _aligned_buffer(b"\x00" * model_size)
    model = core.csmInitializeModelInPlace(moc, model_data, model_size)
    if not model:
        raise RuntimeError("csmInitializeModelInPlace failed")
    core.csmUpdateModel(model)

    count = core.csmGetDrawableCount(model)
    ids_ptr = core.csmGetDrawableIds(model)
    tex_indices = core.csmGetDrawableTextureIndices(model)
    vtx_counts = core.csmGetDrawableVertexCounts(model)
    uvs_ptr = core.csmGetDrawableVertexUvs(model)

    drawables = []
    for i in range(count):
        raw_id = ids_ptr[i]
        drawable_id = raw_id.decode("utf-8") if raw_id else f"ArtMesh{i}"
        tex_idx = tex_indices[i]
        vtx_count = vtx_counts[i]
        if vtx_count == 0:
            continue

        uvs = uvs_ptr[i]
        us = [uvs[j][0] for j in range(vtx_count)]
        vs = [uvs[j][1] for j in range(vtx_count)]

        # UVs are in [0,1] normalized texture space. Return normalized values;
        # caller multiplies by texture_size to get pixel coords.
        drawables.append({
            "id": drawable_id,
            "texture_index": tex_idx,
            "u_min": min(us), "u_max": max(us),
            "v_min": min(vs), "v_max": max(vs),
        })

    return drawables


def show_atlas(texture_path: Path, drawables: list[dict], texture_index: int) -> None:
    """Display texture atlas with all drawable UV bounding boxes overlaid."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    img = Image.open(texture_path).convert("RGBA")
    size = img.size[0]  # assumes square

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img, origin="upper")
    ax.set_title(f"texture_{texture_index:02d} — click bboxes to assign region names")

    for d in drawables:
        if d["texture_index"] != texture_index:
            continue
        x = d["u_min"] * size
        y = d["v_min"] * size
        w = (d["u_max"] - d["u_min"]) * size
        h = (d["v_max"] - d["v_min"]) * size
        rect = patches.Rectangle((x, y), w, h, linewidth=0.5,
                                   edgecolor="lime", facecolor="none", alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, d["id"], fontsize=4, color="yellow", va="top")

    plt.tight_layout()
    plt.show()


def interactive_label(drawables: list[dict], texture_size: int) -> list[dict]:
    """Prompt user to assign canonical region names to drawable groups."""
    print("\nDrawable UV bounding boxes (texture_size={texture_size}):")
    for i, d in enumerate(drawables):
        x = int(d["u_min"] * texture_size)
        y = int(d["v_min"] * texture_size)
        w = int((d["u_max"] - d["u_min"]) * texture_size)
        h = int((d["v_max"] - d["v_min"]) * texture_size)
        print(f"  [{i:3d}] tex={d['texture_index']} id={d['id']:20s} bbox=({x},{y},{w},{h})")

    print("\nEnter region assignments as: <region_name> <drawable_indices...>")
    print("Example: face_skin 12 13 14")
    print("Type 'done' when finished.\n")

    region_groups: dict[str, list[int]] = {}
    while True:
        line = input("> ").strip()
        if line.lower() == "done":
            break
        parts = line.split()
        if len(parts) < 2:
            print("  Format: <region_name> <index> [index ...]")
            continue
        name = parts[0]
        try:
            indices = [int(p) for p in parts[1:]]
        except ValueError:
            print("  Indices must be integers")
            continue
        region_groups[name] = indices
        print(f"  Assigned {name} = {[drawables[i]['id'] for i in indices]}")

    # Build per-region bounding boxes (union of all assigned drawables)
    regions = []
    for name, indices in region_groups.items():
        group = [drawables[i] for i in indices]
        tex_idx = group[0]["texture_index"]
        u_min = min(d["u_min"] for d in group)
        u_max = max(d["u_max"] for d in group)
        v_min = min(d["v_min"] for d in group)
        v_max = max(d["v_max"] for d in group)
        regions.append({
            "name": name,
            "texture_index": tex_idx,
            "x": int(u_min * texture_size),
            "y": int(v_min * texture_size),
            "w": int((u_max - u_min) * texture_size),
            "h": int((v_max - v_min) * texture_size),
        })
    return regions


def write_toml(regions: list[dict], rig: str, template: str,
               texture_size: int, out_path: Path) -> None:
    lines = [
        f'rig          = "{rig}"',
        f'template     = "{template}"',
        f'texture_size = {texture_size}',
        "",
    ]
    for r in regions:
        lines += [
            "[[regions]]",
            f'name          = "{r["name"]}"',
            f'texture_index = {r["texture_index"]}',
            f'x = {r["x"]}',
            f'y = {r["y"]}',
            f'w = {r["w"]}',
            f'h = {r["h"]}',
            "",
        ]
    out_path.write_text("\n".join(lines))
    print(f"Written: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract UV bboxes from moc3 and label regions.")
    parser.add_argument("--moc3", required=True, type=Path)
    parser.add_argument("--textures", required=True, nargs="+", type=Path)
    parser.add_argument("--rig", required=True)
    parser.add_argument("--template", default="humanoid-anime")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--show", action="store_true", help="Show matplotlib atlas overlay")
    args = parser.parse_args()

    print(f"Extracting UV bboxes from {args.moc3} ...")
    drawables = extract_uv_bboxes(args.moc3)
    print(f"Found {len(drawables)} drawables")

    if args.show:
        for i, tex_path in enumerate(args.textures):
            show_atlas(tex_path, drawables, texture_index=i)

    regions = interactive_label(drawables, args.texture_size)
    write_toml(regions, args.rig, args.template, args.texture_size, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify ctypes loading works**

```bash
uv run python -m pipeline.measure_regions --help
```
Expected: argument help text with `--moc3`, `--textures`, etc.

- [ ] **Step 4: Commit**

```bash
git add pipeline/measure_regions.py
git commit -m "feat(pipeline): measure_regions — UV extraction from moc3 via Cubism Core ctypes"
```

---

## Task 6: Run measure_regions for Hiyori + update atlas.toml

This is a user-interactive step. Run the tool, label the Hiyori atlas, commit the result.

- [ ] **Step 1: Install matplotlib if not present**

```bash
uv add matplotlib
```

- [ ] **Step 2: Run the tool**

```bash
HIYORI=/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube\ Studio/VTube\ Studio_Data/StreamingAssets/Live2DModels/hiyori_vts

uv run python -m pipeline.measure_regions \
    --moc3 "$HIYORI/hiyori.moc3" \
    --textures "$HIYORI/hiyori.2048/texture_00.png" "$HIYORI/hiyori.2048/texture_01.png" \
    --rig hiyori \
    --template humanoid-anime \
    --out manifests/hiyori_atlas.toml \
    --show
```

The tool prints 133 drawables with their UV bounding boxes. Use the matplotlib
overlay (--show) to identify which ArtMesh IDs correspond to face_skin, left_eye,
right_eye, left_eyebrow, right_eyebrow, mouth, hair_front.

Enter assignments at the interactive prompt, e.g.:
```
> face_skin 0 1 2 3
> left_eye 15 16
> right_eye 17 18
> done
```

- [ ] **Step 3: Verify the TOML looks correct**

```bash
uv run python -c "
from pipeline.atlas_config import load_atlas_config
from pathlib import Path
config = load_atlas_config(Path('manifests/hiyori_atlas.toml'))
print(f'Loaded {len(config.regions)} regions for {config.rig_name}')
for r in config.regions:
    print(f'  {r.name}: tex={r.texture_index} ({r.x},{r.y},{r.w},{r.h})')
"
```

- [ ] **Step 4: Commit the measured atlas**

```bash
git add manifests/hiyori_atlas.toml
git commit -m "feat(manifests): hiyori_atlas.toml — measured UV bbox coordinates"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `pipeline/atlas_config.py` — AtlasRegion, AtlasConfig, load_atlas_config | Task 1 |
| `templates/humanoid-anime/atlas_schema.toml` | Task 2 |
| `manifests/hiyori_atlas.toml` | Task 2 (stub) + Task 6 (real) |
| `pipeline/texture_swap.py` — swap_region, swap_regions | Task 3 |
| `pipeline/validate.py` — validate_textures, check_region_color | Task 4 |
| `pipeline/measure_regions.py` — ctypes UV extraction + labeling | Task 5 |
| Tests: atlas config loading | Task 1 |
| Tests: region swapping pixels | Task 3 |
| Tests: alpha compositing | Task 3 |
| Tests: validate headless render | Task 4 |
| Tests: check_region_color | Task 4 |

All spec requirements covered. ✓

**Placeholder scan:** None found. ✓

**Type consistency:**
- `AtlasRegion.texture_index: int` matches usage in `swap_regions` (`out[region.texture_index]`) ✓
- `AtlasConfig.get(name) -> AtlasRegion` matches `swap_region(atlas, region, replacement)` ✓
- `validate_textures(config, modified_atlases: dict[int, Image.Image])` ✓
- `swap_regions` returns `dict[int, Image.Image]`, passed directly to `validate_textures` ✓
