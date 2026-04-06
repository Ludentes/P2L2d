# Texture Generation Pipeline — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a portrait photo, generate per-region textures stylized to anime flat-cell art and assembled into a deliverable Live2D model directory.

**Architecture:** Portrait → style transfer (ComfyUI SDXL img2img) → affine warp to template space → per-region crop → face skin inpainting (FLUX Fill) + hair segmentation (BiRefNet) → swap_regions() → package moc3 + modified atlases into output dir. Three ComfyUI workflow JSONs are parameterized with placeholder strings substituted at runtime.

**Tech Stack:** Python 3.12, PIL/Pillow, NumPy, MediaPipe (face landmarks), ComfyUI REST API (existing client at `comfyui/client.py`), pytest-asyncio

---

## Codebase Context

Read these before touching anything:
- `pipeline/atlas_config.py` — `AtlasConfig`, `AtlasRegion`, `load_atlas_config()`
- `pipeline/texture_swap.py` — `swap_regions(atlases, config, replacements)`
- `pipeline/validate.py` — `validate_textures()`, `check_region_color()`
- `rig/config.py` — `RigConfig`, `RIG_HIYORI`
- `comfyui/client.py` — `ComfyUIClient` (upload_image, submit, wait, download)
- `comfyui/exceptions.py` — `ComfyUIConnectionError`
- `manifests/hiyori_atlas.toml` — live atlas with 15 regions; **DO NOT MODIFY**
- `templates/humanoid-anime/schema.toml` — template param schema

Key facts:
- `AtlasConfig.texture_size = 2048` (Hiyori). All atlas bbox coords are in 2048×2048 space.
- `RIG_HIYORI.textures = [Path("...texture_00.png"), Path("...texture_01.png")]`
- ComfyUI workflow JSON uses placeholder strings `"__IMAGE__"`, `"__MASK__"`, `"__MODEL__"`, `"__STRENGTH__"` that are string-replaced before submitting.
- `client.wait()` returns `{node_id: {"images": [{"filename": "...", "subfolder": "...", "type": "output"}]}}`
- Tests in `tests/pipeline/` already exist for atlas_config, texture_swap, validate. Follow the same pattern: imports inside test functions, `skip_no_model` marker for integration tests.
- `asyncio_mode = "auto"` in pyproject.toml — async tests don't need `@pytest.mark.asyncio`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `templates/humanoid-anime/schema.toml` | Modify | Add `[texture_generation]` section |
| `pipeline/exceptions.py` | Create | `MediaPipeLandmarkError` |
| `pipeline/face_align.py` | Create | Landmark detection, affine warp, region crop, inpaint mask |
| `pipeline/style_transfer.py` | Create | `TextureGenConfig`, `load_texture_gen_config`, `stylize_portrait` |
| `pipeline/face_inpaint.py` | Create | `inpaint_face_skin` via ComfyUI FLUX Fill |
| `pipeline/hair_segment.py` | Create | `segment_hair`, `extract_hair_regions` via ComfyUI BiRefNet |
| `pipeline/texture_gen.py` | Create | `generate_textures` — full Phase 1 orchestrator |
| `pipeline/package.py` | Create | `package_output` — assemble deliverable dir |
| `pipeline/run.py` | Create | `load_atlases`, `run_portrait_to_rig`, CLI `__main__` |
| `comfyui/workflows/style_transfer_anime.json` | Create | SDXL img2img workflow template |
| `comfyui/workflows/face_inpaint.json` | Create | FLUX Fill inpainting workflow template |
| `comfyui/workflows/hair_segment.json` | Create | BiRefNet segmentation workflow template |
| `scripts/generate_template_landmarks.py` | Create | One-time script: render rig → MediaPipe → save JSON |
| `templates/humanoid-anime/face_landmarks.json` | Generate | Run the script above |
| `tests/pipeline/test_face_align.py` | Create | Unit tests — no model, no ComfyUI |
| `tests/pipeline/test_style_transfer.py` | Create | Unit tests with mock ComfyUI client |
| `tests/pipeline/test_face_inpaint.py` | Create | Unit tests with mock ComfyUI client |
| `tests/pipeline/test_hair_segment.py` | Create | Unit tests with mock ComfyUI client |
| `tests/pipeline/test_texture_gen.py` | Create | Unit tests with full mock pipeline |
| `tests/pipeline/test_package.py` | Create | Unit tests — file system only |
| `tests/pipeline/test_run.py` | Create | Unit tests — CLI arg parsing, load_atlases |
| `tests/pipeline/test_texture_gen_integration.py` | Create | Integration test (skip if no model/ComfyUI) |

---

## Task 1: Schema extension + exceptions

**Files:**
- Modify: `templates/humanoid-anime/schema.toml`
- Create: `pipeline/exceptions.py`

- [ ] **Step 1: Append `[texture_generation]` to schema.toml**

Add to the end of `templates/humanoid-anime/schema.toml`:

```toml

[texture_generation]
style_transfer = "anime_flat_cell"   # "none" | "anime_flat_cell"
style_model    = "noobai-xl"         # ComfyUI checkpoint name (ignored if "none")
style_strength = 0.65                # img2img denoise strength (0.0–1.0)
```

- [ ] **Step 2: Verify schema.toml loads cleanly**

```bash
python3 -c "
import tomllib
data = tomllib.loads(open('templates/humanoid-anime/schema.toml').read())
tg = data['texture_generation']
assert tg['style_transfer'] == 'anime_flat_cell'
assert tg['style_model'] == 'noobai-xl'
assert abs(tg['style_strength'] - 0.65) < 1e-9
print('OK:', tg)
"
```

Expected: `OK: {'style_transfer': 'anime_flat_cell', 'style_model': 'noobai-xl', 'style_strength': 0.65}`

- [ ] **Step 3: Create `pipeline/exceptions.py`**

```python
class MediaPipeLandmarkError(Exception):
    """MediaPipe failed to detect face landmarks in the portrait."""
```

- [ ] **Step 4: Commit**

```bash
git add templates/humanoid-anime/schema.toml pipeline/exceptions.py
git commit -m "feat(pipeline): schema texture_generation config + MediaPipeLandmarkError"
```

---

## Task 2: generate_template_landmarks.py + face_landmarks.json

**Files:**
- Create: `scripts/generate_template_landmarks.py`
- Generate: `templates/humanoid-anime/face_landmarks.json`

This script renders Hiyori at neutral pose (512×512), runs MediaPipe, saves all 478 landmark pixel positions plus 7 named landmarks to JSON. Run once; commit the output.

- [ ] **Step 1: Write the script**

Create `scripts/generate_template_landmarks.py`:

```python
#!/usr/bin/env python3
"""Generate face_landmarks.json for a template by rendering its rig and running MediaPipe.

Usage:
    python scripts/generate_template_landmarks.py \\
        --template humanoid-anime \\
        --rig hiyori \\
        --size 512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mediapipe as mp
import numpy as np
from PIL import Image

from rig.config import RIG_HIYORI
from rig.render import RigRenderer

_MP_FACE = mp.solutions.face_mesh  # type: ignore[attr-defined]

# MediaPipe FaceMesh 478-point topology — named landmark indices
_NAMED_INDICES = {
    "left_eye_center": 473,    # iris centre left (requires refine_landmarks=True)
    "right_eye_center": 468,   # iris centre right
    "nose_tip": 1,
    "mouth_center": 13,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
}


def _run_mediapipe(frame_rgba: np.ndarray) -> np.ndarray:
    """Run MediaPipe FaceMesh on RGBA uint8 frame. Returns (478, 2) float32 pixel coords."""
    rgb = frame_rgba[:, :, :3].copy()
    h, w = rgb.shape[:2]
    with _MP_FACE.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        raise RuntimeError("MediaPipe detected no face in the rig render")
    lms = result.multi_face_landmarks[0].landmark
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


def generate(template: str, rig_config, size: int, out_path: Path) -> None:
    print(f"Rendering {rig_config.name} at {size}x{size} ...")
    with RigRenderer(rig_config) as renderer:
        frame = renderer.render(params=None)  # neutral pose, (H, W, 4) uint8

    img = Image.fromarray(frame, mode="RGBA").resize((size, size), Image.Resampling.LANCZOS)
    frame_resized = np.array(img)

    print("Running MediaPipe ...")
    pts = _run_mediapipe(frame_resized)  # (478, 2)

    named = {name: pts[idx].tolist() for name, idx in _NAMED_INDICES.items()}
    named["mediapipe_full"] = pts.tolist()

    data = {
        "template": template,
        "render_size": [size, size],
        "generated_from": rig_config.name,
        "landmarks": named,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(pts)} landmarks -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="humanoid-anime")
    parser.add_argument("--rig", default="hiyori")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    rig_map = {"hiyori": RIG_HIYORI}
    if args.rig not in rig_map:
        raise SystemExit(f"Unknown rig {args.rig!r}. Available: {list(rig_map)}")

    out = Path(f"templates/{args.template}/face_landmarks.json")
    generate(args.template, rig_map[args.rig], args.size, out)
```

- [ ] **Step 2: Run the script (requires Hiyori model)**

```bash
python scripts/generate_template_landmarks.py --template humanoid-anime --rig hiyori --size 512
```

Expected:
```
Rendering hiyori at 512x512 ...
Running MediaPipe ...
Saved 478 landmarks -> templates/humanoid-anime/face_landmarks.json
```

- [ ] **Step 3: Verify the JSON**

```bash
python3 -c "
import json
data = json.loads(open('templates/humanoid-anime/face_landmarks.json').read())
lms = data['landmarks']
print('template:', data['template'])
print('render_size:', data['render_size'])
print('full landmark count:', len(lms['mediapipe_full']))
print('nose_tip:', lms['nose_tip'])
"
```

Expected: 478 landmarks, nose_tip near [256, 256] for a centred render.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_template_landmarks.py templates/humanoid-anime/face_landmarks.json
git commit -m "feat(scripts): generate_template_landmarks + humanoid-anime face_landmarks.json"
```

---

## Task 3: face_align.py

**Files:**
- Create: `pipeline/face_align.py`
- Create: `tests/pipeline/test_face_align.py`

Provides: landmark detection from portrait, loading template landmarks, affine warp to template space, region crop scaled from atlas coords, face inpaint mask builder.

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_face_align.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion


def _make_landmarks_file(tmp_path: Path, size: int = 512) -> Path:
    """Write a minimal face_landmarks.json with 478 random points."""
    pts = np.random.default_rng(0).uniform(50, 462, size=(478, 2)).tolist()
    named = {
        "left_eye_center": pts[473],
        "right_eye_center": pts[468],
        "nose_tip": pts[1],
        "mouth_center": pts[13],
        "chin": pts[152],
        "left_cheek": pts[234],
        "right_cheek": pts[454],
        "mediapipe_full": pts,
    }
    data = {"template": "test", "render_size": [size, size],
            "generated_from": "test_rig", "landmarks": named}
    p = tmp_path / "face_landmarks.json"
    p.write_text(json.dumps(data))
    return p


def test_load_template_landmarks_returns_array(tmp_path):
    from pipeline.face_align import load_template_landmarks

    lm_file = _make_landmarks_file(tmp_path)
    pts, render_size = load_template_landmarks(lm_file)

    assert pts.shape == (478, 2)
    assert pts.dtype == np.float32
    assert render_size == (512, 512)


def test_compute_affine_identity():
    from pipeline.face_align import compute_affine_transform

    pts = np.random.default_rng(1).uniform(0, 512, (478, 2)).astype(np.float32)
    M = compute_affine_transform(pts, pts)

    ones = np.ones((478, 1), dtype=np.float32)
    result = (M @ np.hstack([pts, ones]).T).T
    np.testing.assert_allclose(result, pts, atol=1e-3)


def test_warp_image_output_size(tmp_path):
    from pipeline.face_align import compute_affine_transform, warp_image

    portrait = Image.new("RGB", (300, 400), color=(128, 64, 32))
    rng = np.random.default_rng(2)
    src = rng.uniform(0, 300, (478, 2)).astype(np.float32)
    dst = rng.uniform(0, 512, (478, 2)).astype(np.float32)
    M = compute_affine_transform(src, dst)

    warped = warp_image(portrait, M, output_size=(512, 512))

    assert warped.size == (512, 512)
    assert warped.mode == "RGB"


def test_warp_image_rgba_preserves_mode():
    from pipeline.face_align import compute_affine_transform, warp_image

    portrait = Image.new("RGBA", (300, 400), color=(100, 200, 50, 180))
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 300, (478, 2)).astype(np.float32)
    dst = rng.uniform(0, 512, (478, 2)).astype(np.float32)
    M = compute_affine_transform(src, dst)

    warped = warp_image(portrait, M, output_size=(512, 512))

    assert warped.mode == "RGBA"


def test_crop_region_scales_atlas_coords():
    from pipeline.face_align import crop_region

    region = AtlasRegion(name="face_skin", texture_index=0, x=100, y=50, w=400, h=600)
    cfg = AtlasConfig(rig_name="t", template_name="t", texture_size=2048, regions=[region])
    warped = Image.new("RGB", (512, 512), color=(200, 100, 50))

    crop = crop_region(warped, cfg, "face_skin")

    # x=100/2048*512=25, y=50/2048*512≈12, w=400/2048*512=100, h=600/2048*512=150
    assert crop.width == 100
    assert crop.height == 150


def test_build_face_inpaint_mask_shape():
    from pipeline.face_align import build_face_inpaint_mask

    face = AtlasRegion("face_skin", 0, 0, 0, 512, 512)
    eye_l = AtlasRegion("left_eye", 0, 50, 100, 80, 60)
    eye_r = AtlasRegion("right_eye", 0, 200, 100, 80, 60)
    cfg = AtlasConfig("t", "t", 512, [face, eye_l, eye_r])

    mask = build_face_inpaint_mask(
        cfg, warped_size=(512, 512), face_region_name="face_skin",
        feature_regions=["left_eye", "right_eye"], dilation_px=4,
    )

    assert mask.mode == "L"
    assert mask.size == (512, 512)
    arr = np.array(mask)
    assert arr.max() == 255   # white where eyes are
    assert arr.min() == 0     # black elsewhere
```

- [ ] **Step 2: Run tests, confirm ImportError**

```bash
pytest tests/pipeline/test_face_align.py -v 2>&1 | head -15
```

- [ ] **Step 3: Write `pipeline/face_align.py`**

```python
"""Face alignment — landmark detection, affine warp, region crop, inpaint mask."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pipeline.atlas_config import AtlasConfig
from pipeline.exceptions import MediaPipeLandmarkError


def detect_landmarks(image: Image.Image) -> np.ndarray:
    """Run MediaPipe FaceMesh on image. Returns (478, 2) float32 pixel coords.

    Raises MediaPipeLandmarkError if no face detected.
    """
    import mediapipe as mp  # deferred — heavy import

    rgb = image.convert("RGB")
    arr = np.array(rgb)
    h, w = arr.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(arr)
    if not result.multi_face_landmarks:
        raise MediaPipeLandmarkError(
            "MediaPipe detected no face in the portrait. "
            "Try a clearer frontal photo."
        )
    lms = result.multi_face_landmarks[0].landmark
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


def load_template_landmarks(lm_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load face_landmarks.json.

    Returns:
        pts: (478, 2) float32 in render space.
        render_size: (width, height) of the render used to generate the JSON.
    """
    data = json.loads(lm_path.read_text())
    pts = np.array(data["landmarks"]["mediapipe_full"], dtype=np.float32)
    w, h = data["render_size"]
    return pts, (w, h)


def compute_affine_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute 2x3 affine matrix mapping src_pts -> dst_pts (least-squares).

    Returns M (2, 3) such that M @ [x, y, 1]^T -> [x', y']^T.
    """
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float32)])
    M_T, _, _, _ = np.linalg.lstsq(src_h, dst_pts, rcond=None)
    return M_T.T.astype(np.float32)  # (2, 3)


def warp_image(
    image: Image.Image,
    M: np.ndarray,
    output_size: tuple[int, int],
) -> Image.Image:
    """Warp image using affine matrix M (forward: source->output).

    PIL expects inverse mapping; M is inverted internally.
    """
    M_full = np.vstack([M, [0.0, 0.0, 1.0]])
    M_inv = np.linalg.inv(M_full)[:2, :]
    data = M_inv.flatten().tolist()
    fill: tuple = (0, 0, 0, 0) if image.mode == "RGBA" else (0, 0, 0)
    return image.transform(
        output_size,
        Image.Transform.AFFINE,
        data,
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )


def align_portrait(
    portrait: Image.Image,
    template_landmarks_path: Path,
    output_size: tuple[int, int] = (512, 512),
) -> tuple[Image.Image, np.ndarray]:
    """Detect landmarks in portrait, warp to template space.

    Returns:
        warped: Portrait in template space at output_size.
        M: (2, 3) affine matrix (portrait -> output). Reuse with warp_image()
           to warp other images (e.g. hair RGBA) with the same transform.

    Raises MediaPipeLandmarkError if no face detected.
    """
    template_pts, render_size = load_template_landmarks(template_landmarks_path)
    portrait_pts = detect_landmarks(portrait)
    sx = output_size[0] / render_size[0]
    sy = output_size[1] / render_size[1]
    template_pts_scaled = template_pts * np.array([sx, sy], dtype=np.float32)
    M = compute_affine_transform(portrait_pts, template_pts_scaled)
    warped = warp_image(portrait, M, output_size)
    return warped, M


def crop_region(
    warped: Image.Image,
    atlas_cfg: AtlasConfig,
    region_name: str,
) -> Image.Image:
    """Crop named region from warped portrait, scaling atlas bbox to warped size."""
    region = atlas_cfg.get(region_name)
    scale_x = warped.width / atlas_cfg.texture_size
    scale_y = warped.height / atlas_cfg.texture_size
    x = int(region.x * scale_x)
    y = int(region.y * scale_y)
    w = max(1, int(region.w * scale_x))
    h = max(1, int(region.h * scale_y))
    return warped.crop((x, y, x + w, y + h))


def build_face_inpaint_mask(
    atlas_cfg: AtlasConfig,
    warped_size: tuple[int, int] = (512, 512),
    face_region_name: str = "face_skin",
    feature_regions: list[str] | None = None,
    dilation_px: int = 8,
) -> Image.Image:
    """Build inpainting mask for face skin region.

    White = inpaint (where animated features appear), black = keep.
    Sized to the face crop dimensions (atlas face bbox scaled to warped_size).
    """
    if feature_regions is None:
        feature_regions = ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "mouth"]

    scale_x = warped_size[0] / atlas_cfg.texture_size
    scale_y = warped_size[1] / atlas_cfg.texture_size

    face_r = atlas_cfg.get(face_region_name)
    face_x = int(face_r.x * scale_x)
    face_y = int(face_r.y * scale_y)
    face_w = max(1, int(face_r.w * scale_x))
    face_h = max(1, int(face_r.h * scale_y))

    mask = Image.new("L", (face_w, face_h), 0)
    draw = ImageDraw.Draw(mask)

    for feat_name in feature_regions:
        if not atlas_cfg.has(feat_name):
            continue
        feat = atlas_cfg.get(feat_name)
        fx = int(feat.x * scale_x) - face_x
        fy = int(feat.y * scale_y) - face_y
        fw = max(1, int(feat.w * scale_x))
        fh = max(1, int(feat.h * scale_y))
        draw.rectangle([fx, fy, fx + fw, fy + fh], fill=255)

    if dilation_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=dilation_px * 2 + 1))

    return mask
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_face_align.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/face_align.py tests/pipeline/test_face_align.py
git commit -m "feat(pipeline): face_align — landmark detection, affine warp, crop, inpaint mask"
```

---

## Task 4: style_transfer.py + ComfyUI workflow

**Files:**
- Create: `pipeline/style_transfer.py`
- Create: `comfyui/workflows/style_transfer_anime.json`
- Create: `tests/pipeline/test_style_transfer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_style_transfer.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image


def test_load_texture_gen_config():
    from pipeline.style_transfer import load_texture_gen_config

    cfg = load_texture_gen_config("humanoid-anime")

    assert cfg.style_transfer == "anime_flat_cell"
    assert cfg.style_model == "noobai-xl"
    assert 0.0 < cfg.style_strength <= 1.0


def test_load_texture_gen_config_missing_template():
    from pipeline.style_transfer import load_texture_gen_config

    with pytest.raises(FileNotFoundError):
        load_texture_gen_config("nonexistent-template")


async def test_stylize_portrait_none_passthrough():
    from pipeline.style_transfer import stylize_portrait

    portrait = Image.new("RGB", (512, 512), color=(200, 100, 50))
    client = MagicMock()

    result = await stylize_portrait(portrait, style="none", model="any", strength=0.5, client=client)

    assert result is portrait
    client.upload_image.assert_not_called()


async def test_stylize_portrait_anime_calls_comfyui():
    from pipeline.style_transfer import stylize_portrait

    portrait = Image.new("RGB", (512, 512), color=(200, 100, 50))
    fake_output = Image.new("RGB", (512, 512), color=(50, 150, 200))

    client = AsyncMock()
    client.upload_image.return_value = "portrait_123.png"
    client.submit.return_value = "prompt-abc"
    client.wait.return_value = {
        "8": {"images": [{"filename": "p2l_style_00001.png", "subfolder": "", "type": "output"}]}
    }

    async def fake_download(filename, dest, subfolder="", file_type="output"):
        fake_output.save(dest)

    client.download.side_effect = fake_download

    result = await stylize_portrait(
        portrait, style="anime_flat_cell", model="noobai-xl", strength=0.65, client=client
    )

    client.upload_image.assert_called_once()
    client.submit.assert_called_once()
    client.wait.assert_called_once_with("prompt-abc")
    client.download.assert_called_once()
    assert isinstance(result, Image.Image)
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/pipeline/test_style_transfer.py -v 2>&1 | head -15
```

- [ ] **Step 3: Create `comfyui/workflows/style_transfer_anime.json`**

```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {"image": "__IMAGE__", "upload": "image"},
    "_meta": {"title": "Load Portrait"}
  },
  "2": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {"ckpt_name": "__MODEL__"},
    "_meta": {"title": "Load Anime Model"}
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "anime style illustration, flat cell shading, clean crisp lines, vibrant colors",
      "clip": ["2", 1]
    },
    "_meta": {"title": "Positive Prompt"}
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "photo, realistic, 3d render, blurry, noisy", "clip": ["2", 1]},
    "_meta": {"title": "Negative Prompt"}
  },
  "5": {
    "class_type": "VAEEncode",
    "inputs": {"pixels": ["1", 0], "vae": ["2", 2]},
    "_meta": {"title": "Encode Image"}
  },
  "6": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["2", 0],
      "positive": ["3", 0],
      "negative": ["4", 0],
      "latent_image": ["5", 0],
      "seed": 42,
      "steps": 20,
      "cfg": 7.0,
      "sampler_name": "euler_ancestral",
      "scheduler": "karras",
      "denoise": "__STRENGTH__"
    },
    "_meta": {"title": "Sample"}
  },
  "7": {
    "class_type": "VAEDecode",
    "inputs": {"samples": ["6", 0], "vae": ["2", 2]},
    "_meta": {"title": "Decode"}
  },
  "8": {
    "class_type": "SaveImage",
    "inputs": {"images": ["7", 0], "filename_prefix": "p2l_style"},
    "_meta": {"title": "Save"}
  }
}
```

Note: `__IMAGE__` and `__MODEL__` are JSON string placeholders (replaced with `json.dumps(value)`); `__STRENGTH__` is a JSON number placeholder (replaced with `str(value)`).

- [ ] **Step 4: Write `pipeline/style_transfer.py`**

```python
"""Style transfer — stylize portrait to match rig art style via ComfyUI."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tomllib
from PIL import Image

from comfyui.client import ComfyUIClient

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_WORKFLOWS_DIR = Path(__file__).parent.parent / "comfyui" / "workflows"


@dataclass
class TextureGenConfig:
    style_transfer: str    # "none" | "anime_flat_cell"
    style_model: str       # ComfyUI checkpoint name
    style_strength: float  # img2img denoise strength 0.0-1.0


def load_texture_gen_config(template_name: str) -> TextureGenConfig:
    """Load [texture_generation] from templates/{template_name}/schema.toml.

    Raises FileNotFoundError if template does not exist.
    """
    schema_path = _TEMPLATES_DIR / template_name / "schema.toml"
    if not schema_path.exists():
        raise FileNotFoundError(f"Template schema not found: {schema_path}")
    with open(schema_path, "rb") as f:
        data = tomllib.load(f)
    tg = data["texture_generation"]
    return TextureGenConfig(
        style_transfer=tg["style_transfer"],
        style_model=tg["style_model"],
        style_strength=float(tg["style_strength"]),
    )


async def stylize_portrait(
    portrait: Image.Image,
    style: str,
    model: str,
    strength: float,
    client: ComfyUIClient,
) -> Image.Image:
    """Stylize portrait via ComfyUI. Returns portrait unchanged if style == 'none'."""
    if style == "none":
        return portrait

    workflow_text = (_WORKFLOWS_DIR / "style_transfer_anime.json").read_text()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    portrait.save(tmp_path, format="PNG")
    uploaded_name = await client.upload_image(tmp_path)
    tmp_path.unlink(missing_ok=True)

    workflow_text = workflow_text.replace('"__IMAGE__"', json.dumps(uploaded_name))
    workflow_text = workflow_text.replace('"__MODEL__"', json.dumps(model))
    workflow_text = workflow_text.replace('"__STRENGTH__"', str(strength))
    workflow = json.loads(workflow_text)

    prompt_id = await client.submit(workflow)
    outputs = await client.wait(prompt_id, timeout=180.0)
    out_filename = _extract_output_filename(outputs)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        dest = Path(tmp.name)
    await client.download(out_filename, dest)
    result = Image.open(dest).copy()
    dest.unlink(missing_ok=True)

    return result


def _extract_output_filename(outputs: dict) -> str:
    for node_outputs in outputs.values():
        images = node_outputs.get("images", [])
        if images:
            return images[0]["filename"]
    raise ValueError(f"No images in ComfyUI outputs: {outputs}")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/pipeline/test_style_transfer.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/style_transfer.py comfyui/workflows/style_transfer_anime.json tests/pipeline/test_style_transfer.py
git commit -m "feat(pipeline): style_transfer — SDXL img2img + TextureGenConfig loader"
```

---

## Task 5: face_inpaint.py + workflow

**Files:**
- Create: `pipeline/face_inpaint.py`
- Create: `comfyui/workflows/face_inpaint.json`
- Create: `tests/pipeline/test_face_inpaint.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_face_inpaint.py`:

```python
from unittest.mock import AsyncMock

import pytest
from PIL import Image


async def test_inpaint_face_skin_uploads_two_images():
    from pipeline.face_inpaint import inpaint_face_skin

    face_crop = Image.new("RGB", (256, 256), color=(220, 180, 160))
    mask = Image.new("L", (256, 256), 0)
    fake_output = Image.new("RGB", (256, 256), color=(210, 170, 150))

    client = AsyncMock()
    client.upload_image.return_value = "img.png"
    client.submit.return_value = "pid"
    client.wait.return_value = {
        "10": {"images": [{"filename": "p2l_inpaint_00001.png", "subfolder": "", "type": "output"}]}
    }

    async def fake_download(filename, dest, subfolder="", file_type="output"):
        fake_output.save(dest)

    client.download.side_effect = fake_download

    result = await inpaint_face_skin(face_crop, mask, client)

    assert client.upload_image.call_count == 2  # face + mask
    assert isinstance(result, Image.Image)


async def test_inpaint_face_skin_injects_prompt():
    from pipeline.face_inpaint import inpaint_face_skin

    face_crop = Image.new("RGB", (128, 128))
    mask = Image.new("L", (128, 128), 255)
    fake_output = Image.new("RGB", (128, 128))

    captured = {}

    client = AsyncMock()
    client.upload_image.return_value = "img.png"
    client.wait.return_value = {
        "10": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
    }

    async def capture_submit(workflow):
        captured["workflow"] = workflow
        return "pid"

    client.submit.side_effect = capture_submit

    async def fake_download(filename, dest, subfolder="", file_type="output"):
        fake_output.save(dest)

    client.download.side_effect = fake_download

    custom_prompt = "pale alabaster skin"
    await inpaint_face_skin(face_crop, mask, client, prompt=custom_prompt)

    assert custom_prompt in str(captured["workflow"])
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/pipeline/test_face_inpaint.py -v 2>&1 | head -10
```

- [ ] **Step 3: Create `comfyui/workflows/face_inpaint.json`**

```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {"image": "__IMAGE__", "upload": "image"},
    "_meta": {"title": "Load Face Crop"}
  },
  "2": {
    "class_type": "LoadImage",
    "inputs": {"image": "__MASK__", "upload": "image"},
    "_meta": {"title": "Load Mask"}
  },
  "3": {
    "class_type": "ImageToMask",
    "inputs": {"image": ["2", 0], "channel": "red"},
    "_meta": {"title": "Convert to Mask"}
  },
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {"ckpt_name": "flux1-fill-dev.safetensors"},
    "_meta": {"title": "Load FLUX Fill"}
  },
  "5": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "__PROMPT__", "clip": ["4", 1]},
    "_meta": {"title": "Prompt"}
  },
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "", "clip": ["4", 1]},
    "_meta": {"title": "Negative"}
  },
  "7": {
    "class_type": "VAEEncodeForInpaint",
    "inputs": {"pixels": ["1", 0], "vae": ["4", 2], "mask": ["3", 0], "grow_mask_by": 8},
    "_meta": {"title": "Encode for Inpaint"}
  },
  "8": {
    "class_type": "KSampler",
    "inputs": {
      "model": ["4", 0],
      "positive": ["5", 0],
      "negative": ["6", 0],
      "latent_image": ["7", 0],
      "seed": 42,
      "steps": 20,
      "cfg": 1.0,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 0.85
    },
    "_meta": {"title": "Sample"}
  },
  "9": {
    "class_type": "VAEDecode",
    "inputs": {"samples": ["8", 0], "vae": ["4", 2]},
    "_meta": {"title": "Decode"}
  },
  "10": {
    "class_type": "SaveImage",
    "inputs": {"images": ["9", 0], "filename_prefix": "p2l_inpaint"},
    "_meta": {"title": "Save"}
  }
}
```

- [ ] **Step 4: Write `pipeline/face_inpaint.py`**

```python
"""Face skin inpainting — remove portrait facial features, replace with clean skin."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from comfyui.client import ComfyUIClient

_WORKFLOWS_DIR = Path(__file__).parent.parent / "comfyui" / "workflows"
_DEFAULT_PROMPT = (
    "smooth skin texture, same skin tone as surrounding area, "
    "no eyes, no eyebrows, no mouth, no facial features"
)


async def inpaint_face_skin(
    face_crop: Image.Image,
    mask: Image.Image,
    client: ComfyUIClient,
    prompt: str = _DEFAULT_PROMPT,
) -> Image.Image:
    """Inpaint masked regions of face_crop with clean skin via FLUX Fill.

    Args:
        face_crop: RGB face skin crop.
        mask: L-mode image same size as face_crop. White = inpaint, black = keep.
        client: ComfyUI client.
        prompt: Inpainting text prompt.

    Returns:
        Inpainted RGB image.
    """
    workflow_text = (_WORKFLOWS_DIR / "face_inpaint.json").read_text()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        face_path = Path(tmp.name)
    face_crop.convert("RGB").save(face_path, format="PNG")
    face_name = await client.upload_image(face_path)
    face_path.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        mask_path = Path(tmp.name)
    Image.merge("RGB", [mask, mask, mask]).save(mask_path, format="PNG")
    mask_name = await client.upload_image(mask_path)
    mask_path.unlink(missing_ok=True)

    workflow_text = workflow_text.replace('"__IMAGE__"', json.dumps(face_name))
    workflow_text = workflow_text.replace('"__MASK__"', json.dumps(mask_name))
    workflow_text = workflow_text.replace('"__PROMPT__"', json.dumps(prompt))
    workflow = json.loads(workflow_text)

    prompt_id = await client.submit(workflow)
    outputs = await client.wait(prompt_id, timeout=180.0)
    out_filename = _extract_output_filename(outputs)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        dest = Path(tmp.name)
    await client.download(out_filename, dest)
    result = Image.open(dest).convert("RGB").copy()
    dest.unlink(missing_ok=True)

    return result


def _extract_output_filename(outputs: dict) -> str:
    for node_outputs in outputs.values():
        images = node_outputs.get("images", [])
        if images:
            return images[0]["filename"]
    raise ValueError(f"No images in ComfyUI outputs: {outputs}")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/pipeline/test_face_inpaint.py -v
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/face_inpaint.py comfyui/workflows/face_inpaint.json tests/pipeline/test_face_inpaint.py
git commit -m "feat(pipeline): face_inpaint — FLUX Fill skin inpainting via ComfyUI"
```

---

## Task 6: hair_segment.py + workflow

**Files:**
- Create: `pipeline/hair_segment.py`
- Create: `comfyui/workflows/hair_segment.json`
- Create: `tests/pipeline/test_hair_segment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_hair_segment.py`:

```python
import warnings
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion


def _cfg():
    return AtlasConfig("t", "t", 512, [
        AtlasRegion("hair_front", 0, 300, 0, 100, 80),
        AtlasRegion("hair_back", 0, 300, 100, 100, 80),
    ])


async def test_segment_hair_returns_rgba():
    from pipeline.hair_segment import segment_hair

    portrait = Image.new("RGB", (512, 512), color=(200, 150, 100))
    fake_rgba = Image.new("RGBA", (512, 512), color=(80, 60, 40, 200))

    client = AsyncMock()
    client.upload_image.return_value = "portrait.png"
    client.submit.return_value = "pid"
    client.wait.return_value = {
        "6": {"images": [{"filename": "hair_seg.png", "subfolder": "", "type": "output"}]}
    }

    async def fake_download(filename, dest, subfolder="", file_type="output"):
        fake_rgba.save(dest)

    client.download.side_effect = fake_download

    result = await segment_hair(portrait, client)

    assert result.mode == "RGBA"
    client.upload_image.assert_called_once()


def test_extract_hair_regions_crops_correctly():
    from pipeline.hair_segment import extract_hair_regions

    cfg = _cfg()
    hair_rgba = Image.new("RGBA", (512, 512), color=(0, 200, 0, 255))

    regions = extract_hair_regions(hair_rgba, cfg, ["hair_front", "hair_back"])

    assert set(regions.keys()) == {"hair_front", "hair_back"}
    assert regions["hair_front"].size == (100, 80)
    assert regions["hair_back"].size == (100, 80)
    assert regions["hair_front"].mode == "RGBA"


def test_extract_hair_regions_warns_on_sparse():
    from pipeline.hair_segment import extract_hair_regions

    cfg = _cfg()
    hair_rgba = Image.new("RGBA", (512, 512), color=(0, 0, 0, 0))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extract_hair_regions(hair_rgba, cfg, ["hair_front"])

    assert len(w) == 1
    assert "hair" in str(w[0].message).lower() or "sparse" in str(w[0].message).lower()
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/pipeline/test_hair_segment.py -v 2>&1 | head -10
```

- [ ] **Step 3: Create `comfyui/workflows/hair_segment.json`**

```json
{
  "1": {
    "class_type": "LoadImage",
    "inputs": {"image": "__IMAGE__", "upload": "image"},
    "_meta": {"title": "Load Portrait"}
  },
  "2": {
    "class_type": "BiRefNetUltra",
    "inputs": {"images": ["1", 0], "model": "General", "render_size": 1024, "threshold": 0.5},
    "_meta": {
      "title": "BiRefNet Hair Segmentation",
      "note": "Requires ComfyUI-BiRefNet-Ultra. Install from https://github.com/viperyl/ComfyUI-BiRefNet-Ultra"
    }
  },
  "3": {
    "class_type": "JoinImageWithAlpha",
    "inputs": {"image": ["1", 0], "alpha": ["2", 1]},
    "_meta": {"title": "Apply Hair Mask as Alpha"}
  },
  "6": {
    "class_type": "SaveImage",
    "inputs": {"images": ["3", 0], "filename_prefix": "p2l_hair"},
    "_meta": {"title": "Save Hair RGBA"}
  }
}
```

Note: Node ID 6 is intentional (matches `client.wait` output key in tests). BiRefNet output index 1 is the mask. Adjust node class_type and inputs based on the exact BiRefNet custom node installed in your ComfyUI.

- [ ] **Step 4: Write `pipeline/hair_segment.py`**

```python
"""Hair segmentation — extract RGBA hair image via BiRefNet ComfyUI workflow."""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from comfyui.client import ComfyUIClient
from pipeline.atlas_config import AtlasConfig

_WORKFLOWS_DIR = Path(__file__).parent.parent / "comfyui" / "workflows"
_SPARSE_THRESHOLD = 0.01  # warn if < 1% pixels have any alpha


async def segment_hair(
    portrait: Image.Image,
    client: ComfyUIClient,
) -> Image.Image:
    """Segment hair from portrait via BiRefNet ComfyUI workflow.

    Returns RGBA: hair pixels kept, everything else transparent.
    """
    workflow_text = (_WORKFLOWS_DIR / "hair_segment.json").read_text()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        portrait_path = Path(tmp.name)
    portrait.convert("RGB").save(portrait_path, format="PNG")
    uploaded_name = await client.upload_image(portrait_path)
    portrait_path.unlink(missing_ok=True)

    workflow_text = workflow_text.replace('"__IMAGE__"', json.dumps(uploaded_name))
    workflow = json.loads(workflow_text)

    prompt_id = await client.submit(workflow)
    outputs = await client.wait(prompt_id, timeout=180.0)
    out_filename = _extract_output_filename(outputs)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        dest = Path(tmp.name)
    await client.download(out_filename, dest)
    result = Image.open(dest).convert("RGBA").copy()
    dest.unlink(missing_ok=True)

    return result


def extract_hair_regions(
    hair_rgba: Image.Image,
    atlas_cfg: AtlasConfig,
    region_names: list[str],
) -> dict[str, Image.Image]:
    """Crop per-region RGBA from warped (template-space) hair image.

    hair_rgba must already be warped to template space.
    Warns if < 1% pixels have any alpha (likely segmentation failure).
    """
    arr = np.array(hair_rgba)
    alpha_ratio = (arr[:, :, 3] > 0).mean()
    if alpha_ratio < _SPARSE_THRESHOLD:
        warnings.warn(
            f"Hair segmentation: < {_SPARSE_THRESHOLD*100:.0f}% non-transparent pixels. "
            "Hair regions will be mostly empty. Check portrait or hair model.",
            stacklevel=2,
        )

    result: dict[str, Image.Image] = {}
    scale_x = hair_rgba.width / atlas_cfg.texture_size
    scale_y = hair_rgba.height / atlas_cfg.texture_size

    for name in region_names:
        if not atlas_cfg.has(name):
            continue
        r = atlas_cfg.get(name)
        x = int(r.x * scale_x)
        y = int(r.y * scale_y)
        w = max(1, int(r.w * scale_x))
        h = max(1, int(r.h * scale_y))
        result[name] = hair_rgba.crop((x, y, x + w, y + h))

    return result


def _extract_output_filename(outputs: dict) -> str:
    for node_outputs in outputs.values():
        images = node_outputs.get("images", [])
        if images:
            return images[0]["filename"]
    raise ValueError(f"No images in ComfyUI outputs: {outputs}")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/pipeline/test_hair_segment.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/hair_segment.py comfyui/workflows/hair_segment.json tests/pipeline/test_hair_segment.py
git commit -m "feat(pipeline): hair_segment — BiRefNet segmentation + per-region RGBA crop"
```

---

## Task 7: texture_gen.py

**Files:**
- Create: `pipeline/texture_gen.py`
- Create: `tests/pipeline/test_texture_gen.py`

- [ ] **Step 1: Write failing test**

Create `tests/pipeline/test_texture_gen.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion
from rig.config import RIG_HIYORI


def _atlas():
    return AtlasConfig("t", "t", 512, [
        AtlasRegion("face_skin", 0, 0, 0, 200, 200),
        AtlasRegion("left_eye", 0, 210, 50, 60, 40),
        AtlasRegion("right_eye", 0, 210, 100, 60, 40),
        AtlasRegion("left_eyebrow", 0, 210, 150, 60, 20),
        AtlasRegion("right_eyebrow", 0, 210, 175, 60, 20),
        AtlasRegion("mouth", 0, 210, 200, 80, 40),
        AtlasRegion("hair_front", 0, 300, 0, 100, 80),
        AtlasRegion("hair_back", 0, 300, 90, 100, 80),
        AtlasRegion("hair_side_left", 0, 300, 180, 50, 80),
        AtlasRegion("hair_side_right", 0, 300, 270, 50, 80),
    ])


async def test_generate_textures_returns_expected_regions():
    from pipeline.texture_gen import generate_textures

    portrait = Image.new("RGB", (512, 512), color=(200, 150, 100))
    atlas_cfg = _atlas()
    client = AsyncMock()

    fake_img = Image.new("RGB", (512, 512), color=(180, 130, 90))
    fake_rgba = Image.new("RGBA", (512, 512), color=(80, 60, 40, 200))

    with (
        patch("pipeline.texture_gen.stylize_portrait", AsyncMock(return_value=fake_img)),
        patch("pipeline.texture_gen.align_portrait",
              return_value=(fake_img, np.eye(2, 3, dtype=np.float32))),
        patch("pipeline.texture_gen.inpaint_face_skin", AsyncMock(return_value=fake_img)),
        patch("pipeline.texture_gen.segment_hair", AsyncMock(return_value=fake_rgba)),
        patch("pipeline.texture_gen.warp_image", return_value=fake_rgba),
        patch("pipeline.texture_gen.load_texture_gen_config",
              return_value=MagicMock(style_transfer="none", style_model="m", style_strength=0.5)),
    ):
        result = await generate_textures(
            portrait=portrait,
            atlas_cfg=atlas_cfg,
            rig=RIG_HIYORI,
            client=client,
            template_name="humanoid-anime",
        )

    expected = {
        "face_skin", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow",
        "mouth", "hair_front", "hair_back", "hair_side_left", "hair_side_right",
    }
    assert set(result.keys()) == expected
    for name, img in result.items():
        assert isinstance(img, Image.Image), f"{name!r} is not a PIL Image"
```

- [ ] **Step 2: Run test, confirm it fails**

```bash
pytest tests/pipeline/test_texture_gen.py -v 2>&1 | head -10
```

- [ ] **Step 3: Write `pipeline/texture_gen.py`**

```python
"""Texture generation orchestrator — Phase 1 pipeline."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from comfyui.client import ComfyUIClient
from pipeline.atlas_config import AtlasConfig
from pipeline.face_align import (
    align_portrait,
    build_face_inpaint_mask,
    crop_region,
    warp_image,
)
from pipeline.face_inpaint import inpaint_face_skin
from pipeline.hair_segment import extract_hair_regions, segment_hair
from pipeline.style_transfer import load_texture_gen_config, stylize_portrait
from rig.config import RigConfig

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

_FACE_REGIONS = [
    "face_skin", "left_eye", "right_eye",
    "left_eyebrow", "right_eyebrow", "mouth",
    "left_cheek", "right_cheek",
]
_HAIR_REGIONS = ["hair_front", "hair_back", "hair_side_left", "hair_side_right"]
_WARP_SIZE = (512, 512)
_INPAINT_SIZE = (512, 512)


async def generate_textures(
    portrait: Image.Image,
    atlas_cfg: AtlasConfig,
    rig: RigConfig,
    client: ComfyUIClient,
    template_name: str = "humanoid-anime",
) -> dict[str, Image.Image]:
    """Generate per-region textures from portrait.

    Returns {region_name: PIL.Image} for all face + hair regions present in atlas_cfg.
    Skips regions absent from atlas_cfg without error.
    """
    cfg = load_texture_gen_config(template_name)
    lm_path = _TEMPLATES_DIR / template_name / "face_landmarks.json"

    # 1. Style transfer
    stylized = await stylize_portrait(
        portrait, style=cfg.style_transfer, model=cfg.style_model,
        strength=cfg.style_strength, client=client,
    )

    # 2. Align portrait to template space
    warped, M = align_portrait(stylized, lm_path, output_size=_WARP_SIZE)

    # 3. Crop face feature regions (not face_skin — that gets inpainted below)
    replacements: dict[str, Image.Image] = {}
    for region_name in _FACE_REGIONS:
        if region_name == "face_skin" or not atlas_cfg.has(region_name):
            continue
        replacements[region_name] = crop_region(warped, atlas_cfg, region_name)

    # 4. Inpaint face skin (remove portrait eyes/brows/mouth from the skin layer)
    if atlas_cfg.has("face_skin"):
        face_crop = crop_region(warped, atlas_cfg, "face_skin")
        face_large = face_crop.resize(_INPAINT_SIZE, Image.Resampling.LANCZOS)
        mask_raw = build_face_inpaint_mask(
            atlas_cfg, warped_size=_WARP_SIZE, face_region_name="face_skin",
            feature_regions=["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "mouth"],
            dilation_px=8,
        )
        mask_large = mask_raw.resize(_INPAINT_SIZE, Image.Resampling.NEAREST)
        replacements["face_skin"] = await inpaint_face_skin(face_large, mask_large, client)

    # 5. Hair segmentation (on stylized pre-warp portrait, then warp the mask)
    present_hair = [r for r in _HAIR_REGIONS if atlas_cfg.has(r)]
    if present_hair:
        hair_rgba = await segment_hair(stylized, client)
        warped_hair = warp_image(hair_rgba, M, output_size=_WARP_SIZE)
        replacements.update(extract_hair_regions(warped_hair, atlas_cfg, present_hair))

    return replacements
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_texture_gen.py -v
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add pipeline/texture_gen.py tests/pipeline/test_texture_gen.py
git commit -m "feat(pipeline): texture_gen — Phase 1 full orchestrator"
```

---

## Task 8: package.py

**Files:**
- Create: `pipeline/package.py`
- Create: `tests/pipeline/test_package.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_package.py`:

```python
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rig.config import RigConfig


def _fake_rig(tmp_path: Path) -> RigConfig:
    model_dir = tmp_path / "rig"
    tex_dir = model_dir / "textures"
    tex_dir.mkdir(parents=True)
    moc3 = model_dir / "char.moc3"
    moc3.write_bytes(b"moc3data")
    model3 = model_dir / "char.model3.json"
    model3.write_text('{"Version": 3}')
    tex0 = tex_dir / "texture_00.png"
    tex1 = tex_dir / "texture_01.png"
    Image.new("RGBA", (64, 64), color=(255, 0, 0, 255)).save(tex0)
    Image.new("RGBA", (64, 64), color=(0, 255, 0, 255)).save(tex1)
    return RigConfig(
        name="fake", model_dir=model_dir, moc3_path=moc3,
        model3_json_path=model3, textures=[tex0, tex1], param_ids=["P"],
    )


def test_package_output_creates_dir(tmp_path):
    from pipeline.package import package_output

    rig = _fake_rig(tmp_path)
    out = tmp_path / "out"
    result = package_output(rig, {}, out)
    assert result == out
    assert out.is_dir()


def test_package_output_copies_moc3_and_model3(tmp_path):
    from pipeline.package import package_output

    rig = _fake_rig(tmp_path)
    out = tmp_path / "out"
    package_output(rig, {}, out)
    assert (out / "char.moc3").read_bytes() == b"moc3data"
    assert (out / "char.model3.json").exists()


def test_package_output_copies_textures(tmp_path):
    from pipeline.package import package_output

    rig = _fake_rig(tmp_path)
    out = tmp_path / "out"
    package_output(rig, {}, out)
    assert (out / "textures" / "texture_00.png").exists()
    assert (out / "textures" / "texture_01.png").exists()


def test_package_output_writes_modified_atlas(tmp_path):
    from pipeline.package import package_output

    rig = _fake_rig(tmp_path)
    out = tmp_path / "out"
    blue = Image.new("RGBA", (64, 64), color=(0, 0, 255, 255))
    package_output(rig, {0: blue}, out)

    saved = Image.open(out / "textures" / "texture_00.png").convert("RGBA")
    arr = np.array(saved)
    assert np.all(arr[:, :, 2] == 255)  # all pixels blue
    assert np.all(arr[:, :, 0] == 0)    # no red


def test_package_output_idempotent(tmp_path):
    from pipeline.package import package_output

    rig = _fake_rig(tmp_path)
    out = tmp_path / "out"
    package_output(rig, {}, out)
    package_output(rig, {}, out)  # second call should not error
    assert out.is_dir()
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/pipeline/test_package.py -v 2>&1 | head -10
```

- [ ] **Step 3: Write `pipeline/package.py`**

```python
"""Artifact packager — assemble deliverable Live2D model directory."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from rig.config import RigConfig


def package_output(
    rig: "RigConfig",
    modified_atlases: "dict[int, PILImage.Image]",
    output_dir: Path,
) -> Path:
    """Copy rig files into output_dir, replacing atlases with modified versions.

    Writes:
      - <moc3 filename>           (from rig.moc3_path)
      - <model3.json filename>    (from rig.model3_json_path)
      - textures at relative paths (originals, overwritten by modified_atlases entries)

    Does NOT copy model.pt — MLP lives in templates/ and is loaded by muse-vtuber.

    Returns output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(rig.moc3_path, output_dir / rig.moc3_path.name)
    shutil.copy2(rig.model3_json_path, output_dir / rig.model3_json_path.name)

    for idx, tex_path in enumerate(rig.textures):
        rel = tex_path.relative_to(rig.model_dir)
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if idx in modified_atlases:
            modified_atlases[idx].save(dest)
        else:
            shutil.copy2(tex_path, dest)

    return output_dir
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_package.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/package.py tests/pipeline/test_package.py
git commit -m "feat(pipeline): package_output — assemble deliverable Live2D model dir"
```

---

## Task 9: run.py (CLI orchestrator)

**Files:**
- Create: `pipeline/run.py`
- Create: `tests/pipeline/test_run.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_run.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from rig.config import RigConfig


def _fake_rig(tmp_path: Path) -> RigConfig:
    model_dir = tmp_path / "rig"
    tex_dir = model_dir / "textures"
    tex_dir.mkdir(parents=True)
    moc3 = model_dir / "char.moc3"
    moc3.write_bytes(b"data")
    model3 = model_dir / "char.model3.json"
    model3.write_text('{"Version": 3}')
    tex0 = tex_dir / "texture_00.png"
    Image.new("RGBA", (64, 64)).save(tex0)
    return RigConfig(
        name="fake", model_dir=model_dir, moc3_path=moc3,
        model3_json_path=model3, textures=[tex0], param_ids=["P"],
    )


def test_load_atlases_returns_pil_images(tmp_path):
    from pipeline.run import load_atlases

    rig = _fake_rig(tmp_path)
    atlases = load_atlases(rig)

    assert set(atlases.keys()) == {0}
    assert isinstance(atlases[0], Image.Image)
    assert atlases[0].mode == "RGBA"


async def test_run_portrait_to_rig_returns_output_dir(tmp_path):
    from pipeline.atlas_config import AtlasConfig, AtlasRegion
    from pipeline.run import run_portrait_to_rig

    rig = _fake_rig(tmp_path)
    atlas_cfg = AtlasConfig("fake", "t", 64, [AtlasRegion("face_skin", 0, 0, 0, 32, 32)])
    portrait_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (256, 256), color=(200, 150, 100)).save(portrait_path)
    out_dir = tmp_path / "output"
    client = AsyncMock()

    fake_replacements = {"face_skin": Image.new("RGB", (32, 32), color=(180, 130, 90))}

    with patch("pipeline.run.generate_textures", AsyncMock(return_value=fake_replacements)):
        result = await run_portrait_to_rig(
            portrait_path=portrait_path,
            rig_config=rig,
            atlas_cfg=atlas_cfg,
            output_dir=out_dir,
            template_name="humanoid-anime",
            client=client,
        )

    assert result == out_dir
    assert out_dir.is_dir()
    assert (out_dir / "char.moc3").exists()
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/pipeline/test_run.py -v 2>&1 | head -10
```

- [ ] **Step 3: Write `pipeline/run.py`**

```python
"""Top-level orchestrator and CLI for texture generation pipeline.

Usage:
    python -m pipeline.run portrait.jpg \\
        --rig hiyori \\
        --atlas manifests/hiyori_atlas.toml \\
        --out ./output/hiyori_portrait/ \\
        [--template humanoid-anime] \\
        [--comfyui http://127.0.0.1:8188]
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from PIL import Image

from comfyui.client import ComfyUIClient
from pipeline.atlas_config import AtlasConfig, load_atlas_config
from pipeline.package import package_output
from pipeline.texture_gen import generate_textures
from pipeline.texture_swap import swap_regions
from rig.config import RIG_HIYORI, RigConfig


def load_atlases(rig_config: RigConfig) -> dict[int, Image.Image]:
    """Open each texture in rig_config.textures as RGBA PIL Image, keyed by index."""
    return {
        idx: Image.open(tex_path).convert("RGBA")
        for idx, tex_path in enumerate(rig_config.textures)
    }


async def run_portrait_to_rig(
    portrait_path: Path,
    rig_config: RigConfig,
    atlas_cfg: AtlasConfig,
    output_dir: Path,
    template_name: str = "humanoid-anime",
    client: ComfyUIClient | None = None,
) -> Path:
    """Full texture pipeline: portrait -> deliverable Live2D model directory.

    Steps:
      1. generate_textures() -> replacements
      2. load_atlases() -> original atlas dict
      3. swap_regions() -> modified atlas dict
      4. package_output() -> writes output_dir

    Returns output_dir.
    """
    portrait = Image.open(portrait_path).convert("RGB")

    own_client = client is None
    if client is None:
        client = ComfyUIClient()

    try:
        replacements = await generate_textures(
            portrait=portrait,
            atlas_cfg=atlas_cfg,
            rig=rig_config,
            client=client,
            template_name=template_name,
        )
        atlases = load_atlases(rig_config)
        modified_atlases = swap_regions(atlases, atlas_cfg, replacements)
        package_output(rig_config, modified_atlases, output_dir)
    finally:
        if own_client:
            await client.close()

    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portrait-styled Live2D model")
    parser.add_argument("portrait", type=Path)
    parser.add_argument("--rig", default="hiyori", choices=["hiyori"])
    parser.add_argument("--atlas", type=Path, default=Path("manifests/hiyori_atlas.toml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", default="humanoid-anime")
    parser.add_argument("--comfyui", default="http://127.0.0.1:8188")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    rig_map = {"hiyori": RIG_HIYORI}
    rig_config = rig_map[args.rig]
    atlas_cfg = load_atlas_config(args.atlas)

    async with ComfyUIClient(base_url=args.comfyui) as client:
        out = await run_portrait_to_rig(
            portrait_path=args.portrait,
            rig_config=rig_config,
            atlas_cfg=atlas_cfg,
            output_dir=args.out,
            template_name=args.template,
            client=client,
        )
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_run.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Run full unit test suite**

```bash
pytest tests/ -v --ignore=tests/pipeline/test_texture_gen_integration.py
```

Expected: all tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add pipeline/run.py tests/pipeline/test_run.py
git commit -m "feat(pipeline): run.py — CLI connecting texture gen + swap + package"
```

---

## Task 10: Integration test

**Files:**
- Create: `tests/pipeline/test_texture_gen_integration.py`

Skipped if Hiyori model absent or ComfyUI not running.

- [ ] **Step 1: Write the integration test**

Create `tests/pipeline/test_texture_gen_integration.py`:

```python
"""Integration test: portrait -> output dir via full pipeline.

Requires Hiyori model files and ComfyUI running at http://127.0.0.1:8188.
Automatically skipped if either is absent.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HIYORI_MODEL3 = Path(
    "/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam"
    "/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets"
    "/Live2DModels/hiyori_vts/hiyori.model3.json"
)

skip_no_model = pytest.mark.skipif(
    not HIYORI_MODEL3.exists(), reason="Hiyori model files not present"
)


async def _comfyui_running() -> bool:
    try:
        from comfyui.client import ComfyUIClient
        async with ComfyUIClient() as c:
            await c.health()
        return True
    except Exception:
        return False


@skip_no_model
async def test_full_pipeline_hiyori(tmp_path):
    """End-to-end: synthetic portrait -> output dir with modified atlases."""
    if not await _comfyui_running():
        pytest.skip("ComfyUI not running at http://127.0.0.1:8188")

    from pipeline.atlas_config import load_atlas_config
    from pipeline.run import run_portrait_to_rig
    from rig.config import RIG_HIYORI

    portrait_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (512, 512), color=(220, 170, 140)).save(portrait_path)

    atlas_cfg = load_atlas_config(Path("manifests/hiyori_atlas.toml"))
    out_dir = tmp_path / "output"

    result = await run_portrait_to_rig(
        portrait_path=portrait_path,
        rig_config=RIG_HIYORI,
        atlas_cfg=atlas_cfg,
        output_dir=out_dir,
        template_name="humanoid-anime",
    )

    assert result.is_dir()
    assert (result / "hiyori.moc3").exists()
    assert (result / "hiyori.model3.json").exists()
    assert (result / "hiyori.2048" / "texture_00.png").exists()
    assert (result / "hiyori.2048" / "texture_01.png").exists()

    # Atlas was modified (not identical to original)
    original = np.array(Image.open(RIG_HIYORI.textures[0]).convert("RGBA"))
    modified = np.array(Image.open(result / "hiyori.2048" / "texture_00.png").convert("RGBA"))
    assert not np.array_equal(original, modified), (
        "Modified atlas identical to original — swap may not have run"
    )

    print(f"Output at: {result}")
```

- [ ] **Step 2: Verify it skips cleanly without ComfyUI**

```bash
pytest tests/pipeline/test_texture_gen_integration.py -v
```

Expected: `SKIPPED`.

- [ ] **Step 3: Commit**

```bash
git add tests/pipeline/test_texture_gen_integration.py
git commit -m "test(pipeline): integration test — portrait -> output dir with Hiyori"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| `[texture_generation]` in schema.toml | 1 |
| `MediaPipeLandmarkError` | 1 |
| `scripts/generate_template_landmarks.py` + `face_landmarks.json` | 2 |
| `detect_landmarks`, `load_template_landmarks`, `align_portrait` | 3 |
| `compute_affine_transform`, `warp_image`, `crop_region` | 3 |
| `build_face_inpaint_mask` (UV bbox + dilation) | 3 |
| `TextureGenConfig`, `load_texture_gen_config` | 4 |
| `stylize_portrait` (none passthrough + ComfyUI submit) | 4 |
| `style_transfer_anime.json` workflow | 4 |
| `inpaint_face_skin` | 5 |
| `face_inpaint.json` workflow | 5 |
| `segment_hair`, `extract_hair_regions` | 6 |
| `hair_segment.json` workflow | 6 |
| Sparse hair warning | 6 |
| `generate_textures` Phase 1 orchestrator | 7 |
| Face skin inpaint upscale for quality | 7 |
| `package_output` | 8 |
| `load_atlases`, `run_portrait_to_rig`, CLI `__main__` | 9 |
| Integration test end-to-end | 10 |

**Type consistency across tasks:**
- `align_portrait` → `(Image.Image, np.ndarray)`: defined Task 3, consumed Task 7 as `warped, M = align_portrait(...)` ✓
- `warp_image(image, M, output_size)`: defined Task 3, called Task 7 ✓
- `build_face_inpaint_mask(atlas_cfg, warped_size=..., ...)`: defined Task 3, called Task 7 ✓
- `extract_hair_regions(hair_rgba, atlas_cfg, region_names)`: defined Task 6, called Task 7 ✓
- `package_output(rig, modified_atlases, output_dir) -> Path`: defined Task 8, called Task 9 ✓
- `swap_regions(atlases, config, replacements)`: existing function, signature unchanged ✓
