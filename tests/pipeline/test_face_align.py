import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion


def _make_landmarks_file(tmp_path: Path, size: int = 512) -> Path:
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


def test_warp_image_output_size():
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
    assert arr.max() == 255
    assert arr.min() == 0
