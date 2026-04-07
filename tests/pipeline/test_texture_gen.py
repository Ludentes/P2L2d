# tests/pipeline/test_texture_gen.py
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image

from pipeline.atlas_config import AtlasConfig, AtlasRegion
from rig.config import RIG_HIYORI


def _atlas():
    return AtlasConfig("t", "t", 2048, [
        AtlasRegion("face_skin", 0, 0, 0, 532, 603),
        AtlasRegion("left_eye", 0, 24, 647, 171, 107),
        AtlasRegion("right_eye", 0, 326, 640, 155, 110),
        AtlasRegion("left_eyebrow", 0, 15, 574, 148, 71),
        AtlasRegion("right_eyebrow", 0, 398, 572, 133, 67),
        AtlasRegion("mouth", 0, 219, 744, 78, 50),
        AtlasRegion("hair_front", 0, 549, 25, 500, 438),
        AtlasRegion("hair_back", 0, 1164, 856, 869, 1166),
        AtlasRegion("hair_side_left", 0, 664, 451, 162, 191),
        AtlasRegion("hair_side_right", 0, 1028, 585, 97, 541),
    ])


async def test_generate_textures_returns_expected_regions():
    from pipeline.texture_gen import generate_textures

    portrait = Image.new("RGB", (512, 512), color=(200, 150, 100))
    atlas_cfg = _atlas()
    client = AsyncMock()

    fake_img = Image.new("RGB", (512, 512), color=(180, 130, 90))
    fake_rgba = Image.new("RGBA", (512, 512), color=(80, 60, 40, 200))
    fake_landmarks = np.random.rand(478, 2).astype(np.float32) * 400 + 50

    with (
        patch("pipeline.texture_gen.detect_landmarks", return_value=fake_landmarks),
        patch("pipeline.texture_gen.stylize_portrait", AsyncMock(return_value=fake_img)),
        patch("pipeline.texture_gen.inpaint_face_skin", AsyncMock(return_value=fake_img)),
        patch("pipeline.texture_gen.segment_hair", AsyncMock(return_value=fake_rgba)),
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
