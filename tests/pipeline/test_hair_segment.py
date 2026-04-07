# tests/pipeline/test_hair_segment.py
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
        "4": {"images": [{"filename": "hair_seg.png", "subfolder": "", "type": "output"}]}
    }

    async def fake_download(_filename, dest, _subfolder="", _file_type="output"):
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
