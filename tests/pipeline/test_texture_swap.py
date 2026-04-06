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
