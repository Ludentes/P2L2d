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
    from rig.config import RIG_HIYORI
    from pipeline.validate import validate_textures

    tex_dir = HIYORI_MODEL3.parent / "hiyori.2048"
    atlases = {
        0: Image.open(tex_dir / "texture_00.png").convert("RGBA"),
        1: Image.open(tex_dir / "texture_01.png").convert("RGBA"),
    }

    frame = validate_textures(RIG_HIYORI, atlases)
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 4


@skip_no_model
def test_validate_face_color_replacement():
    """Paint face_skin red → red pixels appear in rendered frame."""
    from rig.config import RIG_HIYORI
    from pipeline.validate import validate_textures, check_region_color
    from pipeline.atlas_config import load_atlas_config
    from pipeline.texture_swap import swap_regions

    atlas_cfg = load_atlas_config(Path("manifests/hiyori_atlas.toml"))

    tex_dir = HIYORI_MODEL3.parent / "hiyori.2048"
    atlases = {
        0: Image.open(tex_dir / "texture_00.png").convert("RGBA"),
        1: Image.open(tex_dir / "texture_01.png").convert("RGBA"),
    }

    red = Image.new("RGBA", (532, 603), (255, 0, 0, 255))
    modified = swap_regions(atlases, atlas_cfg, {"face_skin": red})
    frame = validate_textures(RIG_HIYORI, modified)
    assert check_region_color(frame, (255, 0, 0), tolerance=30)
