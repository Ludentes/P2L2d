"""Tests for template palette extraction from atlas textures."""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

HIYORI_TEX_DIR = Path(
    "/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam"
    "/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets"
    "/Live2DModels/hiyori_vts/hiyori.2048"
)
ATLAS_TOML = Path("manifests/hiyori_atlas.toml")

skip_no_atlas = pytest.mark.skipif(
    not HIYORI_TEX_DIR.exists() or not ATLAS_TOML.exists(),
    reason="Hiyori atlas textures or config not present",
)


def _load_hiyori():
    from pipeline.atlas_config import load_atlas_config

    atlas_config = load_atlas_config(ATLAS_TOML)
    atlases = {
        0: Image.open(HIYORI_TEX_DIR / "texture_00.png").convert("RGBA"),
        1: Image.open(HIYORI_TEX_DIR / "texture_01.png").convert("RGBA"),
    }
    return atlases, atlas_config


class TestTemplatePalette:
    @skip_no_atlas
    def test_extract_from_hiyori(self):
        """Should return valid ColorPalette with correct shapes."""
        from pipeline.template_palette import extract_template_palette

        atlases, atlas_config = _load_hiyori()
        palette = extract_template_palette(atlases, atlas_config)

        # LAB arrays should be shape (3,)
        assert palette.hair.shape == (3,), f"hair shape: {palette.hair.shape}"
        assert palette.skin.shape == (3,), f"skin shape: {palette.skin.shape}"
        assert palette.lip_color.shape == (3,), f"lip shape: {palette.lip_color.shape}"
        assert palette.clothing.shape == (3,), f"clothing shape: {palette.clothing.shape}"

        # LAB values in valid range (0-255)
        for name, arr in [("hair", palette.hair), ("skin", palette.skin),
                          ("lip", palette.lip_color), ("clothing", palette.clothing)]:
            assert np.all(arr >= 0) and np.all(arr <= 255), f"{name} out of range: {arr}"

        # Eye hue in valid range (0-180)
        assert 0 <= palette.eye_color <= 180, f"eye_color: {palette.eye_color}"
        # Eye saturation in valid range (0-255)
        assert 0 <= palette.eye_saturation <= 255, f"eye_sat: {palette.eye_saturation}"

    @skip_no_atlas
    def test_hair_is_brownish(self):
        """Hiyori's hair is dark brown -- L should be moderate (30-200)."""
        from pipeline.template_palette import extract_template_palette

        atlases, atlas_config = _load_hiyori()
        palette = extract_template_palette(atlases, atlas_config)

        # L channel: not pure black, not pure white
        assert 30 <= palette.hair[0] <= 200, f"hair L={palette.hair[0]}"
