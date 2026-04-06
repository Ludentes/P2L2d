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


def test_load_atlas_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_atlas_config(tmp_path / "nonexistent_atlas.toml")


def test_load_atlas_config_invalid_texture_size(tmp_path):
    bad_toml = """\
rig = "test"
template = "humanoid-anime"
texture_size = 0
"""
    p = tmp_path / "bad.toml"
    p.write_text(bad_toml)
    with pytest.raises(ValueError, match="texture_size"):
        load_atlas_config(p)


def test_atlas_region_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        AtlasRegion(name="face_skin", texture_index=0, x=0, y=0, w=0, h=100)
    with pytest.raises(ValueError, match="positive"):
        AtlasRegion(name="face_skin", texture_index=0, x=0, y=0, w=100, h=-1)
