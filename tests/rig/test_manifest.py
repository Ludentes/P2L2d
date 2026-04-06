"""Tests for rig.manifest — per-rig param name mapping."""
from pathlib import Path

import pytest

from rig.manifest import Manifest, load_manifest


def test_load_manifest():
    m = load_manifest(Path("manifests/hiyori.toml"))
    assert m.template_name == "humanoid-anime"
    assert "ParamAngleX" in m.param_map


def test_manifest_remap_forward():
    m = load_manifest(Path("manifests/hiyori.toml"))
    template_output = {"AngleX": -5.2, "MouthOpenY": 0.73}
    rig_output = m.remap(template_output)
    assert rig_output == {"ParamAngleX": -5.2, "ParamMouthOpenY": 0.73}


def test_manifest_remap_drops_unmapped():
    m = Manifest(template_name="t", param_map={"ParamX": "TemplateX"})
    result = m.remap({"TemplateX": 1.0, "TemplateY": 2.0})
    assert result == {"ParamX": 1.0}
    assert "TemplateY" not in result


def test_manifest_remap_empty_input():
    m = Manifest(template_name="t", param_map={"ParamX": "TemplateX"})
    assert m.remap({}) == {}


def test_manifest_all_13_params_mapped():
    m = load_manifest(Path("manifests/hiyori.toml"))
    expected = [
        "AngleX", "AngleY", "AngleZ", "EyeLOpen", "EyeROpen",
        "EyeBallX", "EyeLSmile", "EyeRSmile", "BrowLY", "BrowRY",
        "MouthOpenY", "MouthForm", "Cheek",
    ]
    mapped = set(m.param_map.values())
    for p in expected:
        assert p in mapped, f"Template param {p!r} not mapped in manifest"


def test_load_manifest_missing_file():
    with pytest.raises(FileNotFoundError):
        load_manifest(Path("manifests/nonexistent.toml"))
