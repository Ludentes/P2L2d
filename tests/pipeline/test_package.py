# tests/pipeline/test_package.py
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
