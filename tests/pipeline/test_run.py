from pathlib import Path
from unittest.mock import AsyncMock, patch

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
