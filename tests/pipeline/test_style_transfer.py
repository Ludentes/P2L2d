from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image


def test_load_texture_gen_config():
    from pipeline.style_transfer import load_texture_gen_config

    cfg = load_texture_gen_config("humanoid-anime")

    assert cfg.style_transfer == "anime_flat_cell"
    assert cfg.style_model == "flux1-krea-dev_fp8_scaled.safetensors"
    assert 0.0 < cfg.style_strength <= 1.0


def test_load_texture_gen_config_missing_template():
    from pipeline.style_transfer import load_texture_gen_config

    with pytest.raises(FileNotFoundError):
        load_texture_gen_config("nonexistent-template")


async def test_stylize_portrait_none_passthrough():
    from pipeline.style_transfer import stylize_portrait

    portrait = Image.new("RGB", (512, 512), color=(200, 100, 50))
    client = MagicMock()

    result = await stylize_portrait(portrait, style="none", model="any", strength=0.5, client=client)

    assert result is portrait
    client.upload_image.assert_not_called()


async def test_stylize_portrait_anime_calls_comfyui():
    from pipeline.style_transfer import stylize_portrait

    portrait = Image.new("RGB", (512, 512), color=(200, 100, 50))
    fake_output = Image.new("RGB", (512, 512), color=(50, 150, 200))

    client = AsyncMock()
    client.upload_image.return_value = "portrait_123.png"
    client.submit.return_value = "prompt-abc"
    client.wait.return_value = {
        "11": {"images": [{"filename": "p2l_style_00001.png", "subfolder": "", "type": "output"}]}
    }

    async def fake_download(_filename, dest, _subfolder="", _file_type="output"):
        fake_output.save(dest)

    client.download.side_effect = fake_download

    result = await stylize_portrait(
        portrait, style="anime_flat_cell", model="noobai-xl", strength=0.65, client=client
    )

    client.upload_image.assert_called_once()
    client.submit.assert_called_once()
    client.wait.assert_called_once_with("prompt-abc")
    client.download.assert_called_once()
    assert isinstance(result, Image.Image)
