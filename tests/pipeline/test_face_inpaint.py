# tests/pipeline/test_face_inpaint.py
from unittest.mock import AsyncMock

import pytest
from PIL import Image


async def test_inpaint_face_skin_uploads_two_images():
    from pipeline.face_inpaint import inpaint_face_skin

    face_crop = Image.new("RGB", (256, 256), color=(220, 180, 160))
    mask = Image.new("L", (256, 256), 0)
    fake_output = Image.new("RGB", (256, 256), color=(210, 170, 150))

    client = AsyncMock()
    client.upload_image.return_value = "img.png"
    client.submit.return_value = "pid"
    client.wait.return_value = {
        "10": {"images": [{"filename": "p2l_inpaint_00001.png", "subfolder": "", "type": "output"}]}
    }

    async def fake_download(_filename, dest, _subfolder="", _file_type="output"):
        fake_output.save(dest)

    client.download.side_effect = fake_download

    result = await inpaint_face_skin(face_crop, mask, client)

    assert client.upload_image.call_count == 2  # face + mask
    assert isinstance(result, Image.Image)


async def test_inpaint_face_skin_injects_prompt():
    from pipeline.face_inpaint import inpaint_face_skin

    face_crop = Image.new("RGB", (128, 128))
    mask = Image.new("L", (128, 128), 255)
    fake_output = Image.new("RGB", (128, 128))

    captured = {}

    client = AsyncMock()
    client.upload_image.return_value = "img.png"
    client.wait.return_value = {
        "10": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
    }

    async def capture_submit(workflow):
        captured["workflow"] = workflow
        return "pid"

    client.submit.side_effect = capture_submit

    async def fake_download(_filename, dest, _subfolder="", _file_type="output"):
        fake_output.save(dest)

    client.download.side_effect = fake_download

    custom_prompt = "pale alabaster skin"
    await inpaint_face_skin(face_crop, mask, client, prompt=custom_prompt)

    assert custom_prompt in str(captured["workflow"])
