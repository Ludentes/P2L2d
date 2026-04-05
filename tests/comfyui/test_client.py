import pytest
import httpx as _httpx
import respx
import tempfile
import asyncio as _asyncio
from pathlib import Path

from comfyui import (
    ComfyUIClient,
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUIJobError,
    ComfyUITimeoutError,
)


SAMPLE_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {"seed": 42, "steps": 20},
    }
}


SAMPLE_OUTPUTS = {
    "9": {
        "images": [
            {"filename": "out_00001.png", "subfolder": "", "type": "output"}
        ]
    }
}


def test_exception_hierarchy():
    assert issubclass(ComfyUIConnectionError, ComfyUIError)
    assert issubclass(ComfyUIJobError, ComfyUIError)
    assert issubclass(ComfyUITimeoutError, ComfyUIError)


def test_client_instantiates_with_default_url():
    client = ComfyUIClient()
    assert client._base_url == "http://127.0.0.1:8188"


def test_client_instantiates_with_custom_url():
    client = ComfyUIClient("http://10.0.0.5:9999")
    assert client._base_url == "http://10.0.0.5:9999"


async def test_client_context_manager():
    async with ComfyUIClient() as client:
        assert client._base_url == "http://127.0.0.1:8188"


@respx.mock
async def test_health_returns_system_stats():
    respx.get("http://127.0.0.1:8188/system_stats").mock(
        return_value=_httpx.Response(200, json={"system": {"os": "posix"}, "devices": []})
    )
    async with ComfyUIClient() as client:
        result = await client.health()
    assert result == {"system": {"os": "posix"}, "devices": []}


@respx.mock
async def test_health_raises_on_connection_error():
    respx.get("http://127.0.0.1:8188/system_stats").mock(
        side_effect=_httpx.ConnectError("connection refused")
    )
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUIConnectionError):
            await client.health()


@respx.mock
async def test_list_models_returns_filenames():
    respx.get("http://127.0.0.1:8188/models/checkpoints").mock(
        return_value=_httpx.Response(
            200, json=["flux1-dev.safetensors", "sdxl.safetensors"]
        )
    )
    async with ComfyUIClient() as client:
        result = await client.list_models("checkpoints")
    assert result == ["flux1-dev.safetensors", "sdxl.safetensors"]


@respx.mock
async def test_list_models_raises_on_connection_error():
    respx.get("http://127.0.0.1:8188/models/loras").mock(
        side_effect=_httpx.ConnectError("connection refused")
    )
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUIConnectionError):
            await client.list_models("loras")


@respx.mock
async def test_upload_image_returns_server_filename():
    respx.post("http://127.0.0.1:8188/upload/image").mock(
        return_value=_httpx.Response(
            200,
            json={"name": "portrait_0001.png", "subfolder": "", "type": "input"},
        )
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        tmp_path = Path(f.name)
    async with ComfyUIClient() as client:
        result = await client.upload_image(tmp_path)
    tmp_path.unlink()
    assert result == "portrait_0001.png"


@respx.mock
async def test_upload_image_raises_on_connection_error():
    respx.post("http://127.0.0.1:8188/upload/image").mock(
        side_effect=_httpx.ConnectError("connection refused")
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        tmp_path = Path(f.name)
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUIConnectionError):
            await client.upload_image(tmp_path)
    tmp_path.unlink()


@respx.mock
async def test_submit_returns_prompt_id():
    respx.post("http://127.0.0.1:8188/prompt").mock(
        return_value=_httpx.Response(
            200, json={"prompt_id": "abc-123", "number": 1}
        )
    )
    async with ComfyUIClient() as client:
        result = await client.submit(SAMPLE_WORKFLOW)
    assert result == "abc-123"


@respx.mock
async def test_submit_raises_on_node_errors():
    respx.post("http://127.0.0.1:8188/prompt").mock(
        return_value=_httpx.Response(
            200,
            json={
                "error": {"type": "prompt_no_outputs", "message": "no output nodes"},
                "node_errors": {"3": ["missing required input"]},
            },
        )
    )
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUIJobError):
            await client.submit(SAMPLE_WORKFLOW)


@respx.mock
async def test_submit_raises_on_connection_error():
    respx.post("http://127.0.0.1:8188/prompt").mock(
        side_effect=_httpx.ConnectError("connection refused")
    )
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUIConnectionError):
            await client.submit(SAMPLE_WORKFLOW)
