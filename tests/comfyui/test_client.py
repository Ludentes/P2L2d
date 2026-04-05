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


@respx.mock
async def test_wait_returns_outputs_when_already_complete():
    respx.get("http://127.0.0.1:8188/history/abc-123").mock(
        return_value=_httpx.Response(
            200,
            json={
                "abc-123": {
                    "outputs": SAMPLE_OUTPUTS,
                    "status": {"status_str": "success", "completed": True},
                }
            },
        )
    )
    async with ComfyUIClient() as client:
        result = await client.wait("abc-123")
    assert result == SAMPLE_OUTPUTS


@respx.mock
async def test_wait_polls_until_job_appears():
    call_count = 0

    async def history_side_effect(request: _httpx.Request) -> _httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _httpx.Response(200, json={})  # not ready yet
        return _httpx.Response(
            200,
            json={
                "abc-456": {
                    "outputs": SAMPLE_OUTPUTS,
                    "status": {"status_str": "success", "completed": True},
                }
            },
        )

    respx.get("http://127.0.0.1:8188/history/abc-456").mock(
        side_effect=history_side_effect
    )
    async with ComfyUIClient() as client:
        result = await client.wait("abc-456", poll_interval=0.0)
    assert result == SAMPLE_OUTPUTS
    assert call_count == 2


@respx.mock
async def test_wait_raises_on_job_error():
    respx.get("http://127.0.0.1:8188/history/abc-789").mock(
        return_value=_httpx.Response(
            200,
            json={
                "abc-789": {
                    "outputs": {},
                    "status": {"status_str": "error", "completed": True},
                }
            },
        )
    )
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUIJobError):
            await client.wait("abc-789")


@respx.mock
async def test_wait_raises_on_timeout():
    # Returns "not done" on every call — timeout is 50ms, poll every 10ms
    respx.get("http://127.0.0.1:8188/history/abc-999").mock(
        return_value=_httpx.Response(200, json={})
    )
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUITimeoutError):
            await client.wait("abc-999", timeout=0.05, poll_interval=0.01)


@respx.mock
async def test_download_writes_bytes_to_file(tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    respx.get("http://127.0.0.1:8188/view").mock(
        return_value=_httpx.Response(200, content=image_bytes)
    )
    dest = tmp_path / "out.png"
    async with ComfyUIClient() as client:
        await client.download("out_00001.png", dest)
    assert dest.read_bytes() == image_bytes


@respx.mock
async def test_download_creates_parent_directories(tmp_path):
    respx.get("http://127.0.0.1:8188/view").mock(
        return_value=_httpx.Response(200, content=b"\x89PNG\r\n\x1a\n")
    )
    dest = tmp_path / "deep" / "nested" / "out.png"
    async with ComfyUIClient() as client:
        await client.download("out_00001.png", dest)
    assert dest.exists()


@respx.mock
async def test_download_raises_on_connection_error(tmp_path):
    respx.get("http://127.0.0.1:8188/view").mock(
        side_effect=_httpx.ConnectError("connection refused")
    )
    async with ComfyUIClient() as client:
        with pytest.raises(ComfyUIConnectionError):
            await client.download("out.png", tmp_path / "out.png")
