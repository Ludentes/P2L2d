import pytest
import httpx as _httpx
import respx

from comfyui import (
    ComfyUIClient,
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUIJobError,
    ComfyUITimeoutError,
)


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
