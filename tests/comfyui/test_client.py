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
