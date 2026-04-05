import httpx
from pathlib import Path

from comfyui.exceptions import ComfyUIConnectionError, ComfyUIJobError, ComfyUITimeoutError


class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "ComfyUIClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def health(self) -> dict:
        try:
            response = await self._http.get("/system_stats")
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(str(exc)) from exc
        return response.json()

    async def list_models(self, folder: str) -> list[str]:
        try:
            response = await self._http.get(f"/models/{folder}")
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(str(exc)) from exc
        return response.json()

    async def upload_image(self, path: Path) -> str:
        with open(path, "rb") as fh:
            try:
                response = await self._http.post(
                    "/upload/image",
                    files={"image": (path.name, fh, "image/png")},
                    data={"type": "input", "overwrite": "true"},
                )
                response.raise_for_status()
            except httpx.ConnectError as exc:
                raise ComfyUIConnectionError(str(exc)) from exc
        return response.json()["name"]
