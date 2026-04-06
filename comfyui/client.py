import asyncio
import mimetypes
import time
import uuid
from pathlib import Path

import httpx

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
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            try:
                response = await self._http.post(
                    "/upload/image",
                    files={"image": (path.name, fh, mime_type)},
                    data={"type": "input", "overwrite": "true"},
                )
                response.raise_for_status()
            except httpx.ConnectError as exc:
                raise ComfyUIConnectionError(str(exc)) from exc
        return response.json()["name"]

    async def submit(self, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        try:
            response = await self._http.post("/prompt", json=body)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(str(exc)) from exc
        data = response.json()
        if data.get("error") or data.get("node_errors"):
            raise ComfyUIJobError(
                f"Workflow validation failed: "
                f"{data.get('error') or data.get('node_errors')}"
            )
        return data["prompt_id"]

    async def wait(
        self,
        prompt_id: str,
        timeout: float = 120.0,
        poll_interval: float = 0.5,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise ComfyUITimeoutError(
                    f"Job {prompt_id!r} timed out after {timeout}s"
                )
            try:
                response = await self._http.get(f"/history/{prompt_id}")
                response.raise_for_status()
            except httpx.ConnectError as exc:
                raise ComfyUIConnectionError(str(exc)) from exc
            data = response.json()
            if prompt_id in data:
                job = data[prompt_id]
                status = job.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIJobError(
                        f"Job {prompt_id!r} failed: {status}"
                    )
                if status.get("completed"):
                    return job["outputs"]
            await asyncio.sleep(poll_interval)

    async def download(
        self,
        filename: str,
        dest: Path,
        subfolder: str = "",
        file_type: str = "output",
    ) -> None:
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        try:
            async with self._http.stream("GET", "/view", params=params) as response:
                response.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    async for chunk in response.aiter_bytes():
                        fh.write(chunk)
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(str(exc)) from exc
