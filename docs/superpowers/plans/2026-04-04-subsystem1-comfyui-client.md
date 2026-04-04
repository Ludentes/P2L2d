# ComfyUI HTTP Client — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin async httpx client (`comfyui/client.py`) wrapping six ComfyUI REST API endpoints, with typed exceptions and a manual smoke-test script.

**Architecture:** Single `ComfyUIClient` class backed by a persistent `httpx.AsyncClient`. `wait()` polls `GET /history/{prompt_id}` at a configurable interval using `time.monotonic()` for deadline tracking. Four typed exception classes surface all error conditions. Tests mock the HTTP transport layer with `respx`. This is Subsystem 1 of 5 — remaining plans in order: 2 (portrait pipeline), 4 (BCI injection), 3 (MLP), 5 (runtime bridge).

**Tech Stack:** Python 3.12, uv, httpx ≥0.27, pytest ≥8.0, pytest-asyncio ≥0.23, respx ≥0.21

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | Create | uv project config, runtime + dev deps, pytest config |
| `comfyui/__init__.py` | Create | Re-exports `ComfyUIClient` and all exception types |
| `comfyui/exceptions.py` | Create | `ComfyUIError`, `ComfyUIConnectionError`, `ComfyUIJobError`, `ComfyUITimeoutError` |
| `comfyui/client.py` | Create | `ComfyUIClient` with 6 async methods |
| `comfyui/workflows/.gitkeep` | Create | Keeps directory tracked in git |
| `comfyui/workflows/README.md` | Create | How to export API-format workflows from ComfyUI GUI |
| `comfyui/test_connection.py` | Create | Manual smoke test against a live ComfyUI instance |
| `tests/__init__.py` | Create | Test package root |
| `tests/comfyui/__init__.py` | Create | Test sub-package |
| `tests/comfyui/test_client.py` | Create | Unit tests for all 6 methods (respx mocks) |

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/comfyui/__init__.py`

- [ ] **Step 1: Initialise uv project**

```bash
cd /home/newub/w/portrait-to-live2d
uv init --no-readme .
rm -f hello.py  # remove generated stub
```

- [ ] **Step 2: Add dependencies**

```bash
uv add httpx
uv add --dev pytest pytest-asyncio respx
```

- [ ] **Step 3: Add pytest config to `pyproject.toml`**

Append this block to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

The final `pyproject.toml` should look like:

```toml
[project]
name = "portrait-to-live2d"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]
```

- [ ] **Step 4: Create test package skeleton**

```bash
mkdir -p tests/comfyui
touch tests/__init__.py tests/comfyui/__init__.py
```

- [ ] **Step 5: Verify pytest runs with no errors**

```bash
uv run pytest --collect-only
```

Expected output contains:
```
========================= no tests ran =========================
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version tests/
git commit -m "chore: init uv project with httpx, pytest-asyncio, respx"
```

---

### Task 2: Exception types

**Files:**
- Create: `comfyui/exceptions.py`
- Create: `comfyui/__init__.py` (stub)
- Test: `tests/comfyui/test_client.py`

- [ ] **Step 1: Create `comfyui/exceptions.py`**

```python
class ComfyUIError(Exception):
    """Base exception for all ComfyUI client errors."""


class ComfyUIConnectionError(ComfyUIError):
    """Cannot reach the ComfyUI server."""


class ComfyUIJobError(ComfyUIError):
    """ComfyUI reported an error for this job or workflow."""


class ComfyUITimeoutError(ComfyUIError):
    """Polling timed out before the job completed."""
```

- [ ] **Step 2: Create `comfyui/__init__.py`** (stub — grows as client is added)

```python
from comfyui.exceptions import (
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUIJobError,
    ComfyUITimeoutError,
)

__all__ = [
    "ComfyUIConnectionError",
    "ComfyUIError",
    "ComfyUIJobError",
    "ComfyUITimeoutError",
]
```

- [ ] **Step 3: Write failing test for exception hierarchy**

Create `tests/comfyui/test_client.py`:

```python
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
```

- [ ] **Step 4: Run to confirm it fails**

```bash
uv run pytest tests/comfyui/test_client.py::test_exception_hierarchy -v
```

Expected: `ImportError: cannot import name 'ComfyUIClient' from 'comfyui'`

- [ ] **Step 5: Commit exceptions (client comes in next task)**

```bash
git add comfyui/
git commit -m "feat: add ComfyUI exception types"
```

---

### Task 3: Client skeleton — init, close, context manager

**Files:**
- Create: `comfyui/client.py`
- Modify: `comfyui/__init__.py`
- Test: `tests/comfyui/test_client.py`

- [ ] **Step 1: Append failing tests to `tests/comfyui/test_client.py`**

```python
def test_client_instantiates_with_default_url():
    client = ComfyUIClient()
    assert client._base_url == "http://127.0.0.1:8188"


def test_client_instantiates_with_custom_url():
    client = ComfyUIClient("http://10.0.0.5:9999")
    assert client._base_url == "http://10.0.0.5:9999"


async def test_client_context_manager():
    async with ComfyUIClient() as client:
        assert client._base_url == "http://127.0.0.1:8188"
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/comfyui/test_client.py -k "instantiate or context_manager" -v
```

Expected: `ImportError: cannot import name 'ComfyUIClient' from 'comfyui'`

- [ ] **Step 3: Create `comfyui/client.py`**

```python
import httpx


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
```

- [ ] **Step 4: Re-export `ComfyUIClient` from `comfyui/__init__.py`**

Replace the contents of `comfyui/__init__.py` with:

```python
from comfyui.client import ComfyUIClient
from comfyui.exceptions import (
    ComfyUIConnectionError,
    ComfyUIError,
    ComfyUIJobError,
    ComfyUITimeoutError,
)

__all__ = [
    "ComfyUIClient",
    "ComfyUIConnectionError",
    "ComfyUIError",
    "ComfyUIJobError",
    "ComfyUITimeoutError",
]
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/comfyui/test_client.py -k "instantiate or context_manager or hierarchy" -v
```

Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add comfyui/client.py comfyui/__init__.py tests/comfyui/test_client.py
git commit -m "feat: add ComfyUIClient skeleton with async context manager"
```

---

### Task 4: `health()` — GET /system_stats

**Files:**
- Modify: `comfyui/client.py`
- Modify: `tests/comfyui/test_client.py`

- [ ] **Step 1: Append failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/comfyui/test_client.py -k "health" -v
```

Expected: `AttributeError: 'ComfyUIClient' object has no attribute 'health'`

- [ ] **Step 3: Implement `health()` — add to `comfyui/client.py`**

Add import at top:
```python
from comfyui.exceptions import ComfyUIConnectionError, ComfyUIJobError, ComfyUITimeoutError
```

Add method inside `ComfyUIClient`:

```python
    async def health(self) -> dict:
        try:
            response = await self._http.get("/system_stats")
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(str(exc)) from exc
        return response.json()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/comfyui/test_client.py -k "health" -v
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add comfyui/client.py tests/comfyui/test_client.py
git commit -m "feat: add ComfyUIClient.health()"
```

---

### Task 5: `list_models()` — GET /models/{folder}

**Files:**
- Modify: `comfyui/client.py`
- Modify: `tests/comfyui/test_client.py`

- [ ] **Step 1: Append failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/comfyui/test_client.py -k "list_models" -v
```

Expected: `AttributeError: 'ComfyUIClient' object has no attribute 'list_models'`

- [ ] **Step 3: Implement `list_models()` — add to `comfyui/client.py`**

```python
    async def list_models(self, folder: str) -> list[str]:
        try:
            response = await self._http.get(f"/models/{folder}")
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(str(exc)) from exc
        return response.json()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/comfyui/test_client.py -k "list_models" -v
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add comfyui/client.py tests/comfyui/test_client.py
git commit -m "feat: add ComfyUIClient.list_models()"
```

---

### Task 6: `upload_image()` — POST /upload/image

**Files:**
- Modify: `comfyui/client.py`
- Modify: `tests/comfyui/test_client.py`

- [ ] **Step 1: Append failing tests**

```python
import tempfile
from pathlib import Path


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
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/comfyui/test_client.py -k "upload_image" -v
```

Expected: `AttributeError: 'ComfyUIClient' object has no attribute 'upload_image'`

- [ ] **Step 3: Implement `upload_image()` — add to `comfyui/client.py`**

Add to imports at top:
```python
from pathlib import Path
```

Add method:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/comfyui/test_client.py -k "upload_image" -v
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add comfyui/client.py tests/comfyui/test_client.py
git commit -m "feat: add ComfyUIClient.upload_image()"
```

---

### Task 7: `submit()` — POST /prompt

**Files:**
- Modify: `comfyui/client.py`
- Modify: `tests/comfyui/test_client.py`

- [ ] **Step 1: Append failing tests**

```python
SAMPLE_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {"seed": 42, "steps": 20},
    }
}


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
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/comfyui/test_client.py -k "submit" -v
```

Expected: `AttributeError: 'ComfyUIClient' object has no attribute 'submit'`

- [ ] **Step 3: Implement `submit()` — add to `comfyui/client.py`**

Add to imports at top:
```python
import uuid
```

Add method:

```python
    async def submit(self, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        try:
            response = await self._http.post("/prompt", json=body)
        except httpx.ConnectError as exc:
            raise ComfyUIConnectionError(str(exc)) from exc
        data = response.json()
        if "error" in data or "node_errors" in data:
            raise ComfyUIJobError(
                f"Workflow validation failed: "
                f"{data.get('error') or data.get('node_errors')}"
            )
        return data["prompt_id"]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/comfyui/test_client.py -k "submit" -v
```

Expected: all 3 PASS

- [ ] **Step 5: Commit**

```bash
git add comfyui/client.py tests/comfyui/test_client.py
git commit -m "feat: add ComfyUIClient.submit()"
```

---

### Task 8: `wait()` — poll GET /history/{prompt_id}

**Files:**
- Modify: `comfyui/client.py`
- Modify: `tests/comfyui/test_client.py`

The `/history/{prompt_id}` endpoint returns `{}` while a job is queued/running, then the full job dict (with `status.completed == True`) once done. `wait()` polls until the key appears and is complete.

- [ ] **Step 1: Append failing tests**

```python
import asyncio as _asyncio

SAMPLE_OUTPUTS = {
    "9": {
        "images": [
            {"filename": "out_00001.png", "subfolder": "", "type": "output"}
        ]
    }
}


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
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/comfyui/test_client.py -k "wait" -v
```

Expected: `AttributeError: 'ComfyUIClient' object has no attribute 'wait'`

- [ ] **Step 3: Implement `wait()` — add to `comfyui/client.py`**

Add to imports at top:
```python
import asyncio
import time
```

Add method:

```python
    async def wait(
        self,
        prompt_id: str,
        timeout: float = 120.0,
        poll_interval: float = 0.5,
    ) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            response = await self._http.get(f"/history/{prompt_id}")
            response.raise_for_status()
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
            if time.monotonic() >= deadline:
                raise ComfyUITimeoutError(
                    f"Job {prompt_id!r} timed out after {timeout}s"
                )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/comfyui/test_client.py -k "wait" -v
```

Expected: all 4 PASS. The timeout test takes ~50ms wall-clock — that's expected.

- [ ] **Step 5: Commit**

```bash
git add comfyui/client.py tests/comfyui/test_client.py
git commit -m "feat: add ComfyUIClient.wait()"
```

---

### Task 9: `download()` — GET /view (streaming)

**Files:**
- Modify: `comfyui/client.py`
- Modify: `tests/comfyui/test_client.py`

- [ ] **Step 1: Append failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/comfyui/test_client.py -k "download" -v
```

Expected: `AttributeError: 'ComfyUIClient' object has no attribute 'download'`

- [ ] **Step 3: Implement `download()` — add to `comfyui/client.py`**

```python
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
```

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS with no warnings about asyncio mode

- [ ] **Step 5: Commit**

```bash
git add comfyui/client.py tests/comfyui/test_client.py
git commit -m "feat: add ComfyUIClient.download()"
```

---

### Task 10: Smoke test + workflows directory

**Files:**
- Create: `comfyui/test_connection.py`
- Create: `comfyui/workflows/.gitkeep`
- Create: `comfyui/workflows/README.md`

- [ ] **Step 1: Create `comfyui/test_connection.py`**

This is a manual script — run it only when ComfyUI is live at port 8188.

```python
#!/usr/bin/env python
"""
Manual smoke test: verifies ComfyUI is reachable and lists available checkpoints.
Run with: uv run comfyui/test_connection.py
Requires ComfyUI running at http://127.0.0.1:8188.
"""
import asyncio
import sys

from comfyui import ComfyUIClient, ComfyUIConnectionError


async def main() -> None:
    async with ComfyUIClient() as client:
        print("Checking ComfyUI health ... ", end="", flush=True)
        try:
            stats = await client.health()
        except ComfyUIConnectionError as exc:
            print(f"FAILED\n  {exc}")
            sys.exit(1)
        print("OK")
        print(f"  Python : {stats.get('python_version', 'unknown')}")
        for device in stats.get("devices", []):
            vram_gb = device.get("vram_total", 0) // 1024 ** 3
            print(f"  GPU    : {device.get('name', '?')} ({vram_gb} GB)")

        print("\nListing checkpoints ... ", end="", flush=True)
        checkpoints = await client.list_models("checkpoints")
        print(f"found {len(checkpoints)}")
        for name in checkpoints[:5]:
            print(f"  {name}")
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Create `comfyui/workflows/README.md`**

```markdown
# ComfyUI Workflow Files

Workflow JSON files in this directory are in **API format** — not the default UI save format.

## How to export API format from the ComfyUI GUI

1. Open ComfyUI in your browser (`http://127.0.0.1:8188`)
2. Click the gear icon ⚙️ → **Dev Mode Options** → enable **"Enable Dev mode Options"**
3. Load or build the workflow you want to save
4. Click **"Save (API Format)"** — downloads a JSON with a flat node dict

## API format overview

Keys are node IDs (numeric strings). Links between nodes use `["source_node_id", output_index]`:

```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 20,
      "model": ["4", 0]
    }
  },
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {"ckpt_name": "flux1-dev-kontext_fp8_scaled.safetensors"}
  }
}
```

## Naming convention

`<purpose>[-<model-hint>].json` — examples:
- `flux-kontext-portrait.json`
- `flux-kontext-component-inpaint.json`
- `sdxl-anime-flat.json`
```

- [ ] **Step 3: Create workflows directory sentinel**

```bash
touch comfyui/workflows/.gitkeep
```

- [ ] **Step 4: Run full test suite one final time**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add comfyui/test_connection.py comfyui/workflows/
git commit -m "feat: add smoke test and workflows README"
```

---

## Self-Review

**Spec coverage:**
- [x] `health()` → Task 4
- [x] `list_models()` → Task 5
- [x] `upload_image()` → Task 6
- [x] `submit()` → Task 7
- [x] `wait()` → Task 8
- [x] `download()` → Task 9
- [x] `pyproject.toml` with uv + httpx → Task 1
- [x] `test_connection.py` smoke test → Task 10
- [x] `comfyui/workflows/README.md` → Task 10
- [x] `ComfyUIConnectionError` on server unreachable → Tasks 4–9
- [x] `ComfyUIJobError` on workflow/job errors → Tasks 7, 8
- [x] `ComfyUITimeoutError` on poll timeout → Task 8
- [x] Typed exception hierarchy → Task 2

**Placeholder scan:** No TBDs, TODOs, or vague instructions.

**Type consistency:**
- `ComfyUIClient` — consistent across all tasks
- `wait(prompt_id, timeout=120.0, poll_interval=0.5)` — `poll_interval` param introduced in Task 8 step 1 (test) and step 3 (impl) simultaneously; no forward-reference issues
- `download(filename, dest, subfolder="", file_type="output")` — `file_type` (not `type`) used consistently in both test (Task 9 step 1) and implementation (step 3); avoids shadowing the Python builtin
- All exception types imported in tests at the top of `test_client.py` from `comfyui` (not `comfyui.exceptions`)
