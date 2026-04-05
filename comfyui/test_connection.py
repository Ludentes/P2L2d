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
