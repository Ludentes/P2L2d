"""Face tracking → VTube Studio parameter bridge.

Connects MediaPipe webcam stream → CartoonAlive MLP → VTS InjectParameterData
at ~30fps.  Runs as a standalone asyncio task.

VTS WebSocket: ws://localhost:8001
Auth token cached at ~/.config/portrait-to-live2d/vts_token.txt
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import websockets

from mlp.infer import Predictor
from rig.config import RigConfig
from runtime.mediapipe_stream import landmark_stream

log = logging.getLogger(__name__)

_TOKEN_FILE = Path.home() / ".config" / "portrait-to-live2d" / "vts_token.txt"
_VTS_PORT = 8001
_PLUGIN_NAME = "portrait-to-live2d"
_PLUGIN_DEVELOPER = "portrait-to-live2d"


async def _authenticate(ws) -> None:
    """VTS auth handshake — requests token on first run, reuses on subsequent."""
    _TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    saved_token = _TOKEN_FILE.read_text().strip() if _TOKEN_FILE.exists() else ""

    req = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "auth",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": _PLUGIN_NAME,
            "pluginDeveloper": _PLUGIN_DEVELOPER,
            "authenticationToken": saved_token,
        },
    }
    await ws.send(json.dumps(req))
    resp = json.loads(await ws.recv())

    if resp["data"].get("authenticated"):
        log.info("VTS authenticated")
        return

    # Token rejected or missing — request a new one
    token_req = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "token",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": _PLUGIN_NAME,
            "pluginDeveloper": _PLUGIN_DEVELOPER,
        },
    }
    await ws.send(json.dumps(token_req))
    token_resp = json.loads(await ws.recv())
    token = token_resp["data"]["authenticationToken"]
    _TOKEN_FILE.write_text(token)
    log.info("VTS token saved to %s", _TOKEN_FILE)

    req["data"]["authenticationToken"] = token
    await ws.send(json.dumps(req))
    final = json.loads(await ws.recv())
    if not final["data"].get("authenticated"):
        raise RuntimeError("VTS authentication failed after token request")
    log.info("VTS authenticated with new token")


async def _inject(ws, params: dict[str, float]) -> None:
    """Send InjectParameterDataRequest for all predicted params."""
    payload = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "inject",
        "messageType": "InjectParameterDataRequest",
        "data": {
            "faceFound": True,
            "mode": "set",
            "parameterValues": [
                {"id": k, "value": float(v)} for k, v in params.items()
            ],
        },
    }
    await ws.send(json.dumps(payload))
    await ws.recv()  # consume ack; errors logged separately


async def run(
    rig: RigConfig,
    predictor: Predictor,
    camera_index: int = 0,
    vts_port: int = _VTS_PORT,
) -> None:
    """Main loop: webcam → MLP → VTS.  Reconnects on disconnect."""
    url = f"ws://localhost:{vts_port}"
    while True:
        try:
            async with websockets.connect(url) as ws:
                await _authenticate(ws)
                log.info("param_bridge running — camera %d → VTS %s", camera_index, url)
                async for landmarks in landmark_stream(camera_index=camera_index):
                    params = predictor.predict(landmarks)
                    await _inject(ws, params)
        except (OSError, websockets.ConnectionClosed) as e:
            log.warning("VTS connection lost (%s), retrying in 3s", e)
            await asyncio.sleep(3)
