"""BCI → OBS ambient effects bridge.

Connects to the ZyphraExps EEG backend WebSocket (ws://localhost:8765),
reads metrics, and drives an OBS Color Correction filter in real time.

Prerequisite (one-time OBS setup):
  1. In OBS: add a Color Correction filter to any scene source (e.g. a
     full-screen image or solid-color overlay at low opacity).
  2. Name the filter exactly: "EEG_Ambient"
  3. Name the source exactly:  "EEG_Overlay"  (or change SOURCE_NAME below)
  4. Enable OBS WebSocket: Tools → obs-websocket Settings, port 4455.

EEG signal → Color Correction mapping:
  relaxation (0–1)   → saturation 1.0 → 0.0  (relaxed = desaturated/dreamy)
  concentration(0–1) → hue_shift  0   → +25  (focused = cool blue tint)
  jaw_clench (bool)  → brightness pulse +0.3 for 200ms
  heart_rate (bpm)   → gamma: 60bpm=0, 120bpm=+0.15 (subtle warmth at effort)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets
import obsws_python as obs

log = logging.getLogger(__name__)

# --- configurable constants ---
EEG_WS_URL = "ws://localhost:8765"
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = ""          # set if OBS WebSocket has a password
SOURCE_NAME = "EEG_Overlay"
FILTER_NAME = "EEG_Ambient"
UPDATE_HZ = 20             # filter update rate (20 Hz is plenty for EEG timescales)
CLENCH_PULSE_SEC = 0.2     # brightness pulse duration on jaw clench


class ObsBridge:
    """Reads EEG metrics WebSocket, writes OBS Color Correction filter."""

    def __init__(
        self,
        eeg_url: str = EEG_WS_URL,
        obs_host: str = OBS_HOST,
        obs_port: int = OBS_PORT,
        obs_password: str = OBS_PASSWORD,
        source: str = SOURCE_NAME,
        filter_name: str = FILTER_NAME,
    ) -> None:
        self._eeg_url = eeg_url
        self._obs_cfg = dict(host=obs_host, port=obs_port, password=obs_password)
        self._source = source
        self._filter = filter_name

        # latest metric values (updated by _eeg_reader)
        self._concentration: float = 0.0
        self._relaxation: float = 0.0
        self._heart_rate: float = 70.0
        self._clench_until: float = 0.0  # epoch time when brightness pulse expires

    # ------------------------------------------------------------------
    # EEG reader
    # ------------------------------------------------------------------

    async def _eeg_reader(self) -> None:
        """Long-running coroutine: maintains WS connection to ZyphraExps."""
        while True:
            try:
                async with websockets.connect(self._eeg_url) as ws:
                    log.info("obs_bridge: connected to EEG backend %s", self._eeg_url)
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        if msg.get("type") != "metrics":
                            continue
                        self._ingest(msg)
            except (OSError, websockets.ConnectionClosed) as e:
                log.warning("obs_bridge: EEG WS lost (%s), retrying in 3s", e)
                await asyncio.sleep(3)

    def _ingest(self, msg: dict) -> None:
        brain = msg.get("brain") or {}
        self._concentration = float(brain.get("concentration", self._concentration))
        self._relaxation = float(brain.get("relaxation", self._relaxation))

        ppg = msg.get("ppg") or {}
        bpm = ppg.get("heart_rate_bpm")
        if bpm is not None:
            self._heart_rate = float(bpm)

        imu = msg.get("imu") or {}
        if imu.get("jaw_clench"):
            self._clench_until = time.time() + CLENCH_PULSE_SEC

    # ------------------------------------------------------------------
    # OBS writer
    # ------------------------------------------------------------------

    def _compute_settings(self) -> dict:
        """Map current EEG state → Color Correction filter settings."""
        # saturation: relaxed = 0.0 (greyscale), alert = 1.5 (vivid)
        saturation = 1.5 - self._relaxation * 1.5

        # hue_shift: focused = cool (+25°), neutral = 0
        hue_shift = self._concentration * 25.0

        # brightness: +0.3 pulse on jaw clench, else 0
        brightness = 0.3 if time.time() < self._clench_until else 0.0

        # gamma: subtle warmth at elevated heart rate (>80bpm)
        gamma = max(0.0, (self._heart_rate - 80.0) / 400.0)

        return {
            "saturation": round(saturation, 3),
            "hue_shift": round(hue_shift, 2),
            "brightness": round(brightness, 3),
            "gamma": round(gamma, 4),
        }

    async def _obs_writer(self) -> None:
        """Periodic coroutine: push filter settings to OBS at UPDATE_HZ."""
        interval = 1.0 / UPDATE_HZ
        cl = obs.ReqClient(**self._obs_cfg)
        log.info(
            "obs_bridge: connected to OBS ws://%s:%d, updating '%s' → '%s' at %dHz",
            self._obs_cfg["host"], self._obs_cfg["port"],
            self._source, self._filter, UPDATE_HZ,
        )
        try:
            while True:
                t0 = asyncio.get_event_loop().time()
                settings = self._compute_settings()
                try:
                    cl.set_source_filter_settings(
                        source_name=self._source,
                        filter_name=self._filter,
                        filter_settings=settings,
                        overlay=True,
                    )
                except Exception as e:
                    log.warning("obs_bridge: OBS write failed: %s", e)

                elapsed = asyncio.get_event_loop().time() - t0
                await asyncio.sleep(max(0.0, interval - elapsed))
        finally:
            cl.disconnect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run EEG reader and OBS writer concurrently."""
        await asyncio.gather(
            self._eeg_reader(),
            self._obs_writer(),
        )
