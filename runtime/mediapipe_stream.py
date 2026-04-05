"""Webcam → 478 MediaPipe FaceMesh landmarks, async generator.

Yields numpy arrays of shape (478, 2) — normalized (x, y) in [0, 1].
Drops frames if the consumer is slower than the camera.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

import cv2
import mediapipe as mp
import numpy as np

log = logging.getLogger(__name__)

_FACE_MESH = mp.solutions.face_mesh


async def landmark_stream(
    camera_index: int = 0,
    max_fps: float = 30.0,
) -> AsyncGenerator[np.ndarray, None]:
    """Async generator that yields (478, 2) landmark arrays from webcam.

    Args:
        camera_index: OpenCV camera index (0 = default webcam).
        max_fps: cap frame rate to avoid saturating the consumer.
    Yields:
        float32 ndarray of shape (478, 2), normalized [0, 1] x/y coords.
    """
    interval = 1.0 / max_fps
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    loop = asyncio.get_running_loop()

    with _FACE_MESH.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        try:
            while True:
                t0 = loop.time()

                ok, frame = await loop.run_in_executor(None, cap.read)
                if not ok:
                    log.warning("Camera read failed, retrying")
                    await asyncio.sleep(0.1)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = await loop.run_in_executor(None, face_mesh.process, rgb)

                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0].landmark
                    arr = np.array([(p.x, p.y) for p in lm], dtype=np.float32)
                    yield arr

                elapsed = loop.time() - t0
                sleep = max(0.0, interval - elapsed)
                if sleep:
                    await asyncio.sleep(sleep)
        finally:
            cap.release()
