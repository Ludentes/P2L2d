#!/usr/bin/env python3
"""Generate face_landmarks.json for a template by rendering its rig and running MediaPipe.

Usage:
    python scripts/generate_template_landmarks.py \\
        --template humanoid-anime \\
        --rig hiyori \\
        --size 512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
from PIL import Image

from rig.config import RIG_HIYORI
from rig.render import RigRenderer

# Path to the MediaPipe face landmarker model (already in the repo)
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_MODEL = _REPO_ROOT / "mlp" / "data" / "face_landmarker_v2_with_blendshapes.task"

# MediaPipe FaceMesh 478-point topology — named landmark indices
# (same indices as the legacy FaceMesh 478-point model)
_NAMED_INDICES = {
    "left_eye_center": 473,    # iris centre left (requires refine_landmarks)
    "right_eye_center": 468,   # iris centre right
    "nose_tip": 1,
    "mouth_center": 13,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
}

# Hiyori is full-body. The face occupies roughly the top 25% of the character
# bounding box. We crop to that region, scale up to 512x512 for detection, then
# map the resulting pixel coordinates back to the original image space.
_FACE_BODY_FRACTION = 0.28  # top fraction of body bbox to use as face crop


def _crop_face_region(frame_rgba: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop to the face region of a full-body rig render.

    Returns:
        (face_rgb_512, (crop_x0, crop_y0, crop_x1, crop_y1)) where crop coords
        are in the *original* frame space.
    """
    alpha = frame_rgba[:, :, 3]
    rows_with_content = np.any(alpha > 0, axis=1)
    cols_with_content = np.any(alpha > 0, axis=0)

    row_indices = np.where(rows_with_content)[0]
    col_indices = np.where(cols_with_content)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        raise RuntimeError("Rig render appears to be empty (no non-transparent pixels)")

    rmin, rmax = int(row_indices[0]), int(row_indices[-1])
    cmin, cmax = int(col_indices[0]), int(col_indices[-1])

    # Take top fraction of the body as the face region
    body_h = rmax - rmin
    face_rmax = rmin + max(1, int(body_h * _FACE_BODY_FRACTION))

    # Add small horizontal padding (5% each side) clamped to image bounds
    h_full, w_full = frame_rgba.shape[:2]
    col_pad = max(1, int((cmax - cmin) * 0.05))
    x0 = max(0, cmin - col_pad)
    x1 = min(w_full, cmax + col_pad)
    y0 = rmin
    y1 = min(h_full, face_rmax)

    crop_rgba = frame_rgba[y0:y1, x0:x1]
    pil_crop = Image.fromarray(crop_rgba, mode="RGBA").resize(
        (512, 512), Image.Resampling.LANCZOS
    )
    # Keep black background (transparent → black). MediaPipe detects anime
    # faces more reliably against dark backgrounds than white.
    face_rgb = np.array(pil_crop)[:, :, :3].copy()
    return face_rgb, (x0, y0, x1, y1)


def _run_mediapipe(
    face_rgb_512: np.ndarray,
    crop_box: tuple[int, int, int, int],
    orig_size: tuple[int, int],
    model_path: Path,
    min_detection_confidence: float = 0.1,
) -> np.ndarray:
    """Run MediaPipe FaceLandmarker on a 512x512 face crop, remap to original coords.

    Returns (478, 2) float32 pixel coordinates in the *original* image space.
    """
    x0, y0, x1, y1 = crop_box
    crop_w = x1 - x0
    crop_h = y1 - y0

    base_opts = BaseOptions(model_asset_path=str(model_path))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_detection_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb_512)
    with mp_vision.FaceLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        raise RuntimeError("MediaPipe detected no face in the rig render")

    lms = result.face_landmarks[0]
    # Landmark coords are normalised [0,1] within the 512x512 crop image.
    # Map back: px = lm.x * 512 → scale to crop_w → add crop x0
    pts_crop_px = np.array(
        [[lm.x * 512.0, lm.y * 512.0] for lm in lms], dtype=np.float32
    )
    # Scale from 512x512 crop back to original crop dimensions
    pts_orig = np.empty_like(pts_crop_px)
    pts_orig[:, 0] = pts_crop_px[:, 0] / 512.0 * crop_w + x0
    pts_orig[:, 1] = pts_crop_px[:, 1] / 512.0 * crop_h + y0
    return pts_orig


def generate(
    template: str,
    rig_config,
    size: int,
    out_path: Path,
    model_path: Path = _DEFAULT_MODEL,
) -> None:
    print(f"Rendering {rig_config.name} at {size}x{size} ...")
    with RigRenderer(rig_config, width=size, height=size) as renderer:
        frame = renderer.render(params=None)  # neutral pose, (H, W, 4) uint8

    # Crop to face region for MediaPipe detection
    face_rgb_512, crop_box = _crop_face_region(frame)
    print(f"  Face crop box (x0,y0,x1,y1): {crop_box}")

    print("Running MediaPipe ...")
    # Try progressively lower confidence thresholds for anime characters
    pts = None
    for confidence in (0.5, 0.3, 0.1):
        try:
            pts = _run_mediapipe(
                face_rgb_512, crop_box, (size, size), model_path,
                min_detection_confidence=confidence,
            )
            print(f"  Detected face at confidence={confidence}")
            break
        except RuntimeError:
            print(f"  No face at confidence={confidence}, retrying lower ...")

    if pts is None:
        raise RuntimeError(
            "MediaPipe could not detect a face at any confidence level. "
            "Check that the rig renders a visible face."
        )

    named = {name: pts[idx].tolist() for name, idx in _NAMED_INDICES.items()}
    named["mediapipe_full"] = pts.tolist()

    data = {
        "template": template,
        "render_size": [size, size],
        "generated_from": rig_config.name,
        "landmarks": named,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(pts)} landmarks -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="humanoid-anime")
    parser.add_argument("--rig", default="hiyori")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--model",
        default=str(_DEFAULT_MODEL),
        help="Path to MediaPipe face_landmarker*.task model file",
    )
    args = parser.parse_args()

    rig_map = {"hiyori": RIG_HIYORI}
    if args.rig not in rig_map:
        raise SystemExit(f"Unknown rig {args.rig!r}. Available: {list(rig_map)}")

    out = Path(f"templates/{args.template}/face_landmarks.json")
    generate(args.template, rig_map[args.rig], args.size, out, Path(args.model))
