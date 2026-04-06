"""Face alignment utilities for the texture generation pipeline.

Provides landmark detection, affine warp, atlas region cropping, and inpaint
mask generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pipeline.atlas_config import AtlasConfig
from pipeline.exceptions import MediaPipeLandmarkError

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_MODEL = _REPO_ROOT / "mlp" / "data" / "face_landmarker_v2_with_blendshapes.task"

_DEFAULT_FEATURE_REGIONS = [
    "left_eye",
    "right_eye",
    "left_eyebrow",
    "right_eyebrow",
    "mouth",
]


def detect_landmarks(image: Image.Image) -> np.ndarray:
    """Run MediaPipe FaceLandmarker (Tasks API) on image.

    Returns (478, 2) float32 pixel coords.
    Raises MediaPipeLandmarkError if no face detected.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    base_opts = BaseOptions(model_asset_path=str(_DEFAULT_MODEL))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image.convert("RGB")))
    with mp_vision.FaceLandmarker.create_from_options(opts) as landmarker:
        result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        raise MediaPipeLandmarkError(
            "No face detected in the portrait image. "
            "Ensure the image contains a clear frontal face."
        )

    lms = result.face_landmarks[0]
    h, w = np.array(image).shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


def load_template_landmarks(lm_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load face_landmarks.json.

    Returns: pts (478, 2) float32, render_size (width, height).
    """
    with open(lm_path) as f:
        data = json.load(f)

    render_size_raw = data["render_size"]
    render_size: tuple[int, int] = (int(render_size_raw[0]), int(render_size_raw[1]))

    pts = np.array(data["landmarks"]["mediapipe_full"], dtype=np.float32)
    return pts, render_size


def compute_affine_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute 2x3 affine matrix mapping src_pts → dst_pts (least-squares).

    Returns M (2, 3) such that M @ [x, y, 1]^T → [x', y']^T.
    """
    src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float32)])
    M_T, _, _, _ = np.linalg.lstsq(src_h, dst_pts, rcond=None)
    return M_T.T.astype(np.float32)  # (2, 3)


def warp_image(image: Image.Image, M: np.ndarray, output_size: tuple[int, int]) -> Image.Image:
    """Warp image using forward affine matrix M.

    PIL expects inverse mapping; M is inverted internally.
    Preserves image mode (RGB or RGBA).
    """
    M_full = np.vstack([M, [0.0, 0.0, 1.0]])
    M_inv = np.linalg.inv(M_full)[:2, :]
    data = M_inv.flatten().tolist()
    fill: tuple[int, ...] = (0, 0, 0, 0) if image.mode == "RGBA" else (0, 0, 0)
    return image.transform(
        output_size,
        Image.Transform.AFFINE,
        data,
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )


def align_portrait(
    portrait: Image.Image,
    template_landmarks_path: Path,
    output_size: tuple[int, int] = (512, 512),
) -> tuple[Image.Image, np.ndarray]:
    """Detect landmarks in portrait, warp to template space.

    Returns (warped Image, M (2,3) matrix).
    Scales template_pts from render_size to output_size before computing M.
    """
    portrait_pts = detect_landmarks(portrait)
    template_pts, render_size = load_template_landmarks(template_landmarks_path)

    # Scale template pts from render_size to output_size
    scale_x = output_size[0] / render_size[0]
    scale_y = output_size[1] / render_size[1]
    template_pts_scaled = template_pts * np.array([scale_x, scale_y], dtype=np.float32)

    M = compute_affine_transform(portrait_pts, template_pts_scaled)
    warped = warp_image(portrait, M, output_size)
    return warped, M


def crop_region(warped: Image.Image, atlas_cfg: AtlasConfig, region_name: str) -> Image.Image:
    """Crop named region from warped portrait.

    Scales atlas bbox (in texture_size space) to warped image dimensions.
    Uses integer pixel math: x = int(region.x * scale_x), etc.
    width = max(1, int(region.w * scale_x)), height = max(1, int(region.h * scale_y))
    """
    region = atlas_cfg.get(region_name)
    w_img, h_img = warped.size
    scale_x = w_img / atlas_cfg.texture_size
    scale_y = h_img / atlas_cfg.texture_size

    x = int(region.x * scale_x)
    y = int(region.y * scale_y)
    width = max(1, int(region.w * scale_x))
    height = max(1, int(region.h * scale_y))

    return warped.crop((x, y, x + width, y + height))


def build_face_inpaint_mask(
    atlas_cfg: AtlasConfig,
    warped_size: tuple[int, int] = (512, 512),
    face_region_name: str = "face_skin",
    feature_regions: list[str] | None = None,
    dilation_px: int = 8,
) -> Image.Image:
    """Build L-mode inpaint mask sized to the face crop dimensions.

    White (255) = inpaint (where animated features will appear), black = keep.
    Default feature_regions = ["left_eye", "right_eye", "left_eyebrow",
    "right_eyebrow", "mouth"].
    Skips any feature_region not present in atlas_cfg.
    Applies MaxFilter dilation of (dilation_px*2+1) if dilation_px > 0.
    Coordinates of features are relative to face_region origin.
    """
    if feature_regions is None:
        feature_regions = _DEFAULT_FEATURE_REGIONS

    face_region = atlas_cfg.get(face_region_name)
    w_img, h_img = warped_size
    scale_x = w_img / atlas_cfg.texture_size
    scale_y = h_img / atlas_cfg.texture_size

    # Face crop dimensions in warped space
    face_w = max(1, int(face_region.w * scale_x))
    face_h = max(1, int(face_region.h * scale_y))
    face_x = int(face_region.x * scale_x)
    face_y = int(face_region.y * scale_y)

    mask = Image.new("L", (face_w, face_h), 0)
    draw = ImageDraw.Draw(mask)

    for feat_name in feature_regions:
        if not atlas_cfg.has(feat_name):
            continue
        feat = atlas_cfg.get(feat_name)

        # Feature coords in warped space, relative to face_region origin
        fx = int(feat.x * scale_x) - face_x
        fy = int(feat.y * scale_y) - face_y
        fw = max(1, int(feat.w * scale_x))
        fh = max(1, int(feat.h * scale_y))

        draw.rectangle([fx, fy, fx + fw - 1, fy + fh - 1], fill=255)

    if dilation_px > 0:
        kernel_size = dilation_px * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel_size))

    return mask
