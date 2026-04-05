"""VerbRenderer — wraps LivePortrait for parametric verb-based rendering.

Usage:
    renderer = VerbRenderer.from_default_checkpoints()
    source = renderer.precompute_source(np.array(Image.open("ref.jpg")))
    img = renderer.render(source, VerbSliders(blink=-15.0))  # close eyes
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .verb_sliders import VerbSliders, apply_sliders

# Vendored LivePortrait lives at <repo_root>/third_party/LivePortrait
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LP_ROOT = _REPO_ROOT / "third_party" / "LivePortrait"
_LP_WEIGHTS = _LP_ROOT / "pretrained_weights"
if str(_LP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LP_ROOT))

# Imports from vendored LivePortrait (resolved via sys.path insertion above)
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.config.crop_config import CropConfig  # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapper  # noqa: E402
from src.utils.cropper import Cropper  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402


@dataclass
class SourceState:
    """Precomputed source image data. Reusable across many verb renders."""
    kp_info: dict          # raw output of motion_extractor (pitch, yaw, roll, t, exp, scale, kp)
    feature_3d: torch.Tensor    # (1, 32, 16, 64, 64) appearance feature volume
    x_s: torch.Tensor           # (1, 21, 3) source transformed keypoints
    cropped_rgb: np.ndarray     # (256, 256, 3) uint8 — cropped source image (for debugging)


class VerbRenderer:
    """LivePortrait-based parametric verb renderer.

    Source image is cropped + encoded once; then many verbs can be rendered
    cheaply by manipulating the expression/rotation tensors and calling
    the warp+decode pipeline.
    """

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.crop_cfg = crop_cfg

    @classmethod
    def from_default_checkpoints(cls, device_id: int = 0) -> "VerbRenderer":
        """Build a renderer using the vendored LivePortrait weights."""
        inference_cfg = InferenceConfig(
            device_id=device_id,
            checkpoint_F=str(_LP_WEIGHTS / "liveportrait/base_models/appearance_feature_extractor.pth"),
            checkpoint_M=str(_LP_WEIGHTS / "liveportrait/base_models/motion_extractor.pth"),
            checkpoint_W=str(_LP_WEIGHTS / "liveportrait/base_models/warping_module.pth"),
            checkpoint_G=str(_LP_WEIGHTS / "liveportrait/base_models/spade_generator.pth"),
            checkpoint_S=str(_LP_WEIGHTS / "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
            flag_use_half_precision=False,
        )
        crop_cfg = CropConfig(
            insightface_root=str(_LP_WEIGHTS / "insightface"),
            landmark_ckpt_path=str(_LP_WEIGHTS / "liveportrait/landmark.onnx"),
        )
        return cls(inference_cfg, crop_cfg)

    def precompute_source(self, img_rgb: np.ndarray) -> SourceState:
        """Run the cropper + motion extractor + appearance extractor once."""
        crop_info = self.cropper.crop_source_image(img_rgb, self.crop_cfg)
        if crop_info is None:
            raise RuntimeError("Face detection failed on source image")
        cropped = crop_info["img_crop_256x256"]  # (256, 256, 3) uint8

        I_s = self.wrapper.prepare_source(cropped)            # (1, 3, 256, 256)
        kp_info = self.wrapper.get_kp_info(I_s)                # dict with pitch/yaw/roll/t/exp/scale/kp
        feature_3d = self.wrapper.extract_feature_3d(I_s)     # (1, 32, 16, 64, 64)
        x_s = self.wrapper.transform_keypoint(kp_info)         # (1, 21, 3)

        return SourceState(
            kp_info=kp_info,
            feature_3d=feature_3d,
            x_s=x_s,
            cropped_rgb=cropped,
        )

    def render(self, source: SourceState, sliders: VerbSliders) -> np.ndarray:
        """Render the source with verb sliders applied. Returns (256, 256, 3) uint8."""
        # Start from source transformed keypoints
        x_d_new = source.x_s.clone()

        # Apply slider deltas to driving keypoints + accumulate rotation adjustments
        x_d_new, (adj_pitch, adj_yaw, adj_roll) = apply_sliders(x_d_new, sliders)

        # Apply head rotation: the slider rotate_pitch/yaw/roll (plus mouth/wink adjustments)
        # are applied as an additional rotation on top of the source's native pose.
        if adj_pitch != 0.0 or adj_yaw != 0.0 or adj_roll != 0.0:
            device = x_d_new.device
            R_delta = get_rotation_matrix(
                torch.tensor([[adj_pitch]], device=device),
                torch.tensor([[adj_yaw]], device=device),
                torch.tensor([[adj_roll]], device=device),
            )  # (1, 3, 3)
            # Rotate driving keypoints around the origin by R_delta
            scale = source.kp_info["scale"]
            t = source.kp_info["t"]
            # Undo source scale/translation, rotate, re-apply
            centered = (x_d_new - t[:, None, :].to(x_d_new.dtype)) / scale[..., None].to(x_d_new.dtype)
            rotated = centered @ R_delta
            x_d_new = rotated * scale[..., None].to(x_d_new.dtype) + t[:, None, :].to(x_d_new.dtype)

        # Apply stitching for smooth composition
        x_d_stitched = self.wrapper.stitching(source.x_s, x_d_new)

        # Warp + decode
        out = self.wrapper.warp_decode(source.feature_3d, source.x_s, x_d_stitched)
        img = self.wrapper.parse_output(out["out"])  # (1, 256, 256, 3) uint8
        return img[0]  # → (256, 256, 3) uint8
