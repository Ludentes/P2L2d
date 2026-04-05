"""Parametric expression sliders for LivePortrait.

Ported from PowerHouseMan/ComfyUI-AdvancedLivePortrait's calc_fe() (MIT License).
Applies hand-tuned additive offsets to specific keypoint indices of the
transformed driving keypoints x_d (21 implicit keypoints × 3 coordinates).

Slider ranges (from PHM ExpressionEditor):
  rotate_pitch: [-20, 20]    rotate_yaw: [-20, 20]    rotate_roll: [-20, 20]
  blink: [-20, 5]            eyebrow: [-10, 15]       wink: [0, 25]
  pupil_x: [-15, 15]         pupil_y: [-15, 15]
  aaa: [0, 120]              eee: [-20, 15]           woo: [-20, 15]
  smile: [-0.3, 1.3]
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class VerbSliders:
    """Parametric expression sliders. All default to 0 (neutral)."""
    rotate_pitch: float = 0.0
    rotate_yaw:   float = 0.0
    rotate_roll:  float = 0.0
    blink:        float = 0.0   # "eyes" in PHM — negative = close
    eyebrow:      float = 0.0
    wink:         float = 0.0
    pupil_x:      float = 0.0
    pupil_y:      float = 0.0
    aaa:          float = 0.0   # "mouth" in PHM — open vertically
    eee:          float = 0.0
    woo:          float = 0.0
    smile:        float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def apply_sliders(
    x_d_new: torch.Tensor,
    sliders: VerbSliders,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    """Apply PHM slider offsets in-place to transformed driving keypoints.

    x_d_new: (1, 21, 3) driving keypoints (modified in place)
    Returns: (x_d_new, (adj_pitch, adj_yaw, adj_roll)) — rotation adjustments
             that mouth/wink sliders apply to head rotation.
    """
    smile = sliders.smile
    mouth = sliders.aaa            # PHM renamed aaa → mouth internally
    eee = sliders.eee
    woo = sliders.woo
    wink = sliders.wink
    pupil_x = sliders.pupil_x
    pupil_y = sliders.pupil_y
    eyes = sliders.blink           # PHM renamed blink → eyes internally
    eyebrow = sliders.eyebrow
    rotate_pitch = sliders.rotate_pitch
    rotate_yaw = sliders.rotate_yaw
    rotate_roll = sliders.rotate_roll

    # --- smile ---
    x_d_new[0, 20, 1] += smile * -0.01
    x_d_new[0, 14, 1] += smile * -0.02
    x_d_new[0, 17, 1] += smile * 0.0065
    x_d_new[0, 17, 2] += smile * 0.003
    x_d_new[0, 13, 1] += smile * -0.00275
    x_d_new[0, 16, 1] += smile * -0.00275
    x_d_new[0, 3, 1]  += smile * -0.0035
    x_d_new[0, 7, 1]  += smile * -0.0035

    # --- mouth (aaa) ---
    x_d_new[0, 19, 1] += mouth * 0.001
    x_d_new[0, 19, 2] += mouth * 0.0001
    x_d_new[0, 17, 1] += mouth * -0.0001
    rotate_pitch -= mouth * 0.05

    # --- eee ---
    x_d_new[0, 20, 2] += eee * -0.001
    x_d_new[0, 20, 1] += eee * -0.001
    x_d_new[0, 14, 1] += eee * -0.001

    # --- woo ---
    x_d_new[0, 14, 1] += woo * 0.001
    x_d_new[0, 3, 1]  += woo * -0.0005
    x_d_new[0, 7, 1]  += woo * -0.0005
    x_d_new[0, 17, 2] += woo * -0.0005

    # --- wink ---
    x_d_new[0, 11, 1] += wink * 0.001
    x_d_new[0, 13, 1] += wink * -0.0003
    x_d_new[0, 17, 0] += wink * 0.0003
    x_d_new[0, 17, 1] += wink * 0.0003
    x_d_new[0, 3, 1]  += wink * -0.0003
    rotate_roll -= wink * 0.1
    rotate_yaw  -= wink * 0.1

    # --- pupil_x (asymmetric: left vs right gaze use different weights) ---
    if pupil_x > 0:
        x_d_new[0, 11, 0] += pupil_x * 0.0007
        x_d_new[0, 15, 0] += pupil_x * 0.001
    else:
        x_d_new[0, 11, 0] += pupil_x * 0.001
        x_d_new[0, 15, 0] += pupil_x * 0.0007

    # --- pupil_y (couples to eyes/blink) ---
    x_d_new[0, 11, 1] += pupil_y * -0.001
    x_d_new[0, 15, 1] += pupil_y * -0.001
    eyes -= pupil_y / 2.0

    # --- eyes (blink) ---
    x_d_new[0, 11, 1] += eyes * -0.001
    x_d_new[0, 13, 1] += eyes * 0.0003
    x_d_new[0, 15, 1] += eyes * -0.001
    x_d_new[0, 16, 1] += eyes * 0.0003
    x_d_new[0, 1, 1]  += eyes * -0.00025
    x_d_new[0, 2, 1]  += eyes * 0.00025

    # --- eyebrow (asymmetric: raise vs furrow) ---
    if eyebrow > 0:
        x_d_new[0, 1, 1] += eyebrow * 0.001
        x_d_new[0, 2, 1] += eyebrow * -0.001
    else:
        x_d_new[0, 1, 0] += eyebrow * -0.001
        x_d_new[0, 2, 0] += eyebrow * 0.001
        x_d_new[0, 1, 1] += eyebrow * 0.0003
        x_d_new[0, 2, 1] += eyebrow * -0.0003

    return x_d_new, (rotate_pitch, rotate_yaw, rotate_roll)
