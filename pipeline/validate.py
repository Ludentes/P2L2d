"""Headless texture validation for Live2D rigs.

Provides:
  - check_region_color: pure-numpy pixel presence check
  - validate_textures: render a rig with modified atlases in a temp directory
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from rig.config import RigConfig


def check_region_color(
    frame: np.ndarray,
    expected_color: tuple[int, int, int],
    tolerance: int = 20,
) -> bool:
    """Return True if ANY visible pixel in *frame* matches *expected_color*.

    Parameters
    ----------
    frame:
        (H, W, 4) uint8 RGBA array.
    expected_color:
        (R, G, B) tuple, no alpha component.
    tolerance:
        Maximum per-channel absolute difference allowed for a match.

    Returns
    -------
    bool
        True when at least one pixel has alpha > 10 and all three channels
        within *tolerance* of *expected_color*.
    """
    r, g, b = expected_color
    alpha_mask = frame[..., 3] > 10
    color_match = (
        (np.abs(frame[..., 0].astype(np.int16) - r) <= tolerance)
        & (np.abs(frame[..., 1].astype(np.int16) - g) <= tolerance)
        & (np.abs(frame[..., 2].astype(np.int16) - b) <= tolerance)
    )
    return bool(np.any(alpha_mask & color_match))


def validate_textures(
    config: "RigConfig",
    modified_atlases: "dict[int, PILImage.Image]",
) -> np.ndarray:
    """Render the rig with *modified_atlases* substituted for the originals.

    Steps
    -----
    1. Copy the entire model directory to a temporary directory.
    2. Write each modified atlas over the corresponding texture file
       (path derived from ``config.textures[idx].relative_to(config.model_dir)``).
    3. Build a temporary ``RigConfig`` pointing at the temp copy.
    4. Render at neutral pose (no params) via ``RigRenderer``.
    5. Clean up the temp directory.

    Parameters
    ----------
    config:
        Original rig config whose ``model_dir`` and ``textures`` are used
        to derive destination paths.  Not mutated.
    modified_atlases:
        Mapping of atlas index → PIL Image (RGBA) to write in the temp copy.

    Returns
    -------
    np.ndarray
        (H, W, 4) uint8 RGBA frame from the headless renderer.
    """
    from dataclasses import replace

    from rig.render import RigRenderer  # noqa: PLC0415

    tmpdir = Path(tempfile.mkdtemp(prefix="p2l_validate_"))
    try:
        # --- 1. Copy model directory tree ---
        tmp_model_dir = tmpdir / config.model_dir.name
        shutil.copytree(config.model_dir, tmp_model_dir)

        # --- 2. Overwrite modified atlases ---
        for idx, image in modified_atlases.items():
            rel = config.textures[idx].relative_to(config.model_dir)
            dest = tmp_model_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest)

        # --- 3. Build a temporary RigConfig ---
        tmp_textures = [
            tmp_model_dir / tex.relative_to(config.model_dir)
            for tex in config.textures
        ]
        tmp_config = replace(
            config,
            model_dir=tmp_model_dir,
            moc3_path=tmp_model_dir / config.moc3_path.relative_to(config.model_dir),
            model3_json_path=tmp_model_dir / config.model3_json_path.relative_to(config.model_dir),
            textures=tmp_textures,
        )

        # --- 4. Render at neutral pose ---
        with RigRenderer(tmp_config) as renderer:
            frame = renderer.render(params=None)

        return frame

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
