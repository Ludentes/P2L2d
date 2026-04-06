"""Artifact packager — assemble deliverable Live2D model directory."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from rig.config import RigConfig


def package_output(
    rig: "RigConfig",
    modified_atlases: "dict[int, PILImage.Image]",
    output_dir: Path,
) -> Path:
    """Copy rig files into output_dir, replacing atlases with modified versions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(rig.moc3_path, output_dir / rig.moc3_path.name)
    shutil.copy2(rig.model3_json_path, output_dir / rig.model3_json_path.name)

    for idx, tex_path in enumerate(rig.textures):
        rel = tex_path.relative_to(rig.model_dir)
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if idx in modified_atlases:
            modified_atlases[idx].save(dest)
        else:
            shutil.copy2(tex_path, dest)

    return output_dir
