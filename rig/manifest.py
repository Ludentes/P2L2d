"""Per-rig manifest — maps user's custom Live2D param names to template canonical names."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Manifest:
    """Maps a rig's param names to a template's canonical param names.

    Attributes:
        template_name: Which template this rig uses (e.g. "humanoid-anime").
        param_map: Mapping of rig_param_id -> template_param_name.
                   e.g. {"ParamAngleX": "AngleX", "ParamMouthOpenY": "MouthOpenY"}
    """

    template_name: str
    param_map: dict[str, str]  # rig_param -> template_param

    def remap(self, template_output: dict[str, float]) -> dict[str, float]:
        """Convert template param names to rig param names.

        Given MLP output like {"AngleX": -5.2}, returns {"ParamAngleX": -5.2}
        using the inverse of param_map. Template params not mapped to any rig
        param are dropped.
        """
        reverse = {tp: rp for rp, tp in self.param_map.items()}
        return {
            reverse[tp]: value
            for tp, value in template_output.items()
            if tp in reverse
        }


def load_manifest(path: Path) -> Manifest:
    """Load a manifest from a TOML file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Manifest(
        template_name=data["template"],
        param_map=dict(data.get("param_map", {})),
    )
