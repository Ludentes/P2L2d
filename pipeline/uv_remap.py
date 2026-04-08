"""UV remap — rearrange moc3 UV layout into clean rectangular regions for AI editing."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RegionDef:
    """Defines a semantic region as a group of Live2D parts."""
    name: str
    part_names: list[str]


@dataclass
class RegionBBox:
    """A region's UV bounding box on a specific texture sheet."""
    name: str
    texture_index: int
    min_u: float
    min_v: float
    max_u: float
    max_v: float
    mesh_indices: list[int] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.max_u - self.min_u

    @property
    def height(self) -> float:
        return self.max_v - self.min_v


@dataclass
class PackedRegion:
    """A region assigned to a position on the new atlas."""
    name: str
    old_bbox: RegionBBox
    new_x: int  # pixel x on new atlas
    new_y: int  # pixel y on new atlas
    new_w: int  # pixel width on new atlas
    new_h: int  # pixel height on new atlas


def define_regions() -> list[RegionDef]:
    """Default Hiyori region groupings based on part hierarchy."""
    return [
        RegionDef("face", ["PartFace"]),
        RegionDef("eyes", ["PartEye", "PartEyeBall"]),
        RegionDef("brows", ["PartBrow"]),
        RegionDef("mouth", ["PartMouth"]),
        RegionDef("nose", ["PartNose"]),
        RegionDef("ear", ["PartEar"]),
        RegionDef("hair_front", ["PartHairFront"]),
        RegionDef("hair_back", ["PartHairBack"]),
        RegionDef("hair_side", ["PartHairSide"]),
        RegionDef("neck", ["PartNeck"]),
        RegionDef("skinning", ["PartSkinning"]),
        RegionDef("body", ["PartBody"]),
        RegionDef("arms", ["PartArmA", "PartArmB"]),
    ]
