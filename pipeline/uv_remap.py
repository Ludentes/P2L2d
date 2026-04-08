"""UV remap — rearrange moc3 UV layout into clean rectangular regions for AI editing."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pipeline.moc3 import Moc3


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


def compute_region_bboxes(moc: Moc3, regions: list[RegionDef]) -> list[RegionBBox]:
    """Compute per-region UV bounding boxes from mesh data.

    Groups art meshes by parent part -> region mapping.
    Returns one RegionBBox per region (skips regions with no meshes).
    """
    part_ids = moc["part.ids"]
    mesh_parent_parts = moc["art_mesh.parent_part_indices"]
    mesh_tex_indices = moc["art_mesh.texture_indices"]
    uv_begins = moc["art_mesh.uv_begin_indices"]
    uv_vertex_counts = moc["art_mesh.position_index_counts"]
    all_uvs = moc["uv.xys"]

    # Build part_name -> region_name lookup
    part_to_region: dict[str, str] = {}
    for region in regions:
        for pname in region.part_names:
            part_to_region[pname] = region.name

    # Group meshes by region
    region_meshes: dict[str, list[int]] = {r.name: [] for r in regions}
    for mesh_idx in range(len(mesh_parent_parts)):
        part_idx = mesh_parent_parts[mesh_idx]
        if part_idx < 0 or part_idx >= len(part_ids):
            continue
        part_name = part_ids[part_idx]
        rname = part_to_region.get(part_name)
        if rname is not None:
            region_meshes[rname].append(mesh_idx)

    # Compute bboxes
    result: list[RegionBBox] = []
    for region in regions:
        indices = region_meshes[region.name]
        if not indices:
            # Empty region -- create zero-size bbox
            result.append(RegionBBox(region.name, 0, 0.0, 0.0, 0.0, 0.0, []))
            continue

        us: list[float] = []
        vs: list[float] = []
        tex_idx = mesh_tex_indices[indices[0]]

        for mi in indices:
            vc = uv_vertex_counts[mi]
            if vc == 0:
                continue
            uv_start = uv_begins[mi]  # already a float index into uv.xys
            for j in range(vc):
                us.append(all_uvs[uv_start + j * 2])
                vs.append(all_uvs[uv_start + j * 2 + 1])

        if not us:
            result.append(RegionBBox(region.name, tex_idx, 0.0, 0.0, 0.0, 0.0, indices))
            continue

        result.append(RegionBBox(
            name=region.name,
            texture_index=tex_idx,
            min_u=min(us),
            min_v=min(vs),
            max_u=max(us),
            max_v=max(vs),
            mesh_indices=indices,
        ))

    return result
