"""Atlas coordinate config — maps canonical region names to pixel bounding boxes.

Each rig has a corresponding manifests/<rig>_atlas.toml. This file never changes
for a given rig unless the texture atlas layout changes.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AtlasRegion:
    name: str           # canonical name, e.g. "face_skin"
    texture_index: int  # 0-based index into RigConfig.textures
    x: int              # left edge in pixels
    y: int              # top edge in pixels
    w: int              # width in pixels
    h: int              # height in pixels

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(
                f"AtlasRegion {self.name!r}: w and h must be positive, got w={self.w}, h={self.h}"
            )


@dataclass
class AtlasConfig:
    rig_name: str
    template_name: str
    texture_size: int           # square texture side length (e.g. 2048)
    regions: list[AtlasRegion]

    def get(self, name: str) -> AtlasRegion:
        """Return the region with this name. Raises KeyError if not found."""
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"Region {name!r} not found in atlas config for {self.rig_name!r}")

    def has(self, name: str) -> bool:
        return any(r.name == name for r in self.regions)


def load_atlas_config(path: Path) -> AtlasConfig:
    """Load atlas config from a TOML file. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Atlas config not found: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if data["texture_size"] <= 0:
        raise ValueError(f"texture_size must be positive, got {data['texture_size']} in {path}")
    regions = []
    for r in data.get("regions", []):
        # Support both flat format (texture_index/x/y/w/h at region level) and
        # drawable-grouped format ([[regions.drawables]] sub-tables).  When
        # drawables are present the first drawable supplies the bounding box.
        if "texture_index" in r:
            regions.append(AtlasRegion(
                name=r["name"],
                texture_index=r["texture_index"],
                x=r["x"],
                y=r["y"],
                w=r["w"],
                h=r["h"],
            ))
        elif "drawables" in r and r["drawables"]:
            d = r["drawables"][0]
            regions.append(AtlasRegion(
                name=r["name"],
                texture_index=d["texture_index"],
                x=d["x"],
                y=d["y"],
                w=d["w"],
                h=d["h"],
            ))
        else:
            raise ValueError(
                f"Region {r.get('name')!r} in {path} has neither 'texture_index' "
                "nor 'drawables' — cannot determine bounding box."
            )
    return AtlasConfig(
        rig_name=data["rig"],
        template_name=data["template"],
        texture_size=data["texture_size"],
        regions=regions,
    )
