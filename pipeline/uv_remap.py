"""UV remap — rearrange moc3 UV layout into clean rectangular regions for AI editing.

Uses per-triangle rasterization to find actual pixel coverage (not vertex bboxes),
then packs tight content bboxes into a new atlas with non-overlapping rectangular
regions per semantic body part.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.moc3 import Moc3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RegionDef:
    """Defines a semantic region as a group of Live2D parts."""
    name: str
    part_names: list[str]


@dataclass
class RegionBBox:
    """A region's tight content bounding box on a specific texture sheet.

    Unlike vertex bboxes (which span the entire sheet for large deformation grids),
    these are computed from triangle rasterization — the actual pixels the region covers.
    """
    name: str
    texture_index: int
    min_x: int  # pixel coordinates on original texture
    min_y: int
    max_x: int
    max_y: int
    mesh_indices: list[int] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.max_x - self.min_x

    @property
    def height(self) -> int:
        return self.max_y - self.min_y


@dataclass
class PackedRegion:
    """A region assigned to a position on the new atlas."""
    name: str
    old_bbox: RegionBBox
    new_x: int  # pixel x on new atlas
    new_y: int  # pixel y on new atlas
    new_w: int  # pixel width on new atlas
    new_h: int  # pixel height on new atlas


# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _get_mesh_uvs(moc: Moc3, mesh_idx: int) -> np.ndarray:
    """Get UV coordinates for a mesh as (N, 2) float array."""
    uv_begin = moc["art_mesh.uv_begin_indices"][mesh_idx]
    vc = moc["art_mesh.vertex_counts"][mesh_idx]
    if vc == 0:
        return np.empty((0, 2), dtype=np.float32)
    all_uvs = moc["uv.xys"]
    flat = all_uvs[uv_begin : uv_begin + vc * 2]
    return np.array(flat, dtype=np.float32).reshape(-1, 2)


def _get_mesh_triangles(moc: Moc3, mesh_idx: int) -> np.ndarray:
    """Get triangle indices for a mesh as (T, 3) int array."""
    idx_begin = moc["art_mesh.position_index_begin_indices"][mesh_idx]
    idx_count = moc["art_mesh.position_index_counts"][mesh_idx]
    # Triangle indices come in groups of 3; truncate any remainder
    tri_count = idx_count // 3
    if tri_count == 0:
        return np.empty((0, 3), dtype=np.int32)
    all_indices = moc["position_index.indices"]
    flat = all_indices[idx_begin : idx_begin + tri_count * 3]
    return np.array(flat, dtype=np.int32).reshape(-1, 3)


def _group_meshes_by_region(
    moc: Moc3, regions: list[RegionDef],
) -> dict[str, list[int]]:
    """Map each art mesh to its region based on parent part."""
    part_ids = moc["part.ids"]
    parent_parts = moc["art_mesh.parent_part_indices"]

    part_to_region: dict[str, str] = {}
    for r in regions:
        for pname in r.part_names:
            part_to_region[pname] = r.name

    result: dict[str, list[int]] = {r.name: [] for r in regions}
    for mi in range(len(parent_parts)):
        pi = parent_parts[mi]
        if 0 <= pi < len(part_ids):
            rname = part_to_region.get(part_ids[pi])
            if rname is not None:
                result[rname].append(mi)
    return result


# ---------------------------------------------------------------------------
# Triangle rasterization for tight bboxes
# ---------------------------------------------------------------------------

def _rasterize_region_mask(
    moc: Moc3,
    mesh_indices: list[int],
    tex_size: int,
) -> np.ndarray:
    """Rasterize all triangles for a set of meshes into a pixel mask.

    Returns a (tex_size, tex_size) uint8 array where non-zero pixels
    indicate actual triangle coverage.
    """
    mask = np.zeros((tex_size, tex_size), dtype=np.uint8)

    for mi in mesh_indices:
        uvs = _get_mesh_uvs(moc, mi)
        tris = _get_mesh_triangles(moc, mi)
        if len(uvs) == 0 or len(tris) == 0:
            continue

        # Convert UV [0,1] to pixel coordinates
        pts_px = uvs * tex_size  # (N, 2) float — (x, y)

        for t0, t1, t2 in tris:
            if t0 >= len(pts_px) or t1 >= len(pts_px) or t2 >= len(pts_px):
                continue
            tri_pts = np.array([
                pts_px[t0], pts_px[t1], pts_px[t2],
            ], dtype=np.int32)
            cv2.fillPoly(mask, [tri_pts], 255)

    return mask


def compute_region_bboxes(
    moc: Moc3,
    regions: list[RegionDef],
    tex_size: int = 2048,
) -> list[RegionBBox]:
    """Compute per-region tight content bounding boxes via triangle rasterization.

    For each region, rasterizes all mesh triangles into a pixel mask,
    then finds the tight bounding box of non-zero pixels. This gives
    the actual content area, not the vertex bounding box (which can
    span the entire texture for large deformation grids).
    """
    tex_indices = moc["art_mesh.texture_indices"]
    grouped = _group_meshes_by_region(moc, regions)

    result: list[RegionBBox] = []
    for region in regions:
        indices = grouped[region.name]
        if not indices:
            result.append(RegionBBox(region.name, 0, 0, 0, 0, 0, []))
            continue

        tex_idx = tex_indices[indices[0]]
        mask = _rasterize_region_mask(moc, indices, tex_size)

        # Find tight bbox of non-zero pixels
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not rows.any():
            result.append(RegionBBox(region.name, tex_idx, 0, 0, 0, 0, indices))
            continue

        min_y = int(np.argmax(rows))
        max_y = int(tex_size - np.argmax(rows[::-1]))
        min_x = int(np.argmax(cols))
        max_x = int(tex_size - np.argmax(cols[::-1]))

        result.append(RegionBBox(
            name=region.name,
            texture_index=tex_idx,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            mesh_indices=indices,
        ))

    return result


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_regions(
    bboxes: list[RegionBBox],
    atlas_size: int = 2048,
    padding: int = 4,
) -> list[PackedRegion]:
    """Shelf-pack tight region bboxes into a new atlas.

    Sorts by height (tallest first), places left-to-right on shelves.
    Skips empty regions (zero width or height).
    """
    sized = []
    for b in bboxes:
        if b.width <= 0 or b.height <= 0 or not b.mesh_indices:
            continue
        pw = b.width + padding
        ph = b.height + padding
        sized.append((b, pw, ph))

    sized.sort(key=lambda x: x[2], reverse=True)

    result: list[PackedRegion] = []
    shelf_y = 0
    shelf_h = 0
    cursor_x = 0

    for bbox, pw, ph in sized:
        if cursor_x + pw > atlas_size:
            shelf_y += shelf_h
            shelf_h = 0
            cursor_x = 0

        if shelf_y + ph > atlas_size:
            raise ValueError(
                f"Region '{bbox.name}' doesn't fit in {atlas_size}x{atlas_size} atlas. "
                f"Need ({cursor_x + pw}, {shelf_y + ph}). Try atlas_size=4096."
            )

        result.append(PackedRegion(
            name=bbox.name,
            old_bbox=bbox,
            new_x=cursor_x,
            new_y=shelf_y,
            new_w=bbox.width,
            new_h=bbox.height,
        ))

        cursor_x += pw
        shelf_h = max(shelf_h, ph)

    return result


# ---------------------------------------------------------------------------
# UV remapping
# ---------------------------------------------------------------------------

def remap_uvs(
    moc: Moc3,
    packing: list[PackedRegion],
    atlas_size: int = 4096,
    orig_tex_size: int = 2048,
) -> None:
    """Modify moc3 UV coordinates in-place to match new atlas layout.

    For each packed region, transforms vertex UVs from old content bbox
    position to new atlas position. Vertices outside the content bbox
    (transparent grid vertices) get extrapolated — this is fine because
    their triangles only sample transparent pixels.

    Also sets all remapped mesh texture_indices to 0 (single sheet).

    Args:
        moc: Moc3 object to modify in-place.
        packing: List of PackedRegion from pack_regions().
        atlas_size: Size of new atlas in pixels.
        orig_tex_size: Size of original texture sheets in pixels.
    """
    uv_begins = moc["art_mesh.uv_begin_indices"]
    vertex_counts = moc["art_mesh.vertex_counts"]
    uvs = moc["uv.xys"]
    tex_indices = moc["art_mesh.texture_indices"]

    for pr in packing:
        ob = pr.old_bbox
        if ob.width <= 0 or ob.height <= 0:
            continue

        # Old content bbox in UV space (on original texture)
        old_u0 = ob.min_x / orig_tex_size
        old_v0 = ob.min_y / orig_tex_size
        old_uw = ob.width / orig_tex_size
        old_vh = ob.height / orig_tex_size

        # New region in UV space (on new atlas)
        new_u0 = pr.new_x / atlas_size
        new_v0 = pr.new_y / atlas_size
        new_uw = pr.new_w / atlas_size
        new_vh = pr.new_h / atlas_size

        for mi in ob.mesh_indices:
            vc = vertex_counts[mi]
            if vc == 0:
                continue
            uv_start = uv_begins[mi]

            for j in range(vc):
                idx_u = uv_start + j * 2
                idx_v = uv_start + j * 2 + 1

                old_u = uvs[idx_u]
                old_v = uvs[idx_v]

                # Linear transform: old content bbox → new bbox
                uvs[idx_u] = (old_u - old_u0) / old_uw * new_uw + new_u0
                uvs[idx_v] = (old_v - old_v0) / old_vh * new_vh + new_v0

            tex_indices[mi] = 0


# ---------------------------------------------------------------------------
# Texture remapping
# ---------------------------------------------------------------------------

def remap_textures(
    old_textures: dict[int, np.ndarray],
    packing: list[PackedRegion],
    atlas_size: int = 2048,
) -> np.ndarray:
    """Copy pixel regions from old textures to new atlas.

    For each packed region, crops the content bbox from the old texture
    and pastes it into the new atlas at the packed position.

    Returns a single RGBA image (atlas_size x atlas_size x 4).
    """
    new_atlas = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)

    for pr in packing:
        ob = pr.old_bbox
        if ob.width <= 0 or ob.height <= 0:
            continue

        old_tex = old_textures.get(ob.texture_index)
        if old_tex is None:
            continue

        # Crop from old texture using content bbox (pixel coords)
        crop = old_tex[ob.min_y:ob.max_y, ob.min_x:ob.max_x]

        # Resize if dimensions differ (shouldn't normally happen with tight bboxes)
        if crop.shape[1] != pr.new_w or crop.shape[0] != pr.new_h:
            from PIL import Image
            pil_crop = Image.fromarray(crop)
            pil_crop = pil_crop.resize((pr.new_w, pr.new_h), Image.LANCZOS)
            crop = np.array(pil_crop)

        new_atlas[pr.new_y:pr.new_y + pr.new_h, pr.new_x:pr.new_x + pr.new_w] = crop

    return new_atlas


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def remap_model(
    moc3_path: Path,
    texture_dir: Path,
    output_dir: Path,
    atlas_size: int = 2048,
    regions: list[RegionDef] | None = None,
) -> dict:
    """End-to-end UV remap: read model, remap UVs, write output.

    Returns region_map dict (also written to region_map.json).
    """
    from PIL import Image

    if regions is None:
        regions = define_regions()

    moc = Moc3.from_file(moc3_path)

    # Load textures
    old_textures: dict[int, np.ndarray] = {}
    for tex_path in sorted(texture_dir.glob("texture_*.png")):
        idx = int(tex_path.stem.split("_")[1])
        old_textures[idx] = np.array(Image.open(tex_path).convert("RGBA"))

    # Infer texture size from first texture
    tex_size = next(iter(old_textures.values())).shape[0]

    # Compute tight bboxes via triangle rasterization
    bboxes = compute_region_bboxes(moc, regions, tex_size=tex_size)
    packed = pack_regions(bboxes, atlas_size=atlas_size, padding=4)

    # Remap
    remap_uvs(moc, packed, atlas_size=atlas_size)
    new_atlas = remap_textures(old_textures, packed, atlas_size=atlas_size)

    # Build region map
    region_map: dict = {
        "atlas_size": atlas_size,
        "regions": {},
    }
    for pr in packed:
        region_map["regions"][pr.name] = {
            "x": pr.new_x,
            "y": pr.new_y,
            "w": pr.new_w,
            "h": pr.new_h,
        }

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    moc.to_file(output_dir / moc3_path.name)
    Image.fromarray(new_atlas).save(output_dir / "texture_00.png")

    with open(output_dir / "region_map.json", "w") as f:
        json.dump(region_map, f, indent=2)

    # Copy and update model3.json
    for candidate in [
        moc3_path.with_suffix(".model3.json"),
        moc3_path.parent / (moc3_path.stem + ".model3.json"),
    ]:
        if candidate.exists():
            model3 = json.loads(candidate.read_text())
            model3["FileReferences"]["Textures"] = ["texture_00.png"]
            with open(output_dir / candidate.name, "w") as f:
                json.dump(model3, f, indent="\t")
            break

    return region_map
