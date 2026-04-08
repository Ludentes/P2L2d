#!/usr/bin/env python3
"""Analyze UV overlap between semantic regions in Hiyori's Live2D model.

For each texture sheet, rasterizes all art mesh triangles in UV space
and checks whether different semantic regions claim the same pixels.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.moc3 import Moc3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MOC3_PATH = Path.home() / (
    ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/"
    "VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/"
    "hiyori_vts/hiyori.moc3"
)

TEX_SIZE = 2048

# Semantic region groups: region_name -> list of part name substrings to match
REGION_GROUPS = {
    "face": ["PartFace"],
    "eyes": ["PartEye", "PartEyeBall"],
    "brows": ["PartBrow"],
    "mouth": ["PartMouth"],
    "nose": ["PartNose"],
    "ear": ["PartEar"],
    "hair_front": ["PartHairFront"],
    "hair_back": ["PartHairBack"],
    "hair_side": ["PartHairSide"],
    "neck": ["PartNeck"],
    "body": ["PartBody"],
    "arm_a": ["PartArmA"],
    "arm_b": ["PartArmB"],
    "core": ["PartCore"],
    "skinning": ["PartSkinning", "_Skinning"],
    "cheek": ["PartCheek"],
    "background": ["PartBackground"],
}


def classify_part(part_name: str) -> str | None:
    """Return region name for a part, or None if unmatched."""
    for region, prefixes in REGION_GROUPS.items():
        for prefix in prefixes:
            if prefix in part_name:
                return region
    return None


def main():
    print(f"Loading moc3: {MOC3_PATH}")
    moc = Moc3.from_file(MOC3_PATH)

    part_ids = moc["part.ids"]
    art_mesh_ids = moc["art_mesh.ids"]
    parent_part_indices = moc["art_mesh.parent_part_indices"]
    texture_indices = moc["art_mesh.texture_indices"]
    uv_begin = moc["art_mesh.uv_begin_indices"]
    vertex_counts = moc["art_mesh.vertex_counts"]
    idx_begin = moc["art_mesh.position_index_begin_indices"]
    idx_counts = moc["art_mesh.position_index_counts"]
    uv_xys = moc["uv.xys"]
    tri_indices = moc["position_index.indices"]

    n_meshes = len(art_mesh_ids)
    n_textures = max(texture_indices) + 1 if texture_indices else 1

    print(f"Parts: {len(part_ids)}")
    print(f"Art meshes: {n_meshes}")
    print(f"Texture sheets: {n_textures}")
    print()

    # Build region ID map (1-based, 0 = unclaimed)
    region_names = list(REGION_GROUPS.keys())
    region_id_map = {name: i + 1 for i, name in enumerate(region_names)}

    # Classify each part
    part_to_region = {}
    for i, pid in enumerate(part_ids):
        region = classify_part(pid)
        part_to_region[i] = region

    # Print part classification
    print("=== Part Classification ===")
    unmatched_parts = []
    for i, pid in enumerate(part_ids):
        region = part_to_region[i]
        if region:
            print(f"  {pid:40s} -> {region}")
        else:
            unmatched_parts.append(pid)
    if unmatched_parts:
        print(f"\n  Unmatched parts ({len(unmatched_parts)}):")
        for p in unmatched_parts:
            print(f"    {p}")
    print()

    # Create ownership bitmaps: one per texture sheet
    # Use uint16 so we can track region IDs > 255
    # We'll also keep a "count" bitmap to detect overlaps
    ownership = [np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.uint16) for _ in range(n_textures)]
    overlap_count = [np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.uint8) for _ in range(n_textures)]

    # Per-region pixel tracking
    region_pixels = defaultdict(int)  # region_name -> pixel count
    mesh_region_map = {}  # mesh_idx -> region_name

    # Process each art mesh
    for mi in range(n_meshes):
        part_idx = parent_part_indices[mi]
        region = part_to_region.get(part_idx)
        mesh_region_map[mi] = region

        if region is None:
            continue

        rid = region_id_map[region]
        tex_idx = texture_indices[mi]

        # Get UVs for this mesh
        # uv_begin is a FLOAT index into uv_xys (not vertex index)
        # uv_xys is flat: [x0, y0, x1, y1, ...]
        uv_float_start = uv_begin[mi]
        n_verts = vertex_counts[mi]
        uv_float_end = uv_float_start + n_verts * 2
        if uv_float_end > len(uv_xys):
            continue  # skip meshes that overflow
        uvs = []
        for vi in range(n_verts):
            u = uv_xys[uv_float_start + vi * 2]
            v = uv_xys[uv_float_start + vi * 2 + 1]
            uvs.append((u * TEX_SIZE, v * TEX_SIZE))
        uvs = np.array(uvs, dtype=np.float32)

        # Get triangle indices for this mesh
        tri_start = idx_begin[mi]
        tri_count = idx_counts[mi]
        indices = tri_indices[tri_start : tri_start + tri_count]

        # Rasterize triangles
        n_tris = len(indices) // 3
        for t in range(n_tris):
            i0, i1, i2 = indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]
            if max(i0, i1, i2) >= len(uvs):
                continue
            pts = np.array([
                [uvs[i0][0], uvs[i0][1]],
                [uvs[i1][0], uvs[i1][1]],
                [uvs[i2][0], uvs[i2][1]],
            ], dtype=np.int32)

            # Create a temp mask for this triangle
            mask = np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)

            # Where this triangle touches
            touched = mask > 0

            # Check for new ownership vs existing
            existing = ownership[tex_idx][touched]
            new_pixels = existing == 0
            contested = (existing != 0) & (existing != rid)

            # Mark ownership (first-come for non-contested)
            ownership[tex_idx][touched] = np.where(
                existing == 0, rid, existing
            )

            # For contested pixels, mark them specially
            overlap_count[tex_idx][touched] = np.where(
                contested,
                np.maximum(overlap_count[tex_idx][touched], 2),
                np.where(
                    new_pixels,
                    1,
                    overlap_count[tex_idx][touched]
                )
            )

    # --- Analysis ---
    print("=== Per-Region Pixel Counts (per texture sheet) ===")
    for tex_idx in range(n_textures):
        print(f"\n  Texture {tex_idx}:")
        for rname in region_names:
            rid = region_id_map[rname]
            count = int(np.sum(ownership[tex_idx] == rid))
            if count > 0:
                pct = count / (TEX_SIZE * TEX_SIZE) * 100
                print(f"    {rname:20s}: {count:8d} px ({pct:5.2f}%)")

    # More precise overlap detection: re-scan with per-pixel region sets
    print("\n=== Overlap Detection (precise, per texture sheet) ===")

    for tex_idx in range(n_textures):
        # Build a list of (region_id, mask) for this texture
        region_masks = {}
        for mi in range(n_meshes):
            if texture_indices[mi] != tex_idx:
                continue
            region = mesh_region_map.get(mi)
            if region is None:
                continue
            rid = region_id_map[region]

            uv_float_start = uv_begin[mi]
            n_verts = vertex_counts[mi]
            uv_float_end = uv_float_start + n_verts * 2
            if uv_float_end > len(uv_xys):
                continue
            uvs = []
            for vi in range(n_verts):
                u = uv_xys[uv_float_start + vi * 2]
                v = uv_xys[uv_float_start + vi * 2 + 1]
                uvs.append((u * TEX_SIZE, v * TEX_SIZE))
            uvs_arr = np.array(uvs, dtype=np.float32)

            tri_start = idx_begin[mi]
            tri_count = idx_counts[mi]
            indices = tri_indices[tri_start : tri_start + tri_count]

            if rid not in region_masks:
                region_masks[rid] = np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.uint8)

            n_tris = len(indices) // 3
            for t in range(n_tris):
                i0, i1, i2 = indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]
                if max(i0, i1, i2) >= len(uvs):
                    continue
                pts = np.array([
                    [uvs_arr[i0][0], uvs_arr[i0][1]],
                    [uvs_arr[i1][0], uvs_arr[i1][1]],
                    [uvs_arr[i2][0], uvs_arr[i2][1]],
                ], dtype=np.int32)
                cv2.fillPoly(region_masks[rid], [pts], 1)

        # Check pairwise overlaps
        rids_present = sorted(region_masks.keys())
        total_contested = 0
        pair_overlaps = []

        for i, r1 in enumerate(rids_present):
            for r2 in rids_present[i + 1:]:
                overlap = np.sum((region_masks[r1] > 0) & (region_masks[r2] > 0))
                if overlap > 0:
                    n1 = region_names[r1 - 1]
                    n2 = region_names[r2 - 1]
                    pair_overlaps.append((n1, n2, int(overlap)))
                    total_contested += int(overlap)

        print(f"\n  Texture {tex_idx}:")
        if pair_overlaps:
            print(f"    Total contested pixels: {total_contested}")
            print(f"    Region pair overlaps:")
            for n1, n2, count in sorted(pair_overlaps, key=lambda x: -x[2]):
                print(f"      {n1:20s} x {n2:20s}: {count:8d} px")
        else:
            print("    No overlaps detected between semantic regions.")

    # Summary
    print("\n=== Summary ===")
    for tex_idx in range(n_textures):
        total_owned = int(np.sum(ownership[tex_idx] > 0))
        total_pixels = TEX_SIZE * TEX_SIZE
        print(f"  Texture {tex_idx}: {total_owned}/{total_pixels} pixels owned "
              f"({total_owned / total_pixels * 100:.1f}%)")

    # Also report meshes per region
    print("\n=== Meshes Per Region ===")
    region_meshes = defaultdict(list)
    unassigned = []
    for mi in range(n_meshes):
        region = mesh_region_map.get(mi)
        if region:
            region_meshes[region].append(art_mesh_ids[mi])
        else:
            part_idx = parent_part_indices[mi]
            unassigned.append((art_mesh_ids[mi], part_ids[part_idx]))

    for rname in region_names:
        meshes = region_meshes.get(rname, [])
        if meshes:
            print(f"  {rname} ({len(meshes)} meshes):")
            for m in meshes[:5]:
                print(f"    {m}")
            if len(meshes) > 5:
                print(f"    ... and {len(meshes) - 5} more")

    if unassigned:
        print(f"\n  Unassigned ({len(unassigned)} meshes):")
        for mesh_id, part_name in unassigned[:10]:
            print(f"    {mesh_id:40s} (part: {part_name})")
        if len(unassigned) > 10:
            print(f"    ... and {len(unassigned) - 10} more")


if __name__ == "__main__":
    main()
