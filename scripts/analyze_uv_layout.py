#!/usr/bin/env python3
"""Analyze UV layout of a moc3 file — mesh-to-texture mapping, per-part regions, overlap."""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.moc3 import Moc3

MOC3_PATH = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels/hiyori_vts/hiyori.moc3"


def main():
    moc = Moc3.from_file(MOC3_PATH)
    print(moc.summary())
    print()

    # Extract arrays
    mesh_ids = moc.get("art_mesh.ids")
    part_indices = moc.get("art_mesh.parent_part_indices")
    tex_indices = moc.get("art_mesh.texture_indices")
    uv_begins = moc.get("art_mesh.uv_begin_indices")
    vert_counts = moc.get("art_mesh.vertex_counts")
    uv_xys = moc.get("uv.xys")
    part_ids = moc.get("part.ids")

    n_meshes = len(mesh_ids)
    print(f"Total art meshes: {n_meshes}")
    print(f"Total UV floats: {len(uv_xys)} ({len(uv_xys)//2} vertices)")
    print(f"Total parts: {len(part_ids)}")
    print(f"Unique texture indices: {sorted(set(tex_indices))}")
    print()

    # ---------------------------------------------------------------
    # Per-mesh info
    # ---------------------------------------------------------------
    mesh_infos = []
    for i in range(n_meshes):
        mid = mesh_ids[i]
        pid = part_indices[i]
        tid = tex_indices[i]
        vc = vert_counts[i]
        uv_start = uv_begins[i]  # index into uv_xys (counts floats, x then y)

        if vc == 0:
            mesh_infos.append({
                "id": mid, "part_idx": pid, "tex": tid, "verts": 0,
                "uv_min_x": None, "uv_max_x": None,
                "uv_min_y": None, "uv_max_y": None,
            })
            continue

        # uv_begin_indices is index into the flat float array
        # Each vertex = 2 floats (x, y)
        float_start = uv_start * 2  # NO — uv_begin_indices should already be float index
        # Actually let's check: if uv_begins are vertex indices, multiply by 2
        # If they're float indices, use directly.
        # The section "uv.xys" has count = UVS (total UV floats / already total UV float pairs?)
        # Count idx is UVS. Let's check: total UVs count vs total floats
        uv_count = moc.counts[15]  # CountIdx.UVS
        # uv.xys has uv_count elements, each is f32
        # So uv.xys is a flat list of uv_count floats
        # uv_begin_indices: index where this mesh's UVs start (in terms of float pairs? or floats?)

        # Let's figure out: sum of vertex_counts * 2 should equal len(uv_xys) if uv_begin is vertex index
        # Or sum of vertex_counts should equal len(uv_xys)/2 if uv_begin is float-pair index
        # We'll try: uv_begin is a vertex index (pair index)
        float_idx = uv_start * 2

        xs = []
        ys = []
        for v in range(vc):
            fi = float_idx + v * 2
            if fi + 1 < len(uv_xys):
                xs.append(uv_xys[fi])
                ys.append(uv_xys[fi + 1])

        if not xs:
            mesh_infos.append({
                "id": mid, "part_idx": pid, "tex": tid, "verts": vc,
                "uv_min_x": None, "uv_max_x": None,
                "uv_min_y": None, "uv_max_y": None,
            })
            continue

        mesh_infos.append({
            "id": mid, "part_idx": pid, "tex": tid, "verts": vc,
            "uv_min_x": min(xs), "uv_max_x": max(xs),
            "uv_min_y": min(ys), "uv_max_y": max(ys),
        })

    # Validate: check if uv_begin interpretation is correct
    # If uv_begin is vertex index, last mesh's uv_begin + vertex_count should ~= len(uv_xys)/2
    total_verts = sum(m["verts"] for m in mesh_infos)
    print(f"Total vertices across meshes: {total_verts}")
    print(f"UV float count / 2: {len(uv_xys) / 2}")
    if total_verts != len(uv_xys) / 2:
        # Try the other interpretation: uv_begin is float index
        print("NOTE: vertex count != UV pairs. Trying uv_begin as float index...")
        for i in range(n_meshes):
            vc = vert_counts[i]
            if vc == 0:
                continue
            float_idx = uv_begins[i]  # direct float index
            xs = []
            ys = []
            for v in range(vc):
                fi = float_idx + v * 2
                if fi + 1 < len(uv_xys):
                    xs.append(uv_xys[fi])
                    ys.append(uv_xys[fi + 1])
            if xs:
                mesh_infos[i]["uv_min_x"] = min(xs)
                mesh_infos[i]["uv_max_x"] = max(xs)
                mesh_infos[i]["uv_min_y"] = min(ys)
                mesh_infos[i]["uv_max_y"] = max(ys)
    print()

    # ---------------------------------------------------------------
    # Print per-mesh table
    # ---------------------------------------------------------------
    print("=" * 120)
    print(f"{'Mesh ID':<40} {'Part':<30} {'Tex':>3} {'Verts':>5}  {'UV X range':>20}  {'UV Y range':>20}")
    print("-" * 120)
    for m in mesh_infos:
        pname = part_ids[m["part_idx"]] if 0 <= m["part_idx"] < len(part_ids) else f"?{m['part_idx']}"
        if m["uv_min_x"] is not None:
            xr = f"[{m['uv_min_x']:.4f}, {m['uv_max_x']:.4f}]"
            yr = f"[{m['uv_min_y']:.4f}, {m['uv_max_y']:.4f}]"
        else:
            xr = yr = "N/A"
        print(f"{m['id']:<40} {pname:<30} {m['tex']:>3} {m['verts']:>5}  {xr:>20}  {yr:>20}")
    print()

    # ---------------------------------------------------------------
    # Group by part → compute per-part UV bbox per texture
    # ---------------------------------------------------------------
    # part_idx -> list of mesh_infos
    part_meshes = defaultdict(list)
    for m in mesh_infos:
        part_meshes[m["part_idx"]].append(m)

    print("=" * 120)
    print("PER-PART UV REGIONS (grouped by texture)")
    print("=" * 120)

    # (tex, part_idx) -> (min_x, min_y, max_x, max_y, mesh_count, vert_count)
    part_tex_regions = {}

    for pidx in sorted(part_meshes.keys()):
        pname = part_ids[pidx] if 0 <= pidx < len(part_ids) else f"?{pidx}"
        meshes = part_meshes[pidx]

        # Group by texture
        by_tex = defaultdict(list)
        for m in meshes:
            by_tex[m["tex"]].append(m)

        for tid in sorted(by_tex.keys()):
            tex_meshes = by_tex[tid]
            valid = [m for m in tex_meshes if m["uv_min_x"] is not None]
            if not valid:
                continue

            min_x = min(m["uv_min_x"] for m in valid)
            max_x = max(m["uv_max_x"] for m in valid)
            min_y = min(m["uv_min_y"] for m in valid)
            max_y = max(m["uv_max_y"] for m in valid)
            total_v = sum(m["verts"] for m in valid)
            area = (max_x - min_x) * (max_y - min_y)

            part_tex_regions[(tid, pidx)] = (min_x, min_y, max_x, max_y, len(valid), total_v)

            print(f"  Part: {pname:<30} Tex: {tid}  Meshes: {len(valid):>2}  Verts: {total_v:>4}  "
                  f"UV bbox: [{min_x:.4f},{min_y:.4f}]-[{max_x:.4f},{max_y:.4f}]  "
                  f"Area: {area:.4f} ({area*100:.1f}%)")

    print()

    # ---------------------------------------------------------------
    # Overlap analysis: for each texture, find parts whose UV bboxes overlap
    # ---------------------------------------------------------------
    print("=" * 120)
    print("UV BBOX OVERLAP ANALYSIS (same texture sheet)")
    print("=" * 120)

    # Group by texture
    tex_parts = defaultdict(list)
    for (tid, pidx), region in part_tex_regions.items():
        tex_parts[tid].append((pidx, region))

    def bbox_overlap(r1, r2):
        """Return overlap area between two (min_x, min_y, max_x, max_y) bboxes."""
        ox = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
        oy = max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
        return ox * oy

    for tid in sorted(tex_parts.keys()):
        parts = tex_parts[tid]
        print(f"\n--- Texture {tid} ({len(parts)} parts) ---")

        overlaps = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                pidx_a, ra = parts[i]
                pidx_b, rb = parts[j]
                ov = bbox_overlap(ra, rb)
                if ov > 0.0001:  # threshold for reporting
                    pname_a = part_ids[pidx_a] if 0 <= pidx_a < len(part_ids) else f"?{pidx_a}"
                    pname_b = part_ids[pidx_b] if 0 <= pidx_b < len(part_ids) else f"?{pidx_b}"
                    overlaps.append((ov, pname_a, pname_b, ra, rb))

        overlaps.sort(reverse=True)
        if not overlaps:
            print("  No significant UV bbox overlaps.")
        else:
            print(f"  {len(overlaps)} overlapping part pairs:")
            for ov, na, nb, ra, rb in overlaps[:30]:
                area_a = (ra[2]-ra[0])*(ra[3]-ra[1])
                area_b = (rb[2]-rb[0])*(rb[3]-rb[1])
                print(f"    {na:<30} x {nb:<30}  overlap: {ov:.4f} ({ov*100:.1f}%)  "
                      f"areas: {area_a:.4f}/{area_b:.4f}")
            if len(overlaps) > 30:
                print(f"    ... and {len(overlaps) - 30} more")

    print()

    # ---------------------------------------------------------------
    # Separability analysis: which parts are cleanly isolated?
    # ---------------------------------------------------------------
    print("=" * 120)
    print("SEPARABILITY ANALYSIS")
    print("=" * 120)

    for tid in sorted(tex_parts.keys()):
        parts = tex_parts[tid]
        print(f"\n--- Texture {tid} ---")

        clean = []
        overlapping = []

        for i, (pidx, ra) in enumerate(parts):
            pname = part_ids[pidx] if 0 <= pidx < len(part_ids) else f"?{pidx}"
            has_overlap = False
            for j, (pidx_b, rb) in enumerate(parts):
                if i == j:
                    continue
                if bbox_overlap(ra, rb) > 0.0001:
                    has_overlap = True
                    break
            area = (ra[2]-ra[0])*(ra[3]-ra[1])
            if has_overlap:
                overlapping.append((pname, area, ra))
            else:
                clean.append((pname, area, ra))

        print(f"  CLEANLY SEPARABLE ({len(clean)}):")
        for name, area, r in sorted(clean, key=lambda x: -x[1]):
            print(f"    {name:<35} area: {area:.4f} ({area*100:.1f}%)  "
                  f"bbox: [{r[0]:.4f},{r[1]:.4f}]-[{r[2]:.4f},{r[3]:.4f}]")

        print(f"  OVERLAPPING ({len(overlapping)}):")
        for name, area, r in sorted(overlapping, key=lambda x: -x[1]):
            print(f"    {name:<35} area: {area:.4f} ({area*100:.1f}%)  "
                  f"bbox: [{r[0]:.4f},{r[1]:.4f}]-[{r[2]:.4f},{r[3]:.4f}]")

    # ---------------------------------------------------------------
    # Summary stats
    # ---------------------------------------------------------------
    print()
    print("=" * 120)
    print("TEXTURE UTILIZATION SUMMARY")
    print("=" * 120)
    for tid in sorted(tex_parts.keys()):
        parts = tex_parts[tid]
        # Union bbox of all parts on this texture
        all_min_x = min(r[0] for _, r in parts)
        all_min_y = min(r[1] for _, r in parts)
        all_max_x = max(r[2] for _, r in parts)
        all_max_y = max(r[3] for _, r in parts)
        union_area = (all_max_x - all_min_x) * (all_max_y - all_min_y)
        sum_areas = sum((r[2]-r[0])*(r[3]-r[1]) for _, r in parts)
        total_meshes = sum(r[4] for _, r in parts)
        total_verts = sum(r[5] for _, r in parts)
        print(f"  Texture {tid}: {len(parts)} parts, {total_meshes} meshes, {total_verts} verts")
        print(f"    Union bbox: [{all_min_x:.4f},{all_min_y:.4f}]-[{all_max_x:.4f},{all_max_y:.4f}]  "
              f"area: {union_area:.4f} ({union_area*100:.1f}%)")
        print(f"    Sum of part bbox areas: {sum_areas:.4f} ({sum_areas*100:.1f}%)  "
              f"overlap ratio: {sum_areas/union_area:.2f}x" if union_area > 0 else "")


if __name__ == "__main__":
    main()
