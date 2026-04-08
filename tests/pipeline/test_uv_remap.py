"""Tests for pipeline.uv_remap — UV remapping for clean atlas regions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.moc3 import Moc3
from pipeline.uv_remap import (
    RegionDef, RegionBBox, PackedRegion,
    define_regions, compute_region_bboxes, pack_regions,
    remap_uvs, remap_textures, remap_model,
)


HIYORI_MOC3 = Path("models/hiyori/hiyori.moc3")


@pytest.fixture
def hiyori_moc():
    if not HIYORI_MOC3.exists():
        pytest.skip("Hiyori moc3 not available")
    return Moc3.from_file(HIYORI_MOC3)


# ---------------------------------------------------------------------------
# RegionDef
# ---------------------------------------------------------------------------

class TestRegionDef:
    def test_define_regions_returns_list(self):
        regions = define_regions()
        assert isinstance(regions, list)
        assert len(regions) > 0

    def test_all_regions_have_names_and_parts(self):
        for r in define_regions():
            assert isinstance(r.name, str) and len(r.name) > 0
            assert isinstance(r.part_names, list) and len(r.part_names) > 0

    def test_face_region_exists(self):
        names = {r.name for r in define_regions()}
        assert "face" in names
        assert "eyes" in names
        assert "hair_back" in names
        assert "body" in names

    def test_no_duplicate_region_names(self):
        names = [r.name for r in define_regions()]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# compute_region_bboxes (triangle rasterization)
# ---------------------------------------------------------------------------

class TestComputeRegionBBoxes:
    def test_returns_list_of_bboxes(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions)
        assert isinstance(bboxes, list)
        assert all(isinstance(b, RegionBBox) for b in bboxes)

    def test_all_regions_have_bboxes(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions)
        bbox_names = {b.name for b in bboxes}
        region_names = {r.name for r in regions}
        assert bbox_names == region_names

    def test_bboxes_within_texture_bounds(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions, tex_size=2048)
        for b in bboxes:
            assert 0 <= b.min_x <= b.max_x <= 2048, f"{b.name} x out of range"
            assert 0 <= b.min_y <= b.max_y <= 2048, f"{b.name} y out of range"

    def test_face_has_meshes_and_content(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions)
        face = next(b for b in bboxes if b.name == "face")
        assert len(face.mesh_indices) > 0
        assert face.width > 0
        assert face.height > 0

    def test_small_regions_have_tight_bboxes(self, hiyori_moc):
        """Small facial features should have tight content bboxes.

        The nose bbox should be well under 200px — vertex bbox would be much larger
        because the mesh deformation grid spans a wide area.
        """
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions, tex_size=2048)
        nose = next(b for b in bboxes if b.name == "nose")
        assert nose.width < 200, f"nose width {nose.width} not tight"
        assert nose.height < 200, f"nose height {nose.height} not tight"
        # Face should also be reasonably tight (< 700px)
        face = next(b for b in bboxes if b.name == "face")
        assert face.width < 700, f"face width {face.width} not tight"
        assert face.height < 700, f"face height {face.height} not tight"

    def test_mesh_indices_no_duplicates(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions)
        all_indices = []
        for b in bboxes:
            all_indices.extend(b.mesh_indices)
        assert len(all_indices) == len(set(all_indices))


# ---------------------------------------------------------------------------
# pack_regions
# ---------------------------------------------------------------------------

class TestPackRegions:
    def test_returns_packed_regions(self):
        bboxes = [
            RegionBBox("a", 0, 0, 0, 512, 512, [0]),
            RegionBBox("b", 0, 512, 0, 1024, 300, [1]),
        ]
        packed = pack_regions(bboxes, atlas_size=2048, padding=4)
        assert len(packed) == 2
        assert all(isinstance(p, PackedRegion) for p in packed)

    def test_no_overlap(self):
        bboxes = [
            RegionBBox("a", 0, 0, 0, 500, 500, [0]),
            RegionBBox("b", 0, 500, 0, 1000, 300, [1]),
            RegionBBox("c", 0, 0, 500, 400, 900, [2]),
        ]
        packed = pack_regions(bboxes, atlas_size=2048, padding=4)
        for i, a in enumerate(packed):
            for b in packed[i + 1:]:
                overlap_x = a.new_x < b.new_x + b.new_w and b.new_x < a.new_x + a.new_w
                overlap_y = a.new_y < b.new_y + b.new_h and b.new_y < a.new_y + a.new_h
                assert not (overlap_x and overlap_y), f"{a.name} overlaps {b.name}"

    def test_all_fit_within_atlas(self):
        bboxes = [
            RegionBBox("a", 0, 0, 0, 300, 300, [0]),
            RegionBBox("b", 0, 0, 0, 200, 400, [1]),
        ]
        packed = pack_regions(bboxes, atlas_size=1024, padding=4)
        for p in packed:
            assert p.new_x + p.new_w <= 1024, f"{p.name} exceeds atlas width"
            assert p.new_y + p.new_h <= 1024, f"{p.name} exceeds atlas height"

    def test_skips_empty_regions(self):
        bboxes = [
            RegionBBox("a", 0, 0, 0, 500, 500, [0]),
            RegionBBox("empty", 0, 0, 0, 0, 0, []),
        ]
        packed = pack_regions(bboxes, atlas_size=2048, padding=4)
        names = {p.name for p in packed}
        assert "a" in names
        assert "empty" not in names

    def test_with_hiyori(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions, tex_size=2048)
        # Hiyori content is ~204% of 2048 sheet, needs 4096
        packed = pack_regions(bboxes, atlas_size=4096, padding=4)
        assert len(packed) > 0
        for p in packed:
            assert p.new_x + p.new_w <= 4096
            assert p.new_y + p.new_h <= 4096


# ---------------------------------------------------------------------------
# remap_uvs
# ---------------------------------------------------------------------------

class TestRemapUvs:
    def test_uvs_change_after_remap(self, hiyori_moc):
        bboxes = compute_region_bboxes(hiyori_moc, define_regions())
        packed = pack_regions(bboxes, atlas_size=4096, padding=4)

        old_uvs = list(hiyori_moc["uv.xys"])
        remap_uvs(hiyori_moc, packed, atlas_size=4096)
        new_uvs = hiyori_moc["uv.xys"]

        assert old_uvs != new_uvs, "UVs should change after remap"

    def test_texture_indices_all_zero(self, hiyori_moc):
        bboxes = compute_region_bboxes(hiyori_moc, define_regions())
        packed = pack_regions(bboxes, atlas_size=4096, padding=4)

        remap_uvs(hiyori_moc, packed, atlas_size=4096)
        tex_indices = hiyori_moc["art_mesh.texture_indices"]

        remapped_meshes = set()
        for p in packed:
            remapped_meshes.update(p.old_bbox.mesh_indices)
        for mi in remapped_meshes:
            assert tex_indices[mi] == 0

    def test_remapped_moc3_roundtrips(self, hiyori_moc):
        bboxes = compute_region_bboxes(hiyori_moc, define_regions())
        packed = pack_regions(bboxes, atlas_size=4096, padding=4)
        remap_uvs(hiyori_moc, packed, atlas_size=4096)

        data = hiyori_moc.to_bytes()
        moc2 = Moc3.from_bytes(data)
        data2 = moc2.to_bytes()
        assert data == data2


# ---------------------------------------------------------------------------
# remap_textures
# ---------------------------------------------------------------------------

class TestRemapTextures:
    def test_returns_rgba_image(self):
        bboxes = [RegionBBox("a", 0, 0, 0, 128, 128, [0])]
        packed = pack_regions(bboxes, atlas_size=256, padding=4)
        tex = np.zeros((256, 256, 4), dtype=np.uint8)
        tex[:128, :128] = [255, 0, 0, 255]

        result = remap_textures({0: tex}, packed, atlas_size=256)
        assert result.shape == (256, 256, 4)
        assert result.dtype == np.uint8

    def test_pixels_are_copied(self):
        bboxes = [RegionBBox("a", 0, 10, 10, 100, 100, [0])]
        packed = pack_regions(bboxes, atlas_size=256, padding=4)
        tex = np.zeros((256, 256, 4), dtype=np.uint8)
        tex[10:100, 10:100] = [255, 0, 0, 255]

        result = remap_textures({0: tex}, packed, atlas_size=256)
        pr = packed[0]
        region_crop = result[pr.new_y:pr.new_y + pr.new_h, pr.new_x:pr.new_x + pr.new_w]
        assert region_crop.sum() > 0


# ---------------------------------------------------------------------------
# remap_model (end-to-end)
# ---------------------------------------------------------------------------

class TestRemapModel:
    def test_produces_output_files(self, hiyori_moc, tmp_path):
        remap_model(
            moc3_path=HIYORI_MOC3,
            texture_dir=Path("models/hiyori/hiyori.2048"),
            output_dir=tmp_path,
            atlas_size=4096,
        )
        assert (tmp_path / "hiyori.moc3").exists()
        assert (tmp_path / "texture_00.png").exists()
        assert (tmp_path / "region_map.json").exists()

    def test_region_map_format(self, hiyori_moc, tmp_path):
        remap_model(
            moc3_path=HIYORI_MOC3,
            texture_dir=Path("models/hiyori/hiyori.2048"),
            output_dir=tmp_path,
            atlas_size=4096,
        )
        with open(tmp_path / "region_map.json") as f:
            rmap = json.load(f)

        assert rmap["atlas_size"] == 4096
        assert "face" in rmap["regions"]
        face = rmap["regions"]["face"]
        assert all(k in face for k in ("x", "y", "w", "h"))
        assert face["w"] > 0 and face["h"] > 0

    def test_output_moc3_roundtrips(self, hiyori_moc, tmp_path):
        remap_model(
            moc3_path=HIYORI_MOC3,
            texture_dir=Path("models/hiyori/hiyori.2048"),
            output_dir=tmp_path,
            atlas_size=4096,
        )
        moc = Moc3.from_file(tmp_path / "hiyori.moc3")
        data1 = moc.to_bytes()
        moc2 = Moc3.from_bytes(data1)
        assert data1 == moc2.to_bytes()
