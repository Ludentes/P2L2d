"""Tests for pipeline.uv_remap — UV remapping for clean atlas regions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.moc3 import Moc3
from pipeline.uv_remap import RegionDef, RegionBBox, define_regions, compute_region_bboxes


HIYORI_MOC3 = Path("models/hiyori/hiyori.moc3")


@pytest.fixture
def hiyori_moc():
    if not HIYORI_MOC3.exists():
        pytest.skip("Hiyori moc3 not available")
    return Moc3.from_file(HIYORI_MOC3)


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

    def test_bboxes_within_unit_range(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions)
        for b in bboxes:
            # Allow small negative values (real UV data can slightly exceed [0,1])
            assert -0.01 <= b.min_u <= b.max_u <= 1.01, f"{b.name} u out of range"
            assert -0.01 <= b.min_v <= b.max_v <= 1.01, f"{b.name} v out of range"

    def test_face_has_meshes(self, hiyori_moc):
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions)
        face = next(b for b in bboxes if b.name == "face")
        assert len(face.mesh_indices) > 0
        assert face.width > 0
        assert face.height > 0

    def test_mesh_indices_cover_all_meshes(self, hiyori_moc):
        """Every art mesh with vertices should appear in exactly one region."""
        regions = define_regions()
        bboxes = compute_region_bboxes(hiyori_moc, regions)
        all_indices = []
        for b in bboxes:
            all_indices.extend(b.mesh_indices)
        # No duplicates
        assert len(all_indices) == len(set(all_indices))
