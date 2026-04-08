"""Tests for pipeline.uv_remap — UV remapping for clean atlas regions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.uv_remap import RegionDef, define_regions


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
