from rig.config import RIG_HIYORI, RigConfig


def test_rig_hiyori_param_count():
    assert RIG_HIYORI.param_count == 74


def test_rig_hiyori_param_ids_no_duplicates():
    assert len(RIG_HIYORI.param_ids) == len(set(RIG_HIYORI.param_ids))


def test_rig_hiyori_has_core_vts_params():
    ids = set(RIG_HIYORI.param_ids)
    for expected in ("ParamAngleX", "ParamEyeLOpen", "ParamMouthOpenY", "ParamBrowLY"):
        assert expected in ids, f"Missing expected param {expected}"


def test_rig_hiyori_texture_count():
    assert len(RIG_HIYORI.textures) == 2


def test_rig_hiyori_no_uv_config():
    # dev rig skips UV transfer
    assert RIG_HIYORI.uv_config is None


def test_rigconfig_dataclass():
    from pathlib import Path
    rc = RigConfig(
        name="test",
        model_dir=Path("/tmp"),
        moc3_path=Path("/tmp/test.moc3"),
        model3_json_path=Path("/tmp/test.model3.json"),
        textures=[Path("/tmp/tex.png")],
        param_ids=["ParamA", "ParamB"],
    )
    assert rc.param_count == 2
