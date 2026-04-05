"""Rig configuration — all rig-specific constants in one place.

Subsystems (MLP, pipeline, runtime) take a RigConfig and never hard-code
model names, param counts, or texture paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_VTS_MODELS = Path.home() / (
    ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common"
    "/VTube Studio/VTube Studio_Data/StreamingAssets/Live2DModels"
)

# Hiyori's 74 parameter IDs in cdi3.json order.
# The MLP output vector maps to these indices 1-to-1.
_HIYORI_PARAMS: list[str] = [
    "ParamPositionX", "ParamPositionY", "ParamPositionZ",
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamCheek",
    "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
    "ParamBreath", "ParamShoulder", "ParamStep",
    "ParamEyeLOpen", "ParamEyeLSmile",
    "ParamEyeROpen", "ParamEyeRSmile",
    "ParamEyeBallX", "ParamEyeBallY",
    "ParamBrowLY", "ParamBrowRY", "ParamBrowLX", "ParamBrowRX",
    "ParamBrowLAngle", "ParamBrowRAngle",
    "ParamBrowLForm", "ParamBrowRForm",
    "ParamMouthForm", "ParamMouthOpenY", "ParamMouthX",
    "ParamArmLA", "ParamArmRA", "ParamArmLB", "ParamArmRB",
    "ParamHandLB", "ParamHandRB", "ParamHandL", "ParamHandR",
    "ParamBustY",
    "ParamHairAhoge", "ParamHairFront", "ParamHairBack",
    "ParamSideupRibbon", "ParamRibbon",
    "ParamSkirt", "ParamSkirt2",
    "Param_Angle_Rotation_1_ArtMesh62", "Param_Angle_Rotation_2_ArtMesh62",
    "Param_Angle_Rotation_3_ArtMesh62", "Param_Angle_Rotation_4_ArtMesh62",
    "Param_Angle_Rotation_5_ArtMesh62", "Param_Angle_Rotation_6_ArtMesh62",
    "Param_Angle_Rotation_7_ArtMesh62",
    "Param_Angle_Rotation_1_ArtMesh61", "Param_Angle_Rotation_2_ArtMesh61",
    "Param_Angle_Rotation_3_ArtMesh61", "Param_Angle_Rotation_4_ArtMesh61",
    "Param_Angle_Rotation_5_ArtMesh61", "Param_Angle_Rotation_6_ArtMesh61",
    "Param_Angle_Rotation_7_ArtMesh61",
    "Param_Angle_Rotation_1_ArtMesh55", "Param_Angle_Rotation_2_ArtMesh55",
    "Param_Angle_Rotation_3_ArtMesh55", "Param_Angle_Rotation_4_ArtMesh55",
    "Param_Angle_Rotation_5_ArtMesh55", "Param_Angle_Rotation_6_ArtMesh55",
    "Param_Angle_Rotation_7_ArtMesh55",
    "Param_Angle_Rotation_1_ArtMesh54", "Param_Angle_Rotation_2_ArtMesh54",
    "Param_Angle_Rotation_3_ArtMesh54", "Param_Angle_Rotation_4_ArtMesh54",
    "Param_Angle_Rotation_5_ArtMesh54", "Param_Angle_Rotation_6_ArtMesh54",
    "Param_Angle_Rotation_7_ArtMesh54",
]


@dataclass
class RigConfig:
    name: str
    model_dir: Path
    moc3_path: Path
    model3_json_path: Path
    textures: list[Path]
    param_ids: list[str]
    # Path to model_configuration.json (Textoon UV coord table).
    # None = skip UV transfer stage (dev/testing mode).
    uv_config: Path | None = None

    @property
    def param_count(self) -> int:
        return len(self.param_ids)


def _hiyori_dir() -> Path:
    return _VTS_MODELS / "hiyori_vts"


RIG_HIYORI = RigConfig(
    name="hiyori",
    model_dir=_hiyori_dir(),
    moc3_path=_hiyori_dir() / "hiyori.moc3",
    model3_json_path=_hiyori_dir() / "hiyori.model3.json",
    textures=[
        _hiyori_dir() / "hiyori.2048" / "texture_00.png",
        _hiyori_dir() / "hiyori.2048" / "texture_01.png",
    ],
    param_ids=_HIYORI_PARAMS,
    uv_config=None,
)
