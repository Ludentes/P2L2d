"""Integration test: portrait -> output dir via full pipeline.

Requires Hiyori model files and ComfyUI running at http://127.0.0.1:8188.
Automatically skipped if either is absent.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HIYORI_MODEL3 = Path(
    "/home/newub/.var/app/com.valvesoftware.Steam/.local/share/Steam"
    "/steamapps/common/VTube Studio/VTube Studio_Data/StreamingAssets"
    "/Live2DModels/hiyori_vts/hiyori.model3.json"
)

skip_no_model = pytest.mark.skipif(
    not HIYORI_MODEL3.exists(), reason="Hiyori model files not present"
)


async def _comfyui_running() -> bool:
    try:
        from comfyui.client import ComfyUIClient
        async with ComfyUIClient() as c:
            await c.health()
        return True
    except Exception:
        return False


@skip_no_model
async def test_full_pipeline_hiyori(tmp_path):
    """End-to-end: synthetic portrait -> output dir with modified atlases."""
    if not await _comfyui_running():
        pytest.skip("ComfyUI not running at http://127.0.0.1:8188")

    from pipeline.atlas_config import load_atlas_config
    from pipeline.run import run_portrait_to_rig
    from rig.config import RIG_HIYORI

    portrait_path = tmp_path / "portrait.jpg"
    Image.new("RGB", (512, 512), color=(220, 170, 140)).save(portrait_path)

    atlas_cfg = load_atlas_config(Path("manifests/hiyori_atlas.toml"))
    out_dir = tmp_path / "output"

    result = await run_portrait_to_rig(
        portrait_path=portrait_path,
        rig_config=RIG_HIYORI,
        atlas_cfg=atlas_cfg,
        output_dir=out_dir,
        template_name="humanoid-anime",
    )

    assert result.is_dir()
    assert (result / "hiyori.moc3").exists()
    assert (result / "hiyori.model3.json").exists()
    assert (result / "hiyori.2048" / "texture_00.png").exists()
    assert (result / "hiyori.2048" / "texture_01.png").exists()

    # Atlas was modified (not identical to original)
    original = np.array(Image.open(RIG_HIYORI.textures[0]).convert("RGBA"))
    modified = np.array(Image.open(result / "hiyori.2048" / "texture_00.png").convert("RGBA"))
    assert not np.array_equal(original, modified), (
        "Modified atlas identical to original — swap may not have run"
    )

    print(f"Output at: {result}")
