"""End-to-end: features → template predictor → curves → manifest remap → rig params."""
from pathlib import Path

import numpy as np
import pytest


def test_full_pipeline():
    """58-d features → template MLP → curves → manifest → rig param dict."""
    pytest.importorskip("torch")
    from templates.loader import load_template
    from rig.manifest import load_manifest
    from mlp.infer import TemplatePredictor

    template = load_template("humanoid-anime")
    if not template.model_path.exists():
        pytest.skip("template model.pt not present")

    manifest = load_manifest(Path("manifests/hiyori.toml"))
    assert manifest.template_name == template.name

    predictor = TemplatePredictor(template)

    features = np.zeros(58, dtype=np.float32)
    template_params = predictor.predict_with_curves(features)

    rig_params = manifest.remap(template_params)

    assert len(rig_params) == 13
    assert "ParamAngleX" in rig_params
    assert "ParamMouthOpenY" in rig_params
    assert "ParamCheek" in rig_params

    for name, value in rig_params.items():
        assert np.isfinite(value), f"{name} is not finite: {value}"
