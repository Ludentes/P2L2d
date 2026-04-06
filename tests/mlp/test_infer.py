import numpy as np
import pytest

from rig.config import RIG_HIYORI


def test_predictor_output_keys():
    pytest.importorskip("torch")
    from pathlib import Path
    ckpt = Path("mlp/checkpoints/hiyori_v2/model.pt")
    if not ckpt.exists():
        pytest.skip("checkpoint not present")
    from mlp.infer import Predictor
    p = Predictor(RIG_HIYORI, ckpt)
    landmarks = np.zeros((478, 2), dtype=np.float32)
    result = p.predict(landmarks)
    assert set(result.keys()) == set(RIG_HIYORI.param_ids)


def test_predictor_accepts_flat_input():
    pytest.importorskip("torch")
    from pathlib import Path
    ckpt = Path("mlp/checkpoints/hiyori_v2/model.pt")
    if not ckpt.exists():
        pytest.skip("checkpoint not present")
    from mlp.infer import Predictor
    p = Predictor(RIG_HIYORI, ckpt)
    landmarks_flat = np.zeros(956, dtype=np.float32)
    result = p.predict(landmarks_flat)
    assert len(result) == 74


def test_predictor_benchmark_reasonable():
    pytest.importorskip("torch")
    from pathlib import Path
    ckpt = Path("mlp/checkpoints/hiyori_v2/model.pt")
    if not ckpt.exists():
        pytest.skip("checkpoint not present")
    from mlp.infer import Predictor
    p = Predictor(RIG_HIYORI, ckpt)
    ms = p.benchmark(n=50)
    assert ms < 100.0, f"Inference too slow: {ms:.1f}ms"


# ── TemplatePredictor ────────────────────────────────────────────────────────


def test_template_predictor_output_keys():
    pytest.importorskip("torch")
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    if not t.model_path.exists():
        pytest.skip("template model.pt not present")

    from mlp.infer import TemplatePredictor

    p = TemplatePredictor(t)
    features = np.zeros(58, dtype=np.float32)
    result = p.predict(features)
    assert set(result.keys()) == set(t.schema.names)
    assert "AngleX" in result


def test_template_predictor_with_curves():
    pytest.importorskip("torch")
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    if not t.model_path.exists():
        pytest.skip("template model.pt not present")

    from mlp.infer import TemplatePredictor

    p = TemplatePredictor(t)
    features = np.zeros(58, dtype=np.float32)
    raw = p.predict(features)
    curved = p.predict_with_curves(features)
    assert set(curved.keys()) == set(raw.keys())


def test_template_predictor_accepts_batch():
    pytest.importorskip("torch")
    from templates.loader import load_template

    t = load_template("humanoid-anime")
    if not t.model_path.exists():
        pytest.skip("template model.pt not present")

    from mlp.infer import TemplatePredictor

    p = TemplatePredictor(t)
    features = np.zeros((4, 58), dtype=np.float32)
    result = p.predict_batch(features)
    assert result.shape == (4, 13)
