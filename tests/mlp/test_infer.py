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
