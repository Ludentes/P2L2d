import numpy as np
import pytest
import torch

from mlp.model import CartoonAliveMLP


def test_output_shape():
    model = CartoonAliveMLP(n_params=74)
    x = torch.randn(1, 956)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 74)


def test_batch_output_shape():
    model = CartoonAliveMLP(n_params=74)
    x = torch.randn(8, 956)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (8, 74)


def test_arbitrary_param_count():
    model = CartoonAliveMLP(n_params=107)
    x = torch.randn(1, 956)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 107)


def test_checkpoint_loads():
    from pathlib import Path
    import torch
    ckpt_path = Path("mlp/checkpoints/hiyori_v2/model.pt")
    if not ckpt_path.exists():
        pytest.skip("checkpoint not present")
    model = CartoonAliveMLP(n_params=74)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    x = torch.randn(1, 956)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 74)
    assert not torch.isnan(y).any()


def test_deterministic_in_eval():
    model = CartoonAliveMLP(n_params=74)
    model.eval()
    x = torch.randn(1, 956)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.allclose(y1, y2)
