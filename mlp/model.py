"""CartoonAlive MLP — landmarks → Live2D parameters.

Architecture (matches hiyori_v2 checkpoint):
  InputNorm(956)
  Linear(956→512) + LayerNorm(512) + GELU
  Linear(512→256) + LayerNorm(256) + GELU  +  skip Linear(512→256)
  Linear(256→128) + LayerNorm(128) + GELU
  Linear(128→N)
  OutputDenorm(N)

Input: 478 MediaPipe FaceMesh landmarks flattened to (x, y) → 956-d vector.
Output: N Live2D parameter values in rig_config.param_ids order.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _Norm(nn.Module):
    """Stores mean/std as buffers; applies (x - mean) / std or reverse."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std.clamp(min=1e-6)

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean


class CartoonAliveMLP(nn.Module):
    """4-layer MLP with one residual skip connection."""

    def __init__(self, n_params: int) -> None:
        super().__init__()
        self.input_norm = _Norm(956)

        self.fc1 = nn.Linear(956, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.skip = nn.Linear(512, 256)   # residual from fc1 output → fc2 output

        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        self.fc4 = nn.Linear(128, n_params)
        self.output_denorm = _Norm(n_params)

        self._act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_norm(x)

        h1 = self._act(self.ln1(self.fc1(x)))
        h2 = self._act(self.ln2(self.fc2(h1)) + self.skip(h1))
        h3 = self._act(self.ln3(self.fc3(h2)))
        out = self.fc4(h3)

        return self.output_denorm.inverse(out)
