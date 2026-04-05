"""Runtime inference — landmarks → Live2D parameter values.

Loads the checkpoint once at startup; predict() runs in <16ms on CPU.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from mlp.model import CartoonAliveMLP
from rig.config import RigConfig

_DEFAULT_CHECKPOINT = Path(__file__).parent / "checkpoints" / "hiyori_v2" / "model.pt"


class Predictor:
    """Wraps a loaded CartoonAliveMLP for repeated inference calls."""

    def __init__(self, rig: RigConfig, checkpoint: Path = _DEFAULT_CHECKPOINT) -> None:
        self._rig = rig
        self._model = CartoonAliveMLP(n_params=rig.param_count)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self._model.load_state_dict(state)
        self._model.eval()

    def predict(self, landmarks: np.ndarray) -> dict[str, float]:
        """Map 478 (x,y) landmarks → dict of {param_id: value}.

        Args:
            landmarks: float32 array of shape (478, 2) or (956,).
        Returns:
            Dict mapping each param_id to its predicted value.
        """
        flat = landmarks.reshape(1, 956).astype(np.float32)
        x = torch.from_numpy(flat)
        with torch.no_grad():
            y = self._model(x)
        values = y.squeeze(0).numpy()
        return dict(zip(self._rig.param_ids, values.tolist()))

    def benchmark(self, n: int = 200) -> float:
        """Return mean inference time in ms over n runs."""
        dummy = np.zeros((478, 2), dtype=np.float32)
        # warm-up
        for _ in range(10):
            self.predict(dummy)
        start = time.perf_counter()
        for _ in range(n):
            self.predict(dummy)
        return (time.perf_counter() - start) / n * 1000


def load_predictor(rig: RigConfig, checkpoint: Path = _DEFAULT_CHECKPOINT) -> Predictor:
    """Convenience constructor — returns a ready Predictor."""
    return Predictor(rig, checkpoint)
