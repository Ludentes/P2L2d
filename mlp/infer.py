"""Runtime inference — landmarks/features → Live2D parameter values.

Two predictor classes:
  - Predictor: legacy, hardcoded to RigConfig + 956-d landmarks
  - TemplatePredictor: template-aware, loads from Template, any input_dim
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from mlp.model import CartoonAliveMLP
from rig.config import RigConfig
from templates.loader import Template

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


class TemplatePredictor:
    """Template-aware inference: loads model from template, returns template param names."""

    def __init__(self, template: Template, device: str = "cpu") -> None:
        self._template = template
        self._device = device
        ckpt = torch.load(template.model_path, weights_only=False, map_location=device)
        self._model = CartoonAliveMLP(
            n_params=ckpt["n_params"],
            input_dim=ckpt["input_dim"],
        )
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._model.to(device)
        self._param_names: list[str] = ckpt["param_names"]

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Map feature vector -> dict of {template_param_name: value}.

        Args:
            features: float32 array of shape (input_dim,).
        """
        x = torch.from_numpy(features.reshape(1, -1).astype(np.float32)).to(self._device)
        with torch.no_grad():
            y = self._model(x)
        values = y.squeeze(0).cpu().numpy()
        return dict(zip(self._param_names, values.tolist()))

    def predict_with_curves(self, features: np.ndarray) -> dict[str, float]:
        """predict() + apply template response curves."""
        raw = self.predict(features)
        raw_arr = np.array([raw[n] for n in self._param_names], dtype=np.float32)
        curved_arr = self._template.curves.apply(raw_arr, self._param_names)
        return dict(zip(self._param_names, curved_arr.tolist()))

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch inference. Returns (B, n_params) float32 array."""
        x = torch.from_numpy(features.astype(np.float32)).to(self._device)
        with torch.no_grad():
            y = self._model(x)
        return y.cpu().numpy()
