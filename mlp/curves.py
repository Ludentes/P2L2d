"""Response curves: post-MLP exaggeration layer for VTuber animation.

Maps MLP output (in param native range) → final param value sent to VTube Studio.
Applied after inference, not baked into the model — tunable without retraining.

Usage:
    curves = ResponseCurveSet.from_dict(HIYORI_DEFAULTS)
    # or: curves = ResponseCurveSet.from_toml("mlp/curves/hiyori.toml")
    final_params = curves.apply(raw_params, param_ids)

Curve types:
    gamma:        out = lo + (hi - lo) * ((in - lo) / (hi - lo)) ** gamma
                  gamma < 1 lifts midtones (more expressive)
                  gamma > 1 compresses midtones (more subtle)
    linear_scale: out = centre + (in - centre) * scale
                  scale > 1 exaggerates, < 1 dampens
    identity:     out = in (no-op, the default for unlisted params)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-reuse-def]


# ── Curve definitions ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GammaCurve:
    """Gamma lift/compress within [lo, hi]."""
    gamma: float
    lo: float
    hi: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        span = self.hi - self.lo
        if span == 0:
            return values.copy()
        norm = np.clip((values - self.lo) / span, 0.0, 1.0)
        return (self.lo + span * norm ** self.gamma).astype(np.float32)


@dataclass(frozen=True)
class LinearScaleCurve:
    """Scale around a centre point."""
    scale: float
    centre: float
    lo: float
    hi: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        out = self.centre + (values - self.centre) * self.scale
        return np.clip(out, self.lo, self.hi).astype(np.float32)


_Curve = GammaCurve | LinearScaleCurve


def _parse_curve(d: dict[str, Any]) -> _Curve:
    kind = d["type"]
    if kind == "gamma":
        return GammaCurve(gamma=d["gamma"], lo=d.get("lo", 0.0), hi=d.get("hi", 1.0))
    if kind == "linear_scale":
        return LinearScaleCurve(
            scale=d["scale"],
            centre=d.get("centre", 0.0),
            lo=d["lo"],
            hi=d["hi"],
        )
    raise ValueError(f"Unknown curve type: {kind!r}")


# ── ResponseCurveSet ──────────────────────────────────────────────────────────

@dataclass
class ResponseCurveSet:
    """Per-param post-MLP response curves.

    Params not in `curves` pass through unchanged (identity).
    """
    curves: dict[str, _Curve] = field(default_factory=dict)

    def apply(self, params: np.ndarray, param_ids: list[str]) -> np.ndarray:
        """Apply per-param curves.

        Args:
            params:    float32 array (n_params,) or (B, n_params)
            param_ids: list of param IDs matching last axis of params

        Returns:
            float32 array same shape as params
        """
        out = params.astype(np.float32).copy()
        for pid, curve in self.curves.items():
            if pid not in param_ids:
                continue
            i = param_ids.index(pid)
            if params.ndim == 1:
                out[i] = curve.apply(out[i : i + 1])[0]
            else:
                out[:, i] = curve.apply(out[:, i])
        return out

    @classmethod
    def from_dict(cls, spec: dict[str, dict[str, Any]]) -> "ResponseCurveSet":
        """Build from a dict of {param_id: curve_spec_dict}."""
        return cls(curves={pid: _parse_curve(d) for pid, d in spec.items()})

    @classmethod
    def from_toml(cls, path: Path) -> "ResponseCurveSet":
        """Load from a TOML file.

        Expected format:
            [params.ParamEyeLOpen]
            type = "gamma"
            gamma = 0.7
            lo = 0.0
            hi = 1.0
        """
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.from_dict(data.get("params", {}))

    @classmethod
    def from_json(cls, path: Path) -> "ResponseCurveSet":
        """Load from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data.get("params", data))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for pid, curve in self.curves.items():
            if isinstance(curve, GammaCurve):
                out[pid] = {"type": "gamma", "gamma": curve.gamma, "lo": curve.lo, "hi": curve.hi}
            elif isinstance(curve, LinearScaleCurve):
                out[pid] = {
                    "type": "linear_scale",
                    "scale": curve.scale,
                    "centre": curve.centre,
                    "lo": curve.lo,
                    "hi": curve.hi,
                }
        return out


# ── Hiyori defaults ───────────────────────────────────────────────────────────
# Head pose: linear 1:1 (already at native scale, no exaggeration needed)
# Eyes/mouth: gamma < 1 lifts midtones for more expressive animation
# Brows: linear_scale for symmetric exaggeration around 0
HIYORI_DEFAULTS: dict[str, dict[str, Any]] = {
    "ParamEyeLOpen":   {"type": "gamma", "gamma": 0.7, "lo": 0.0, "hi": 1.0},
    "ParamEyeROpen":   {"type": "gamma", "gamma": 0.7, "lo": 0.0, "hi": 1.0},
    "ParamEyeLSmile":  {"type": "gamma", "gamma": 0.8, "lo": 0.0, "hi": 1.0},
    "ParamEyeRSmile":  {"type": "gamma", "gamma": 0.8, "lo": 0.0, "hi": 1.0},
    "ParamMouthOpenY": {"type": "gamma", "gamma": 0.6, "lo": 0.0, "hi": 1.0},
    "ParamMouthForm":  {
        "type": "linear_scale", "scale": 1.3, "centre": 0.0, "lo": -2.0, "hi": 1.0
    },
    "ParamBrowLY":     {
        "type": "linear_scale", "scale": 1.2, "centre": 0.0, "lo": -1.0, "hi": 1.0
    },
    "ParamBrowRY":     {
        "type": "linear_scale", "scale": 1.2, "centre": 0.0, "lo": -1.0, "hi": 1.0
    },
}
